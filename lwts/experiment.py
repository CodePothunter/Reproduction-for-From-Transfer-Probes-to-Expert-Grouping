"""把数据、分配和训练后端连接起来：一个区间（round）的执行与可逆探针。

一个区间的流程：

    next_allocation()  按方法给出连续分配 Q_t（uniform 或预先给定的 fixed 块矩阵）
    → integer_counts   转成整数块配额，行列总量分别锁住模型预算与来源曝光
    → 逐块训练         每块来源纯净，样本从共享的来源流里按游标取

两个方法（METHODS）共享同一套来源流、模型预算与块规则，只有分配不同：

  uniform  每个专家在每个来源上分到相同份额（W 的均匀热身区间就是它）。
  fixed    直接执行一份预先给定的整数块矩阵（候选 C0–C3 的训练区间）。

另有一个与在线循环无关的测量入口：probe() 在独立的 audit split 上测「在来源
k 上临时训练若干块」前后整行能力的变化，并在结束后回滚全部状态。单源迁移
探针（pipeline/probe_transfer.py）就是逐 cell 调它。

数据隔离：train 只进训练，audit 只给 probe；feedback 切分只做存在性校验
（manifest 契约的一部分），route 与 test 本模块完全不碰。
"""
import hashlib
import math
import random
import statistics
from copy import deepcopy
from pathlib import Path

import numpy as np

from .allocation import check_marginals, integer_counts, uniform
from .backend import TrainingBackend
from .storage import canonical, load, save
from .utility import paired_return

METHODS = ("uniform", "fixed")
# tau 被 probe 的 portfolio_return 使用；eta / gamma / rho 在 uniform 与 fixed 下
# 不被读取，但仍按原配置要求显式给出（gamma 另有取值域检查），这样原始命令与
# 配置在本包里按同样的规则通过校验。
REQUIRED = ("tokens_per_interval", "block_tokens", "tau", "eta", "gamma", "rho")


def _digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _apportion(weights, total):
    """把整数 total 按 weights 分配，最大余数法，和恰好等于 total。"""
    raw = [w * total for w in weights]
    counts = [int(math.floor(x)) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


class Experiment:
    """config["experiment"] 需要的字段见 REQUIRED；weights/预算缺省为均匀。

    REQUIRED 里的每一项都必须显式填写，缺失会直接报错——不把未给定的值当作
    "已验证默认值"。
    """

    def __init__(self, config: dict, manifest: dict, backend: TrainingBackend):
        self.config = config
        self.manifest = manifest
        self.backend = backend
        self.round = 0
        self.cursors = {}
        self.allocation = None

        settings = config.get("experiment")
        if not isinstance(settings, dict):
            raise ValueError("config 需要 experiment 段")
        self.method = settings.get("method")
        if self.method not in METHODS:
            raise ValueError(f"method 必须是 {METHODS} 之一")
        self.models = settings.get("models")
        self.sources = settings.get("sources")
        for name, value in (("models", self.models), ("sources", self.sources)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        for name in REQUIRED:
            if settings.get(name) is None:
                raise ValueError(f"config.experiment.{name} 尚未填写；它必须显式给出")
        self.tokens_per_interval = settings["tokens_per_interval"]
        self.block_tokens = settings["block_tokens"]
        self.tau = settings["tau"]
        self.eta = settings["eta"]
        self.gamma = settings["gamma"]
        self.rho = settings["rho"]
        if not 0 < self.gamma < 1:
            raise ValueError("gamma 必须落在 (0, 1)")
        if self.block_tokens < 1 or self.tokens_per_interval < self.block_tokens:
            raise ValueError("tokens_per_interval 必须不小于 block_tokens 且均为正")
        # block_tokens 是**下界不是目标**：_take_block 累加整条记录直到跨过阈值。
        # 记录长度比它大时，名义值完全无效而且不会报错——例如 block_tokens=512
        # 在长解答语料上实测得到 5045 token 的块（9.9×）。这两个开关把那种
        # 静默失配变成构造期的显式错误，见 _block_plan。
        self.block_tolerance = settings.get("block_tolerance", 1.25)
        self.allow_degenerate_blocks = bool(settings.get("allow_degenerate_blocks", False))
        self.warmup_rounds = settings.get("warmup_rounds", 1)
        self.seed = settings.get("seed", 0)
        # diagnostic 模式恢复出来的实验只许读状态，不许再推进轨迹（见 resume）
        self._diagnostic_only = False

        self.weights = settings.get("weights") or [1.0 / self.sources] * self.sources
        self.model_budget = settings.get("model_budget") or [1.0 / self.models] * self.models
        self.source_exposure = settings.get("source_exposure") or [1.0 / self.sources] * self.sources
        if len(self.weights) != self.sources or len(self.model_budget) != self.models \
                or len(self.source_exposure) != self.sources:
            raise ValueError("weights/model_budget/source_exposure 的长度与 models/sources 不符")
        # fixed：直接执行一份预先给定的整数块分配（候选实际收益实验用）。
        # 整数块必须精确满足行列边际，否则不同候选的预算不可比。
        self.fixed_blocks = settings.get("fixed_blocks")
        if self.method == "fixed":
            if self.fixed_blocks is None:
                raise ValueError("fixed 方法需要 config.experiment.fixed_blocks（整数块矩阵）")
            fb = self.fixed_blocks
            if len(fb) != self.models or any(len(r) != self.sources for r in fb) \
                    or any(type(v) is not int or v < 0 for r in fb for v in r):
                raise ValueError("fixed_blocks 形状或取值不合法")

        self.banks = self._banks()
        self.streams = self._streams()
        self.cursors = {k: 0 for k in range(self.sources)}
        self.plan = self._block_plan()
        if self.plan["problems"] and not self.allow_degenerate_blocks:
            raise ValueError(
                "块尺度与配置不符，实验将无法解读：\n  - "
                + "\n  - ".join(self.plan["problems"])
                + "\n实测块大小见 experiment.plan['measured']。若确实要在这个退化"
                  "区间里跑（例如只验证机制的单元测试），显式设置"
                  " config.experiment.allow_degenerate_blocks = true。")
        # 构造期不做任何能力评估：这张表只在 probe 里给 portfolio_return 提供
        # 「其余专家」的行，恒为零。因此 portfolio_return 是本专家的加权自身
        # 增益，不是组合口径；迁移矩阵、逐条翻转与 McNemar 都不读它。
        self.competence = [[0.0] * self.sources for _ in range(self.models)]
        self.logs = []

    # --- 数据 ---------------------------------------------------------------

    def _split(self, split):
        rows = [r for r in self.manifest["records"] if r["split"] == split]
        grouped = {k: [] for k in range(self.sources)}
        for row in rows:
            if row["source"] in grouped:
                grouped[row["source"]].append(row)
        return grouped

    def _banks(self):
        """audit 只给 probe；feedback 只做存在性校验。route 与 test 一律不取。"""
        banks = {}
        for split in ("feedback", "audit"):
            grouped = self._split(split)
            if any(not rows for rows in grouped.values()):
                raise ValueError(f"{split} split 存在空来源，无法做逐来源测量")
            banks[split] = {k: sorted(rows, key=lambda r: r["id"]) for k, rows in grouped.items()}
        return banks

    def _streams(self):
        """各来源的训练流。顺序只由 seed 与 source 决定，与 method 无关，
        因此所有方法看到完全相同的数据流——比较分配而不是比较数据的前提。"""
        grouped = self._split("train")
        streams = {}
        for k in range(self.sources):
            rows = sorted(grouped[k], key=lambda r: r["id"])
            if not rows:
                raise ValueError(f"来源 {k} 没有训练数据")
            random.Random(f"{self.seed}:{k}").shuffle(rows)
            streams[k] = rows
        return streams

    def _block_plan(self):
        """实测块尺度与分配粒度，并挑出会让结果无法解读的两种配置。

        为什么必须实测：块是原子的执行单位，它的大小由**记录长度**决定，
        不由 block_tokens 决定。_take_block 累加整条记录直到跨过阈值，所以

            实际块大小 ≈ block_tokens + E[记录长度的一半]

        当 block_tokens 落在记录长度以下时，这个近似退化成"一条记录"，配置值
        彻底失效。例如长解答语料（max_tokens=16384）上实测：

            block_tokens 2048 → 5949（2.9×）   1024 → 5341（5.2×）   512 → 5045（9.9×）

        两个致命项（problems）：
          1. 实测均值超出名义值 block_tolerance 倍——名义值是虚构的，据此推出的
             块数与成本估计全都错。
          2. 每区间块数少于 cell 数——连"每个 cell 至少一块"都表达不了，分配矩阵
             无法被整数配额近似，方法之间的差异被取整抹平。

        两个警示项（warnings）不阻断运行，但会写进日志：分配粒度过粗，以及
        某来源的数据流在一个区间内被反复回绕。

        逐来源统计而不是只看总体：来源之间的记录长度可以差一个数量级（数学
        长解答与短代码参考解），混在一起平均会把两边的失配都掩盖掉。因此容限
        按**最差的来源**判定。
        """
        blocks = self.tokens_per_interval // self.block_tokens
        cells = self.models * self.sources
        by_source, sizes = {}, []
        for k, stream in self.streams.items():
            position, cut, records = 0, [], []
            while position < len(stream):
                picked, position = self._take_block(k, position)
                cut.append(sum(r["completion_tokens"] for r in picked))
                records.append(len(picked))
            sizes.extend(cut)
            share = blocks * self.source_exposure[k]
            by_source[k] = {
                "records": len(stream),
                "tokens": sum(r["completion_tokens"] for r in stream),
                "blocks_per_full_pass": len(cut),
                "mean": statistics.fmean(cut),
                "median": statistics.median(cut),
                "records_per_block": statistics.fmean(records),
                "ratio_to_nominal": statistics.fmean(cut) / self.block_tokens,
                # 该来源每区间被分到的块数 ÷ 整条流的块数：>1 表示区间内就回绕了
                "passes_per_interval": share / len(cut)}

        worst = max(by_source, key=lambda k: by_source[k]["ratio_to_nominal"])
        ratio = by_source[worst]["ratio_to_nominal"]
        # 一块占某专家该区间预算的比例：_apportion 把块按 model_budget 分行
        granularity = self.models / blocks
        measured = {"mean": statistics.fmean(sizes), "median": statistics.median(sizes),
                    "min": min(sizes), "max": max(sizes),
                    "worst_source": worst, "ratio_to_nominal": ratio,
                    "by_source": by_source}

        problems, warnings = [], []
        if ratio > self.block_tolerance:
            entry = by_source[worst]
            problems.append(
                f"来源 {worst} 的实测块 {entry['mean']:.0f} token 是 block_tokens="
                f"{self.block_tokens} 的 {ratio:.1f} 倍（容限 {self.block_tolerance}）。"
                f"块至少是一条记录，而该来源每块只装得下 {entry['records_per_block']:.1f} "
                f"条；要让名义值生效，block_tokens 必须显著大于记录长度")
        if blocks < cells:
            problems.append(
                f"每区间只有 {blocks} 块，少于 {cells} 个 cell（{self.models}×"
                f"{self.sources}）。整数配额连每个 cell 一块都给不出，分配方法之间"
                f"的差异会被取整抹平")
        if granularity > 1.0 / (4 * self.sources):
            warnings.append(
                f"分配粒度 {granularity:.1%}：一块就是某专家区间预算的 "
                f"{granularity:.1%}，方法无法表达比这更细的偏好")
        recycled = {k: v["passes_per_interval"] for k, v in by_source.items()
                    if v["passes_per_interval"] > 1}
        if recycled:
            warnings.append(
                "来源数据在一个区间内就被回绕重训："
                + "、".join(f"来源 {k} 走 {n:.1f} 遍" for k, n in sorted(recycled.items()))
                + "。来源曝光是按 token 均分的，而各来源的 token 总量可以差一个"
                  "数量级，等额 token 并不等于等额 epoch")
        return {"block_tokens": self.block_tokens, "blocks_per_interval": blocks,
                "cells": cells, "granularity": granularity, "measured": measured,
                "problems": problems, "warnings": warnings}

    def _take_block(self, source, cursor=None):
        """按 block_tokens 从来源流里取一块；流用尽后回绕，不静默停训。"""
        stream = self.streams[source]
        position = self.cursors[source] if cursor is None else cursor
        picked, tokens = [], 0
        while tokens < self.block_tokens and len(picked) < len(stream):
            record = stream[position % len(stream)]
            picked.append(record)
            tokens += record["completion_tokens"]
            position += 1
        if not picked:
            raise RuntimeError(f"来源 {source} 取不到训练块")
        return picked, position

    # --- 分配 ---------------------------------------------------------------

    def next_allocation(self):
        """uniform 与热身区间返回均匀分配；fixed 返回给定块矩阵的归一化。"""
        r, p = self.model_budget, self.source_exposure
        if self.method == "uniform" or self.round < self.warmup_rounds:
            return uniform(r, p)                      # 共同的均匀热身
        total = sum(map(sum, self.fixed_blocks))      # method == "fixed"
        return [[v / total for v in row] for row in self.fixed_blocks]

    def _state(self):
        """区间回滚用的完整状态。每个区间开头都会调一次，因此不能有重拷贝。"""
        return {"backend": self.backend.snapshot(), "cursors": dict(self.cursors),
                "allocation": deepcopy(self.allocation), "round": self.round,
                "logs": deepcopy(self.logs)}

    def _restore_state(self, state):
        """只恢复训练侧状态。

        较早写出的检查点在 state 里还带着其它键（例如每个区间的诊断量与成本
        账），它们不影响训练轨迹，这里一律不读。
        """
        self.backend.restore(state["backend"])
        # JSON 只有字符串键：从检查点恢复时 cursors 的来源编号会变成字符串，
        # 这里统一转回整数，内存回滚与磁盘恢复才是同一条路径。
        self.cursors = {int(k): v for k, v in state["cursors"].items()}
        self.allocation = deepcopy(state["allocation"])
        self.round = state["round"]
        self.logs = deepcopy(state["logs"])

    def run_round(self):
        """完成分配、执行和日志，成功后 round 加一。

        整个区间是原子的：任何一步失败都回滚到区间开始时的完整状态，包括后端
        参数、优化器、scheduler、RNG 与来源游标，绝不留下半个区间。
        """
        rollback = self._state()
        # 回滚快照用完即弃：真实后端每份是数百 MB 的张量文件，不回收就会一直
        # 堆积。检查点引用的快照走的是另一条路径。
        try:
            allocation = self.next_allocation()
            check_marginals(allocation, self.model_budget, self.source_exposure, tolerance=1e-7)
            blocks = self.tokens_per_interval // self.block_tokens
            counts = integer_counts(allocation,
                                    _apportion(self.model_budget, blocks),
                                    _apportion(self.source_exposure, blocks))
            if self.method == "fixed" and self.round >= self.warmup_rounds \
                    and counts != self.fixed_blocks:
                # 给定块的边际与预算不符时，integer_counts 会悄悄改成另一份分配——必须停
                raise ValueError(f"fixed_blocks 与执行的整数块不一致：{counts}")
            executed = []
            for i in range(self.models):
                for k in range(self.sources):
                    remaining = counts[i][k]
                    while remaining:
                        records, position = self._take_block(k)
                        log = self.backend.train_block(i, records)
                        self.cursors[k] = position
                        executed.append(log)
                        remaining -= 1
            self.allocation = allocation
            realized = [l["completion_tokens"] for l in executed]
            self.logs.append({"round": self.round, "blocks": counts,
                              "processed_tokens": sum(l["processed_tokens"] for l in executed),
                              "completion_tokens": sum(realized),
                              # 名义 block_tokens 不等于实际块大小，报实测值
                              "block_tokens_realized": {
                                  "mean": statistics.fmean(realized),
                                  "median": statistics.median(realized),
                                  "max": max(realized), "min": min(realized)},
                              "allocation": deepcopy(allocation)})
            self.round += 1
        except BaseException:
            self._restore_state(rollback)
            raise
        finally:
            self.backend.discard(rollback["backend"])

    def run(self, until: int):
        """已提供外层循环，避免实现忘记推进 round 时陷入无限循环。"""
        if self._diagnostic_only:
            # 诊断模式放宽了 config 摘要比对，续跑出来的轨迹无法解释。
            # 让它在这里硬停，比事后发现强。
            raise RuntimeError("以 diagnostic 模式恢复的实验不得继续 run()："
                               "它只用于在同一状态上做评估或临时更新与回滚")
        if type(until) is not int or until < self.round:
            raise ValueError("目标区间无效")
        while self.round < until:
            previous = self.round
            self.run_round()
            if self.round != previous + 1:
                raise RuntimeError("run_round 必须恰好推进一个区间")

    # --- 持久化 -------------------------------------------------------------

    def _identity(self):
        """配置与 manifest 的摘要；恢复时必须一致，否则状态无法解释。"""
        return {"config": _digest(self.config),
                "manifest": _digest([r["id"] for r in self.manifest["records"]]),
                "method": self.method, "models": self.models, "sources": self.sources}

    # 检查点格式 lwts-experiment-v2：JSON 旁边带一个 <检查点>.<后缀> 的 npz，
    # JSON 里记它的文件名、键名与 sha256。较早的检查点在这个 npz 里存过大向量
    # （每份约 2 GB）；本包写出的 npz 是空的，只为保持同一格式与校验。
    VECTOR_SUFFIX = ".vectors.npz"

    def checkpoint(self, path):
        """记录配置/manifest 摘要、后端状态、游标、分配与日志。

        写入顺序是先 npz 后 JSON：JSON 是原子替换的，只有它落盘才算检查点成立，
        因此不会出现 JSON 指向一个还没写完的 npz。
        """
        state = self._state()
        blob = Path(str(path) + self.VECTOR_SUFFIX)
        blob.parent.mkdir(parents=True, exist_ok=True)
        np.savez(blob)
        payload = {"format": "lwts-experiment-v2", "identity": self._identity(),
                   "vectors": {"path": blob.name, "keys": [],
                               "sha256": hashlib.sha256(blob.read_bytes()).hexdigest()},
                   "state": state}
        save(path, payload)
        return payload

    RESUME_MODES = ("strict", "fork", "diagnostic")

    def resume(self, path, mode="strict"):
        """检查配置与 manifest 是否匹配，再恢复训练侧状态。

        三种模式，宽松程度递增，但**放宽的理由各不相同**，不要混用：

        strict      同一条轨迹续跑。identity 必须逐项相同。

        fork        各候选从**同一个热身状态**分叉。method（以及含 method 的 config
                    摘要）按设计就该不同，只比对 manifest / models / sources。
                    但必须挡住一件事：从**热身之后**的检查点分叉。热身期所有方法都走
                    uniform，状态可共享；一旦进入决策区间，状态里就带上了源方法的
                    分配决定，另一个方法接着跑等于继承了别人的决策，而且**不报任何错**。
                    所以 fork 只允许 round <= warmup_rounds。

        diagnostic  在**同一个状态**上做评估或临时更新再回滚（test 评价用它读取
                    训练完的候选）。它读的是某方法跑完的状态——那是诊断的**对象**，
                    不是被冒充的共同起点，所以 fork 的轮次限制不适用。同样只比对
                    manifest / models / sources。
                    ⚠ 用这个模式恢复出来的实验**不得继续 run()**：它的 config 摘要
                    与检查点不一致，续跑出来的轨迹无法解释。

        npz 的存在性、sha256 与键集仍按检查点索引核对，但其中的数组不被载入：
        训练侧状态全部在后端快照、游标与日志里。
        """
        if mode not in self.RESUME_MODES:
            raise ValueError(f"mode 必须是 {self.RESUME_MODES} 之一")
        payload = load(path)
        if not isinstance(payload, dict) or payload.get("format") != "lwts-experiment-v2":
            raise ValueError("不支持的检查点格式")
        identity, expected = payload.get("identity"), self._identity()
        if mode == "strict":
            if identity != expected:
                raise ValueError("检查点与当前配置或 manifest 不一致，拒绝恢复")
        else:
            shared = ("manifest", "models", "sources")
            if not isinstance(identity, dict) or \
                    {k: identity.get(k) for k in shared} != {k: expected[k] for k in shared}:
                raise ValueError(f"{mode} 要求 manifest/models/sources 一致，拒绝恢复")
            if mode == "fork":
                round_at = payload.get("state", {}).get("round", 0)
                if round_at > self.warmup_rounds:
                    raise ValueError(
                        f"检查点停在第 {round_at} 轮，已超出热身的 {self.warmup_rounds} 轮："
                        f"该状态里带着源方法的分配决定，分叉会让本方法继承别人的决策")
            else:
                self._diagnostic_only = True
        index = payload["vectors"]
        blob = Path(path).parent / index["path"]
        if not blob.exists():
            raise FileNotFoundError(f"检查点的 npz 文件缺失：{blob}")
        if hashlib.sha256(blob.read_bytes()).hexdigest() != index["sha256"]:
            raise ValueError("检查点的 npz 文件摘要不匹配：已被修改或损坏")
        with np.load(blob) as arrays:
            if sorted(arrays.files) != index["keys"]:
                raise ValueError("检查点的 npz 文件的键与检查点不符")
        self._restore_state(payload["state"])
        return self.round

    # --- 探针 ---------------------------------------------------------------

    def audit_baseline(self, model_id):
        """测一次 audit 行，连同它所用的随机流起点一起返回，供多次 probe 共用。

        probe 结束时会回滚全部状态，所以同一个专家的每一行 probe 面对的 before
        状态**完全相同**。各测一次是纯浪费：不但多花一整行生成，还给每一行各自
        引入一份独立噪声。

        随机流起点一起带出来，是为了让每一行的 after 都能复位到同一段随机数，
        见 backend.evaluation_rng。⚠ 真实后端在 GPU 上采样时，这个复位不生效
        （见 probe 里的说明）。
        """
        if type(model_id) is not int or not 0 <= model_id < self.models:
            raise ValueError("model_id 越界")
        rng = self.backend.evaluation_rng()
        row, detail = self._audit_row(model_id)
        return {"model": model_id, "row": row, "detail": detail, "rng": rng}

    def probe(self, model_id, source_id, *, blocks=1, baseline=None):
        """临时更新并独立审计，正常结束和异常路径均需恢复全部状态。

        在独立的 audit split 上测前后能力，得到「在来源 source_id 上训练 blocks
        个连续块」对每个来源的有限块响应。probe 不推进 round、不移动游标，
        无论正常返回还是抛异常都通过 finally 回滚。

        blocks 控制测量前训练多少个**连续**块。单块往往推不动可分辨的信号，
        而加块几乎不花钱：一行 audit 是 来源×bank 次自回归生成，一个块的
        teacher-forced 前向反向只有十来个前向当量，**测量比训练贵几个数量级**。

        功效主要由**翻转次数**决定：McNemar 只看发生变化的题目，所以加 bank
        条数会增加翻转数；换种子（训练流顺序）做独立复现是另一条路。重复
        （repeats）只给误差棒，不增加信号。

        baseline 传 audit_baseline() 的结果时复用它的 before 行与随机流。
        """
        if type(model_id) is not int or not 0 <= model_id < self.models:
            raise ValueError("model_id 越界")
        if type(source_id) is not int or not 0 <= source_id < self.sources:
            raise ValueError("source_id 越界")
        if type(blocks) is not int or blocks < 1:
            raise ValueError("blocks 必须是 >= 1 的整数")
        if baseline is not None and baseline["model"] != model_id:
            raise ValueError("baseline 属于另一个专家")
        rollback = self._state()
        try:
            if baseline is None:
                rng = self.backend.evaluation_rng()
                before, before_detail = self._audit_row(model_id)
            else:
                rng = baseline["rng"]
                before, before_detail = baseline["row"], baseline["detail"]
            # _take_block 不改 self.cursors，位置要自己往前带，否则 blocks 个块
            # 全是同一块，训练量涨了信号却不涨。
            # 逐块日志要留住：grad_norm / loss / lr 是判断「能力下降是真退化还是
            # 训练发散」的唯一证据。lora.train_block 的裁剪阈值是 inf，即只测量
            # 不裁剪，单条离群记录足以把 adapter 推飞而不留痕迹。
            cursor, tokens, training = self.cursors[source_id], 0, []
            for _ in range(blocks):
                records, cursor = self._take_block(source_id, cursor)
                log = self.backend.train_block(model_id, records)
                training.append(log)
                tokens += log["completion_tokens"]
            # 意图是公共随机数：after 与 before 共用同一段随机流。⚠ 这里复位的是
            # 后端记下的 CPU 生成器状态；真实后端在 GPU 上采样时用的是 CUDA 生成器，
            # 复位不生效，before/after 两次采样并不是公共随机数配对。逐题配对的
            # McNemar 检验不依赖这一点，仍然有效。
            self.backend.set_evaluation_rng(rng)
            after, after_detail = self._audit_row(model_id)
            transfer = [(after[j] - before[j]) / tokens for j in range(self.sources)]
            # 逐条翻转数，供 McNemar 检验。逐条得分本来就有，取均值时丢掉了。
            flips = [{"gained": sum(a > b for b, a in zip(before_detail[j], after_detail[j])),
                      "lost": sum(a < b for b, a in zip(before_detail[j], after_detail[j]))}
                     for j in range(self.sources)]
            rows_before = deepcopy(self.competence)
            rows_after = deepcopy(self.competence)
            rows_before[model_id] = before
            rows_after[model_id] = after
            return {"model": model_id, "source": source_id, "blocks": blocks,
                    "completion_tokens": tokens, "training": training,
                    "before": before, "after": after, "transfer": transfer,
                    "flips": flips,
                    "portfolio_return": paired_return(rows_before, rows_after,
                                                      tokens, self.weights, self.tau)}
        finally:
            self._restore_state(rollback)
            self.backend.discard(rollback["backend"])

    def _audit_row(self, model_id):
        """在 audit split 上测一整行；audit 只服务 probe，不进入在线循环。

        同时返回逐条得分：配对检验要数 0→1 与 1→0 的翻转条数，取了均值就没了。
        """
        row, detail = [], []
        for k in range(self.sources):
            scores = [float(s) for s in self.backend.evaluate(model_id, self.banks["audit"][k])]
            detail.append(scores)
            row.append(sum(scores) / len(scores))
        return row, detail
