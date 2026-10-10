#!/usr/bin/env python
"""实测跨来源响应张量 A_{i,k,j}：在来源 k 上训练，看所有来源 j 的能力变化。

    PYTHONPATH=. python pipeline/probe_transfer.py --manifest <manifest.jsonl> ...
    （本报告的两次运行见 pipeline/run_probes.sh）

**要回答的问题。** 把两个来源分到不同专家，前提是它们需要不同的能力——否则
分组无从谈起。这一点不能靠看样例判断：同一标签下的两个来源可能互不迁移，
不同标签下的两个来源也可能互相带动。

要测的量：从同一个起点出发，在来源 k 上训一段，看**所有**来源 j 的能力变化

    A_{i,k,j} ≈ Δc_{i,j} / Δb   （在来源 k 上投入预算 b）

对角项是自身收益，非对角项是迁移。判据：

  · 对角远大于非对角  →  轴厚，分开训练有意义
  · 对角与非对角相当  →  轴薄，训哪个来源效果一样
  · 非对角为负        →  存在干扰

Experiment.probe 已经实现了这件事（在独立的 audit split 上测训练前后的整行
能力，且**无论正常返回还是抛异常都回滚**全部状态：参数、优化器、scheduler、
RNG、数据游标）。本脚本只是逐 cell 调它并把结果排成矩阵。

**成本结构是反直觉的：测量比训练贵两三个数量级。** 一行 audit 是 来源×bank 次
自回归生成，逐 token 串行；一个训练块是 32768 token 的 teacher-forced 前向反向，
序列内并行。所以默认 --blocks 16：加块只给每行加一点开销，信号却明显变大。

功效由**翻转次数**决定：McNemar 只看发生变化的题目，所以加 bank 条数会增加
翻转数（代价是 bank 翻倍则每行 audit 的生成量翻倍）；换 --seed（训练流顺序）
做独立复现是另一条路。不要加 --repeats：probe 会回滚，重复训的是同一段数据、
从同一份权重出发，给的是误差棒不是信号。

两处设计：

  · **before 行只测一次**（audit_baseline）。probe 会回滚，各行面对的 before
    状态完全相同，各测一次既多花一整行生成，又给每行各注入一份独立噪声。
  · after 行复位到 before 行记下的随机流起点，意图是公共随机数。⚠ 在 GPU 上
    这个复位不生效（见 Experiment.probe 的说明），两次采样并不配对；逐题配对
    的 McNemar 不依赖它。

判据因此是 **McNemar**：数 0→1 与 1→0 的翻转条数，逐条得分本来就有。

⚠️ 默认只测 expert 0，用 --models 扩；每多一个专家成本翻倍。

⚠️ 判分会执行模型生成的代码。子进程 + 超时 + 资源上限只是健壮性隔离，
不是安全沙箱：只在可丢弃的机器上跑。
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# block_tokens 取 32768：DeepMath 的 R1 解答中位约 4500-5500 token，16384 的块只
# 装得下 3-4 条，期望超出量就是半条记录，实测块与名义值之比紧贴 1.25 的容限，
# manifest 记录构成稍一变动就被 _block_plan 拦下。
#
#   block_tokens   16384  24576  32768  49152  65536
#   最差来源比值    1.24   1.17   1.12   1.08   1.06
#
# 32768 给出 1.12 的余量。配 --blocks 16 时每格训练 16 × 32768 = 524288 token。
# tau/eta/gamma/rho 只为满足 Experiment 的必填项；探针只用 tau（portfolio_return）。
SETTINGS = {
    "block_tokens": 32768, "tokens_per_interval": 524288,
    "tau": 0.05, "eta": 0.5, "gamma": 0.5, "rho": 0.05,
}


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as source:
        manifest = json.loads(next(source))
        manifest["records"] = [json.loads(line) for line in source]
    return manifest


def mcnemar(gained, lost):
    """配对翻转的精确双侧 p 值：在 b+c 次翻转里看到这么偏的分裂有多罕见。

    零假设是「训练不改变这道题的对错」，此时每次翻转倒向两边的概率各半，于是
    b ~ Binomial(b+c, 1/2)。这正是需要的检验：它只看**发生了变化的题目**，
    不受两次评估共有的那部分噪声影响，也不需要假设正态。

    翻转总数很小时（这正是弱信号的情形）不能用卡方近似，所以直接算精确和。
    """
    n = gained + lost
    if n == 0:
        return 1.0
    extreme = min(gained, lost)
    total = 2 ** n
    tail = sum(math.comb(n, i) for i in range(extreme + 1))
    return min(1.0, 2 * tail / total)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--manifest", required=True)
    p.add_argument("--models", type=int, nargs="+", default=[0],
                   help="测哪些专家；每多一个专家成本翻倍")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=16384)
    # 逐来源取值，默认同 run_pilot：数学 8192、代码 1024。缺省的四个值对应四来源
    # manifest；7 来源 manifest 必须显式传 7 个值（见 run_probes.sh）。用同一个
    # 上限会让数学侧的训练效应混进截断伪影（见 lwts.lora._budgets）。
    p.add_argument("--max-new-tokens", type=int, nargs="+",
                   default=[8192, 8192, 1024, 1024])
    p.add_argument("--batch-size", type=int, default=32,
                   help="批内同步解码：整批墙钟由**批内最长**决定，而数学生成有 "
                        "37%% 走满上限，所以几乎每批都被拖满。加宽批因此近乎免费——"
                        "实测每条成本 批8 45.5s / 批16 32.0s / 批32 28.4s，"
                        "峰值显存 5.1 / 6.5 / 9.5 GB（卡 24 GB）")
    p.add_argument("--bank", type=int, default=80,
                   help="每来源截到多少条 feedback/audit。**成本正比于它**：一行 "
                        "audit 是 来源×bank 次自回归生成。0 表示不截（用 split 全量，"
                        "七来源 manifest 下是 15724 条/行，约 56 小时）")
    p.add_argument("--bank-offset", type=int, default=0,
                   help="从每来源的第几条开始取 bank，用来切出**与既有测量不相交**"
                        "的一组 audit 题。这是目前唯一能做独立 evaluation 复现的"
                        "旋钮：换 --seed 只换训练流顺序，bank 划分不变，所以两次"
                        "测量共享同一批题目与同一个 before——那证明的是对训练顺序"
                        "的稳定性，不是独立复现。配合 offset=80 才是换了题目。"
                        "audit split 每来源最少 639 条（源 4），切七组 80 条都够")
    p.add_argument("--seed", type=int, default=17,
                   help="决定各来源训练流的**顺序**（_streams 按 seed:source 洗牌），"
                        "换种子即换训练数据的取用顺序，是做独立复现的正确做法。"
                        "bank 的划分不受它影响，所以基线行不变——这正好让复现与"
                        "原测共享同一个 before，差异只来自训练数据本身")
    p.add_argument("--sources", type=int, nargs="+", default=None,
                   help="只测这几个训练来源（矩阵的哪几行），缺省测全部。每行约 70 "
                        "分钟，单卡上重跑已有的行是纯浪费。分次跑出的行与一次跑完"
                        "等价：每行都回滚到同一个 before，且共用同一条 baseline")
    p.add_argument("--blocks", type=int, default=16,
                   help="每格测量前训练多少个连续块；这是调信号的旋钮（见下）。"
                        "16 × 32768 = 524288 token，即一个区间的预算")
    p.add_argument("--repeats", type=int, default=1,
                   help="每个 cell 重复几次；只给误差棒，不增加信号（见下）")
    p.add_argument("--out", default="runs/transfer")
    a = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lwts.experiment import Experiment
    from lwts.lora import LoRABackend
    from lwts.verifiers import default_verifiers

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def report(text=""):
        print(text, flush=True)
        lines.append(text)

    manifest = read_manifest(a.manifest)
    sources = len(manifest.get("sources") or [0, 1, 2, 3])
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    base.requires_grad_(False)
    config = LoraConfig(r=a.rank, lora_alpha=2 * a.rank,
                        target_modules=list(LoRABackend.DEFAULT_TARGETS),
                        task_type="CAUSAL_LM")
    model = get_peft_model(base, config, adapter_name="expert_0")
    model.add_adapter("expert_1", config)
    model.cuda()
    backend = LoRABackend(model, tokenizer, models=2, sources=sources,
                          state_dir=str(out / "state"), max_length=a.seq_len,
                          max_new_tokens=(a.max_new_tokens[0] if len(a.max_new_tokens) == 1
                                          else a.max_new_tokens),
                          batch_size=a.batch_size,
                          verifiers=default_verifiers())
    experiment = Experiment(
        {"experiment": dict(SETTINGS, models=2, sources=sources, method="uniform",
                            seed=a.seed, warmup_rounds=1)},
        manifest, backend)
    # **必须自己截断 bank。** 这里的成本正比于 bank 大小：一行 audit 是
    # 来源×bank 次自回归生成。七来源 manifest 存的是 split 的全量，不截的话
    # audit 一行是 15724 条而不是 560 条——**28 倍**，一行要 56 小时。
    # 这是个静默的成本放大：不报错，只是慢到跑不完。
    if a.bank:
        start = a.bank_offset
        for split in ("feedback", "audit"):
            experiment.banks[split] = {k: rows[start:start + a.bank]
                                       for k, rows in experiment.banks[split].items()}
        short = {k: len(v) for k, v in experiment.banks["audit"].items() if len(v) < a.bank}
        if short:
            # 不相交的第二组题不够时必须硬停：悄悄退化成较小的 bank 会让量化步长
            # 和翻转数跟着变，两次测量不再可比——又是一个不报错只让结果无效的坑。
            raise ValueError(f"offset={start} 下以下来源不足 {a.bank} 条：{short}。"
                             f"改小 --bank 或 --bank-offset，不要让它静默退化")
        if start:
            report(f"  bank 取 [{start}, {start + a.bank})：与 offset=0 的既有测量"
                   f"**不相交**，这是独立的 evaluation 复现")
    bank = {k: len(v) for k, v in experiment.banks["audit"].items()}
    report(f"来源数 {sources}   audit bank {bank}   每格训练 {a.blocks} 块")
    # 构造期不做能力评估：迁移矩阵、翻转与 McNemar 都不读它。
    report("构造期不做能力评估。迁移矩阵与 McNemar 不受影响；portfolio_return 是本专家的"
           "加权自身增益（其余专家的行取零），不是组合口径。")
    # 量化步长是仪器刻度，不是可分辨的最小信号：实测两次独立评估的差最大到
    # 0.0875，是步长的 7 倍。判读因此不比阈值，而是对逐条翻转做 McNemar。
    step = 1.0 / min(bank.values())
    report(f"量化步长 {step:.4f}；判读用 McNemar 逐条翻转，不用阈值")

    results = []
    for model_id in a.models:
        report(f"\n{'=' * 78}\n专家 {model_id} 的响应张量 A[{model_id}, k, j]"
               f"（行=训练来源 k，列=被测来源 j，单位 1e-6/token）\n{'=' * 78}")
        started = time.time()
        # before 行测一次，四行共用：probe 会回滚，它们面对的是同一个状态。
        baseline = experiment.audit_baseline(model_id)
        report(f"  基线（测一次，{sources} 行共用，{time.time() - started:.0f}s）  "
               + " ".join(f"{c:.4f}" for c in baseline["row"]))
        header = "  训练来源" + "".join(f"{'→ 源' + str(j):>12}" for j in range(sources))
        report(header + f"{'自身/迁移':>12}")
        # 只跑指定的训练来源。每行约 70 分钟，一块卡上重跑已有的行是纯浪费；而
        # probe 每行都回滚到同一个 before 状态、且共用同一条 baseline，所以分几次
        # 跑出来的行与一次跑完的完全等价——**前提是 baseline 也相同**，它由
        # seed 与 bank 决定，与跑哪几行无关。
        for k in (a.sources if a.sources is not None else range(sources)):
            if not 0 <= k < sources:
                raise ValueError(f"--sources 里的 {k} 越界（0..{sources - 1}）")
            rows = []
            started = time.time()
            for _ in range(a.repeats):
                rows.append(experiment.probe(model_id, k, blocks=a.blocks,
                                             baseline=baseline))
            transfer = [statistics.fmean(r["transfer"][j] for r in rows)
                        for j in range(sources)]
            delta = [statistics.fmean(r["after"][j] - r["before"][j] for r in rows)
                     for j in range(sources)]
            # 翻转数跨 repeats 相加：McNemar 的样本量就是翻转总数。
            flips = [{"gained": sum(r["flips"][j]["gained"] for r in rows),
                      "lost": sum(r["flips"][j]["lost"] for r in rows)}
                     for j in range(sources)]
            pvalues = [mcnemar(f["gained"], f["lost"]) for f in flips]
            own = transfer[k]
            others = [transfer[j] for j in range(sources) if j != k]
            worst = max(abs(x) for x in others) if others else 0.0
            # 「有没有信号」由检验回答，不由阈值回答。
            measurable = pvalues[k] < 0.05
            tokens = statistics.fmean(r["completion_tokens"] for r in rows)
            floor = step / tokens
            bounded = worst < floor
            ratio = abs(own) / max(worst, floor)
            results.append({"model": model_id, "source": k, "transfer": transfer,
                            "delta": delta, "before": baseline["row"], "step": step,
                            "flips": flips, "pvalues": pvalues,
                            "measurable": measurable, "ratio_is_lower_bound": bounded,
                            "own": own, "max_abs_other": worst, "ratio": ratio,
                            "tokens": tokens, "blocks": a.blocks,
                            "portfolio_return": statistics.fmean(
                                r["portfolio_return"] for r in rows)})
            cells = "".join(f"{v * 1e6:>12.2f}" for v in transfer)
            shown = (f"{'不显著':>10}" if not measurable else
                     f"{'≥' + format(ratio, '.1f'):>12}" if bounded else
                     f"{ratio:>12.1f}")
            report(f"  {k:>8}{cells}{shown}   ({time.time() - started:.0f}s)")
            report("           能力差 " + " ".join(f"{d:+.4f}" for d in delta))
            report("           翻转 " + " ".join(
                f"+{f['gained']}/-{f['lost']}(p={p:.3f})"
                for f, p in zip(flips, pvalues)))
            # 能力下降到底是「真退化」还是「训练发散」，只有这几个数能分辨。
            # lora.train_block 用 inf 作裁剪阈值，即从不裁剪，梯度尖峰不留痕迹。
            train = rows[0]["training"]
            norms = [t["grad_norm"] for t in train]
            losses = [t["loss"] for t in train]
            report(f"           训练 {len(train)} 块  "
                   f"loss {losses[0]:.3f}→{losses[-1]:.3f}  "
                   f"grad_norm 中位 {statistics.median(norms):.3f} "
                   f"最大 {max(norms):.3f}（第 {norms.index(max(norms)) + 1} 块）  "
                   f"lr {train[0]['learning_rate']:.2e}→{train[-1]['learning_rate']:.2e}")
            if max(norms) > 10 * statistics.median(norms):
                report(f"           ⚠ 梯度尖峰：最大是中位的 "
                       f"{max(norms) / statistics.median(norms):.0f} 倍，而裁剪阈值是 inf")

    report("\n" + "=" * 74)
    report("判读：对角/非对角比值远大于 1 说明轴厚；接近 1 说明训哪个来源效果相当，")
    report("      分配问题退化。负的非对角项是干扰。")
    for r in results:
        k = r["source"]
        # 自身效应不显著必须先判：此时比值可以是任意大的数（分母只是测量下限），
        # 会被 ratio > 3 误报成「轴厚」——自身效应为零时正是这种形态，绝不能
        # 让它伪装成最强结论。
        if not r["measurable"]:
            flip = r["flips"][k]
            note = (f"⚠ 自身效应不显著：翻转 +{flip['gained']}/-{flip['lost']}，"
                    f"p={r['pvalues'][k]:.3f}，本行不可判读")
        else:
            note = ("轴厚" if r["ratio"] > 3 else
                    "⚠ 轴薄：自身收益与迁移同量级" if r["ratio"] > 1 else
                    "⚠ 训自身还不如训别的来源")
        prefix = "≥" if r["ratio_is_lower_bound"] else ""
        report(f"  专家 {r['model']} 来源 {k}: 自身 {r['own'] * 1e6:+.2f} "
               f"最大非对角 {r['max_abs_other'] * 1e6:.2f}  "
               f"比值 {prefix}{r['ratio']:.1f}   {note}")
    blind = [r for r in results if not r["measurable"]]
    if blind:
        report(f"\n  ⚠ {len(blind)}/{len(results)} 行的自身效应不显著。信号的旋钮是"
               f" --blocks（现在 {a.blocks}），不是 bank 也不是 --repeats：一行 audit"
               f" 的生成成本比一个训练块高三个数量级，加块几乎不花钱。")
    # 只报显著的负迁移。不显著的负号绝大多数是噪声，报出来会制造假的「干扰」。
    negatives = [(r["model"], r["source"], j, r["transfer"][j] * 1e6, r["pvalues"][j])
                 for r in results for j in range(sources)
                 if j != r["source"] and r["transfer"][j] < 0 and r["pvalues"][j] < 0.05]
    if negatives:
        report("\n  负迁移（训 k 反而降低 j，仅列 p<0.05）：")
        for model_id, k, j, v, p in negatives:
            report(f"    专家 {model_id}: 训来源 {k} → 来源 {j} 下降 {abs(v):.2f}"
                   f"（p={p:.3f}）")

    (out / "transfer.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    (out / "transfer.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report(f"\n产物写入 {out}/")


if __name__ == "__main__":
    main()
