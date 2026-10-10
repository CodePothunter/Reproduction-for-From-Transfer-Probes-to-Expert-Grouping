"""训练后端接口与 CPU 替身。

TrainingBackend 是接口；SyntheticBackend 是不需要 GPU 的替身，用来在 CPU 上
跑通实验闭环、验证快照恢复与探针回滚。**合成后端不产生任何报告中的证据**——它的
competence 由一个人造的低维模型算出，不是真实模型的能力。真实实现见 lora.py。

替身刻意保留了完整的可变状态：隐向量（对应 adapter 参数）、动量缓冲（优化器）、
step 计数（scheduler）和随机数发生器。回滚必须恢复 adapters、optimizer buffers、
scheduler、数据游标与 RNG 状态，只有替身本身带着这些状态，snapshot/restore 的
测试才有意义；数据游标属于 experiment.py。
"""
import abc
import hashlib
import math
import random
from copy import deepcopy
from typing import Any


class TrainingBackend(abc.ABC):
    """训练后端接口。真实实现是 lora.LoRABackend。"""

    @abc.abstractmethod
    def train_block(self, model_id: int, records: list[dict]) -> dict:
        """仅接受 train split。返回 processed_tokens、completion_tokens 及训练日志。

        块必须是来源纯净的：一次只更新一个 (model, source) cell，否则测得的
        配对收益无法归因到某个 cell。
        """

    @abc.abstractmethod
    def evaluate(self, model_id: int, records: list[dict]) -> list[float]:
        """对独立评估数据返回逐样本 verifier 分数；不能更新模型或优化器。"""

    @abc.abstractmethod
    def parameters(self, model_id: int) -> list[float]:
        """当前 adapter 坐标，展平成一个向量。

        位移 Δφ = φ⁺ − φ⁻ 由调用方对前后两次结果相减得到，包含优化器的每一步、
        动量、预处理与权重衰减。
        """

    def parameters_array(self, model_id: int):
        """同 parameters，但返回 numpy 数组；实现可以覆盖为零拷贝路径。

        真实 adapter 有 1480 万个参数，走 list[float] 每次要造 1480 万个 Python
        float 对象（约 600 MB）。默认实现保持接口不变，LoRABackend 覆盖它直接
        从张量出。
        """
        import numpy

        return numpy.asarray(self.parameters(model_id), dtype=numpy.float32)

    @abc.abstractmethod
    def snapshot(self) -> Any:
        """完整保存参数、优化器、scheduler、随机数和可变缓存状态。"""

    @abc.abstractmethod
    def restore(self, state: Any) -> None:
        """恢复完整训练状态；不要只加载 adapter 参数。"""

    def discard(self, state: Any) -> None:
        """释放某份快照占用的外部资源；默认无事可做。

        区间回滚每轮都要先存一份快照，成功后那份就再无用处。真实后端把张量写在
        磁盘上（每份数百 MB），不回收就是纯粹的泄漏。检查点引用的快照绝不能走这里。
        """

    def evaluation_rng(self) -> Any:
        """取出判分生成所用的随机数状态；不支持则返回 None。

        用于**公共随机数**：训练前后各评一次时，若两次共用同一段随机流，权重只
        动了一点的绝大多数题目会生成逐 token 相同的答案，配对差就把训练效应从
        采样噪声里分离出来。这不改变被估量，只降方差，且不额外花生成成本。

        ⚠ LoRABackend 记录与复位的是 CPU 生成器状态；在 GPU 上采样时用的是 CUDA
        生成器，复位对采样不生效（见 experiment.probe 的说明）。
        """
        return None

    def set_evaluation_rng(self, state: Any) -> None:
        """把判分生成的随机流复位到 evaluation_rng() 取到的那个状态。

        state 为 None 表示后端不支持，按无操作处理——调用方据此退回独立采样。
        """


def _index(value, limit, name):
    if type(value) is not int or not 0 <= value < limit:
        raise ValueError(f"{name} 必须是 [0, {limit}) 内的整数")
    return value


def _unit(vector):
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise ValueError("方向向量不能为零向量")
    return [x / norm for x in vector]


class SyntheticBackend(TrainingBackend):
    """CPU 训练替身：用低维隐向量模拟能力与跨来源迁移。

    每个模型有隐向量 θ_i，每个来源有单位方向 u_k，能力为

        c_{i,j} = sigmoid(gain·⟨θ_i, u_j⟩ + bias)

    在来源 k 上训练把 θ_i 推向 u_k，于是所有与 u_k 正相关的来源 j 都会被带动。
    这正是迁移探针测的响应张量 A_{i,k,j} 所刻画的结构：没有跨来源迁移，
    怎么分组都没有区别。

    所有模型的初始 θ 都是零向量，对应 identical initial adapters。
    """

    def __init__(self, models, sources, *, dim=8, seed=0, directions=None,
                 learning_rate=0.05, momentum=0.9, total_steps=2000,
                 noise=0.01, gain=1.0, bias=-0.8, response=2e-4):
        if type(models) is not int or models < 1:
            raise ValueError("models 必须是正整数")
        if type(sources) is not int or sources < 1:
            raise ValueError("sources 必须是正整数")
        if type(total_steps) is not int or total_steps < 1:
            raise ValueError("total_steps 必须是正整数")
        for name, value in (("learning_rate", learning_rate), ("noise", noise),
                            ("gain", gain), ("response", response)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数")
        if not 0 <= momentum < 1:
            raise ValueError("momentum 必须落在 [0, 1)")

        self.models = models
        self.sources = sources
        self.dim = dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.total_steps = total_steps
        self.noise = noise
        self.gain = gain
        self.bias = bias
        self.response = response

        if directions is None:
            setup = random.Random(seed)
            directions = [_unit([setup.gauss(0, 1) for _ in range(dim)])
                          for _ in range(sources)]
        else:
            if len(directions) != sources:
                raise ValueError("directions 的条数必须等于 sources")
            directions = [_unit(list(vector)) for vector in directions]
            self.dim = dim = len(directions[0])
            if any(len(vector) != dim for vector in directions):
                raise ValueError("directions 的维度必须一致")
        self.directions = directions

        # identical initial adapters：所有模型从同一个零向量出发。
        self.params = [[0.0] * dim for _ in range(models)]
        self.velocity = [[0.0] * dim for _ in range(models)]
        # 每个 descendant 各有一份 optimizer/scheduler 状态。共享 step 会让先训练的
        # 模型拿到更高的学习率，在对称计划下也产生系统性优势——那是与来源结构无关的
        # 伪信号，分配方法会误当成可学结构。
        self.steps = [0] * models
        self.rng = random.Random(seed)

    # --- 内部量 -------------------------------------------------------------

    @property
    def step(self):
        """已执行的块总数；scheduler 的真实状态是每模型的 steps。"""
        return sum(self.steps)

    def scheduled_rate(self, model_id=0, step=None):
        """线性衰减的学习率；每个模型按自己的 step 计。"""
        step = self.steps[_index(model_id, self.models, "model_id")] if step is None else step
        return self.learning_rate * max(0.0, 1.0 - step / self.total_steps)

    def true_competence(self, model_id, source):
        """替身才有的"真值"能力，真实后端拿不到；仅供检查与构造测试。"""
        _index(model_id, self.models, "model_id")
        _index(source, self.sources, "source")
        inner = sum(a * b for a, b in zip(self.params[model_id], self.directions[source]))
        return 1.0 / (1.0 + math.exp(-(self.gain * inner + self.bias)))

    def _checked(self, model_id, records, *, training):
        _index(model_id, self.models, "model_id")
        if not records:
            raise ValueError("records 不能为空")
        sources = set()
        for record in records:
            for key in ("id", "source", "split", "tokens", "completion_tokens"):
                if key not in record:
                    raise ValueError(f"记录缺少字段 {key}")
            _index(record["source"], self.sources, "record.source")
            sources.add(record["source"])
            if training and record["split"] != "train":
                raise ValueError("train_block 只接受 train split")
            if not training and record["split"] == "train":
                raise ValueError("评估不能使用 train split")
        if training and len(sources) != 1:
            raise ValueError("训练块必须是来源纯净的，不能混合多个 source")
        return sources

    # --- 接口实现 -----------------------------------------------------------

    def train_block(self, model_id: int, records: list[dict]) -> dict:
        """仅接受 train split。返回 processed_tokens、completion_tokens 及训练日志。"""
        source = self._checked(model_id, records, training=True).pop()
        processed = sum(record["tokens"] for record in records)
        supervised = sum(record["completion_tokens"] for record in records)
        if supervised <= 0:
            raise ValueError("训练块的监督 token 数必须为正")

        before = self.true_competence(model_id, source)
        # 更新幅度随监督 token 数增长，但用 (1 − c) 抑制：能力越高边际收益越小，
        # 从而给出响应曲率，而不是线性外推。
        strength = self.response * supervised * (1.0 - before)
        direction = self.directions[source]
        gradient = [strength * value + self.rng.gauss(0.0, self.noise)
                    for value in direction]
        rate = self.scheduled_rate(model_id)
        velocity = self.velocity[model_id]
        params = self.params[model_id]
        for d in range(self.dim):
            velocity[d] = self.momentum * velocity[d] + gradient[d]
            params[d] += rate * velocity[d]
        self.steps[model_id] += 1

        return {"model": model_id, "source": source, "records": len(records),
                "processed_tokens": processed, "completion_tokens": supervised,
                "step": self.steps[model_id], "learning_rate": rate,
                "update_norm": math.sqrt(sum((rate * v) ** 2 for v in velocity)),
                "loss": -math.log(max(before, 1e-12))}

    def evaluate(self, model_id: int, records: list[dict], *, decoding="stochastic") -> list[float]:
        """对独立评估数据返回逐样本 verifier 分数；不能更新模型或优化器。

        判定阈值由 (model_id, record.id) 的哈希决定，与训练 RNG 无关：
        本方法既不消耗随机数也不改动任何状态，因此配对测量前后只有能力在变，
        同一条样本的阈值保持一致，g_b 不会被重抽样噪声淹没。

        decoding 参数为与真实后端保持同一签名而存在。替身没有解码过程，
        两种协议给出相同结果，这里不假造一个差值。
        """
        if decoding not in ("stochastic", "greedy"):
            raise ValueError("decoding 只能是 stochastic 或 greedy")
        self._checked(model_id, records, training=False)
        scores = []
        for record in records:
            probability = self.true_competence(model_id, record["source"])
            digest = hashlib.sha256(f"{model_id}:{record['id']}".encode("utf-8")).digest()
            draw = int.from_bytes(digest[:8], "big") / 2 ** 64
            scores.append(1.0 if draw < probability else 0.0)
        return scores

    # --- 坐标 ---------------------------------------------------------------

    def parameters(self, model_id: int) -> list[float]:
        """隐向量即本替身的 adapter 坐标。"""
        _index(model_id, self.models, "model_id")
        return list(self.params[model_id])

    def snapshot(self) -> Any:
        """完整保存参数、优化器、scheduler、随机数和可变缓存状态。

        返回纯 JSON 可序列化结构，这样 experiment.checkpoint 可以直接交给
        storage.save，CPU 上就能端到端验证中断恢复。
        """
        version, internal, gauss_next = self.rng.getstate()
        return {"format": "lwts-synthetic-v1",
                "params": deepcopy(self.params),
                "velocity": deepcopy(self.velocity),
                "steps": list(self.steps),
                "rng": [version, list(internal), gauss_next]}

    def restore(self, state: Any) -> None:
        """恢复完整训练状态；不要只加载 adapter 参数。"""
        if not isinstance(state, dict) or state.get("format") != "lwts-synthetic-v1":
            raise ValueError("不支持的后端状态格式")
        params, velocity = state.get("params"), state.get("velocity")
        for name, matrix in (("params", params), ("velocity", velocity)):
            if not isinstance(matrix, list) or len(matrix) != self.models \
                    or any(not isinstance(row, list) or len(row) != self.dim for row in matrix):
                raise ValueError(f"{name} 的形状与当前后端不一致")
        steps = state.get("steps")
        if not isinstance(steps, list) or len(steps) != self.models \
                or any(type(s) is not int or s < 0 for s in steps):
            raise ValueError("steps 必须是每模型一项的非负整数列表")
        rng = state.get("rng")
        if not isinstance(rng, list) or len(rng) != 3:
            raise ValueError("rng 状态格式错误")
        self.params = deepcopy(params)
        self.velocity = deepcopy(velocity)
        self.steps = list(steps)
        self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
