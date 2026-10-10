"""真实 LoRA 训练后端：一个冻结基座 + M 个可独立更新的 adapter。

与 SyntheticBackend 的关系是接口相同、实现不同；报告中的训练与评估数字都由这个后端产生。

三处必须与数据层和评估协议对齐，写错了不会报错但结果无效：

1. loss mask 复刻数据层的 offset 规则（token_policy=offset-start-v1-no-special-tokens）：
   对 prompt+completion 一次分词、不加特殊 token，监督位满足 end > start >= len(prompt)。
   训练时实际监督数必须等于 manifest 里的 completion_tokens，本模块会核对。
2. 解码用固定的随机采样，正温度，**不做 top-k / top-p 截断**，所有专家、所有
   评估共用同一协议。
3. 每个 descendant 各有 optimizer 与 scheduler。共享 scheduler 会让先训练的专家
   拿到更高学习率，产生与来源结构无关的系统性优势。

真实 Tensor 不进 JSON。snapshot 把张量写进 state_dir 下的独立文件，只返回可
序列化的元数据，这样 experiment.checkpoint 仍可直接交给 storage.save。
"""
import hashlib
import random
from contextlib import contextmanager
from pathlib import Path

from .backend import TrainingBackend


def _require_torch():
    try:
        import torch
    except ImportError as error:                       # pragma: no cover - 环境相关
        raise ImportError("真实后端需要 torch：pip install -e '.[train]'") from error
    return torch


def _budgets(value, sources):
    """把 max_new_tokens 规整成 {来源: 预算}；给标量则各来源相同。

    **必须允许逐来源取值。** 训练会把数学侧的生成长度放大数倍，而代码侧的
    训练目标通常只有一两百 token。用同一个上限的后果是两头都错：

      · 数学给少了 → 截断会系统性地压低训练后的数学分数，失真还正比于该方法
        给数学分了多少预算，是与自变量同向的混杂，不是噪声。
      · 代码给多了 → 批量解码按批内最长那条算，早已 EOS 的短序列全程陪跑。
    """
    if isinstance(value, dict):
        table = {int(k): int(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        table = {k: int(v) for k, v in enumerate(value)}
    else:
        table = {k: int(value) for k in range(sources)}
    missing = [k for k in range(sources) if k not in table]
    if missing:
        raise ValueError(f"max_new_tokens 缺少来源 {missing} 的取值")
    bad = {k: v for k, v in table.items() if v < 1}
    if bad:
        raise ValueError(f"max_new_tokens 必须为正整数，收到 {bad}")
    return table


class _ChunkedLoss:
    """分块求出的梯度已经算好，这里把它接回 hidden，使调用方仍像普通 loss 那样用。

    支持 float()/detach() 取标量、乘以标量、取负、backward()。backward 时把
    累加的梯度乘上当前缩放送进 hidden，反向只穿过 transformer 一次。
    """

    def __init__(self, hidden, gradient, value, scale=1.0):
        self._hidden, self._gradient, self._value, self._scale = (
            hidden, gradient, value, scale)

    def detach(self):
        return self._value * self._scale

    def __float__(self):
        return float(self._value * self._scale)

    def __mul__(self, factor):
        return _ChunkedLoss(self._hidden, self._gradient, self._value,
                            self._scale * factor)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def backward(self):
        self._hidden.backward(self._gradient * self._scale)


class LoRABackend(TrainingBackend):
    """基座冻结，M 个 LoRA adapter 独立更新。

    构造时接收已加载的 model 与 tokenizer，便于在 CPU 上用微型模型验证机制；
    真实运行的构造见 build.build_backend 与 pipeline/run_pilot.py。
    """

    def __init__(self, model, tokenizer, models, sources, *, state_dir="runs/state",
                 learning_rate=1e-4, total_steps=2000, weight_decay=0.0,
                 max_length=8192, temperature=0.8, max_new_tokens=8192,
                 seed=0, verifiers=None, adapter_prefix="expert",
                 batch_size=16,
                 gradient_checkpointing=True):
        torch = _require_torch()
        if type(models) is not int or models < 1:
            raise ValueError("models 必须是正整数")
        if not tokenizer.is_fast:
            raise ValueError("需要 fast tokenizer：loss mask 依赖 offsets")
        if temperature <= 0:
            raise ValueError("温度必须为正：评估协议是正温度的随机采样")
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.models = models
        self.sources = sources
        self.max_length = max_length
        self.temperature = temperature
        self.max_new_tokens = _budgets(max_new_tokens, sources)
        self.batch_size = batch_size          # 批量解码；单条约 20 tok/s，不批不可行
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.verifiers = verifiers or {}
        self.names = [f"{adapter_prefix}_{i}" for i in range(models)]
        self.state_dir = Path(state_dir)

        # 调用流水：每次 evaluate / train_block 记一条。唯一目的是证明**每个来源
        # 都真的进入了执行路径**——配置文件里出现来源名字不算数：来源数传错时
        # 多出来的来源会被整个忽略而不报错，这种失效只有调用流水能看出来。
        # 纯记录，不改变任何行为。
        self.call_log = []

        # 每个 descendant 一份 optimizer 与 scheduler
        self.optimizers, self.schedulers, self.steps = [], [], [0] * models
        for name in self.names:
            params = self._adapter_params(name)
            if not params:
                raise ValueError(f"adapter {name} 没有可训练参数，检查 LoRA 挂载目标")
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
            self.optimizers.append(optimizer)
            self.schedulers.append(torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: max(0.0, 1.0 - step / total_steps)))
        if gradient_checkpointing:
            # 长序列下 activation 才是主导项：分块交叉熵解决了 logits，但 24 层
            # 在 10875 token 上仍会撑爆 24GB 卡（实测 OOM）。基座冻结时输入没有
            # 需要梯度的张量，必须先 enable_input_require_grads 才能重算。
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        self.gradient_checkpointing = gradient_checkpointing
        self.rng = random.Random(seed)
        self._snapshot_seq = -1               # 快照序号只增不减，见 _next_blob
        torch.manual_seed(seed)
        # 生成用的独立 RNG 流（CPU 侧）：评估不得扰动训练随机序列
        self._eval_rng = torch.Generator().manual_seed(seed + 1).get_state()

    # Qwen3.5 是混合注意力：24 层中 6 层 full_attention、18 层 linear_attention，
    # 两者的投影模块命名完全不同。只写 q/k/v/o_proj 会让 LoRA 仅挂到 6 层，
    # 75% 的模型（全部线性注意力层）拿不到任何可训练参数——不会报错，但训练无效。
    FULL_ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
    LINEAR_ATTENTION = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
    MLP = ("gate_proj", "up_proj", "down_proj")
    DEFAULT_TARGETS = FULL_ATTENTION + LINEAR_ATTENTION      # 覆盖全部 24 层

    @classmethod
    def from_pretrained(cls, model_id, models, sources, *, revision="main",
                        rank=32, alpha=64, dropout=0.0, targets=None, dtype="bfloat16",
                        **kwargs):
        """加载基座并挂上 M 个 LoRA adapter。

        实测（未下权重，仅按 config 构图）：AutoModelForCausalLM 可用，
        映射到 Qwen3_5ForCausalLM；架构标称 Qwen3_5ForConditionalGeneration。

        r=32 下三种目标集的可训练参数量：
            仅 full attention        6/24 层    2.9M   0.16%
            两种注意力（默认）        24/24 层   14.8M   0.78%
            两种注意力 + MLP          24/24 层   33.6M   1.76%
        """
        torch = _require_torch()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype))
        base.requires_grad_(False)
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                            target_modules=list(targets or cls.DEFAULT_TARGETS),
                            task_type="CAUSAL_LM")
        model = get_peft_model(base, config, adapter_name="expert_0")
        for i in range(1, models):
            model.add_adapter(f"expert_{i}", config)
        backend = cls(model, tokenizer, models, sources, **kwargs)
        backend.check_coverage()
        return backend

    def check_coverage(self):
        """核对 LoRA 是否覆盖了每一层。欠覆盖不会报错，但训练无效。"""
        covered = {name.split("layers.")[1].split(".")[0]
                   for name, _ in self.model.named_modules()
                   if "lora_A" in name and "layers." in name}
        total = getattr(getattr(self.model.config, "text_config", self.model.config),
                        "num_hidden_layers", None)
        if total is not None and len(covered) < total:
            raise ValueError(
                f"LoRA 只覆盖了 {len(covered)}/{total} 层。Qwen3.5 是混合注意力，"
                f"full 与 linear 两种层的投影命名不同，target_modules 必须同时包含；"
                f"当前目标：{sorted(self.model.peft_config[self.names[0]].target_modules)}")
        return {"covered_layers": len(covered), "total_layers": total}

    # --- adapter 坐标 -------------------------------------------------------

    def _adapter_params(self, name):
        """按参数名排序返回该 adapter 的可训练张量；顺序固定，位移与梯度共用。"""
        return [p for _, p in sorted(self.model.named_parameters())
                if f".{name}." in _ or _.endswith(f".{name}")]

    def _activate(self, model_id):
        name = self.names[self._index(model_id)]
        self.model.set_adapter(name)
        for i, other in enumerate(self.names):
            requires = i == model_id
            for p in self._adapter_params(other):
                p.requires_grad_(requires)
        return name

    def _index(self, model_id):
        if type(model_id) is not int or not 0 <= model_id < self.models:
            raise ValueError(f"model_id 必须是 [0, {self.models}) 内的整数")
        return model_id

    def parameters_array(self, model_id):
        """展平的 adapter 坐标，直接给 numpy 数组。

        r=32 下一个 adapter 是 1480 万个参数。.tolist() 会为每个数造一个 Python
        float 对象（约 600 MB）。
        """
        name = self.names[self._index(model_id)]
        with self.torch.no_grad():
            return self.torch.cat([p.detach().reshape(-1).float()
                                   for p in self._adapter_params(name)]).cpu().numpy()

    def parameters(self, model_id):
        """展平的 adapter 坐标。位移 Δφ 由调用方前后相减得到。"""
        return self.parameters_array(model_id).tolist()

    # --- 分词与 loss mask ---------------------------------------------------

    @property
    def device(self):
        """模型所在设备。张量必须跟着它走，否则 embedding 查表会跨设备报错。"""
        return next(self.model.parameters()).device

    def _encode(self, prompt, completion):
        """复刻数据层的监督边界：end > start >= len(prompt)，不加特殊 token。"""
        torch = self.torch
        # 不截断：超长必须由数据层的 max_tokens 过滤掉。训练端截断会无声地
        # 改变训练内容，且让实际监督 token 数与 manifest 对不上。
        encoded = self.tokenizer(prompt + completion, add_special_tokens=False,
                                 return_offsets_mapping=True)
        ids = encoded["input_ids"]
        if len(ids) > self.max_length:
            raise ValueError(
                f"序列 {len(ids)} token 超过 max_length={self.max_length}；"
                "应在 prepare_records 时用 max_tokens 过滤，而不是在训练端截断")
        boundary = len(prompt)
        labels = [ident if (end > start and start >= boundary) else -100
                  for ident, (start, end) in zip(ids, encoded["offset_mapping"])]
        device = self.device
        return (torch.tensor([ids], device=device),
                torch.tensor([labels], device=device))

    def _stack(self):
        """取出 transformer 主体与输出头，用于分块计算 logits。

        PEFT 包装后层次是 peft_model.base_model.model.{model, lm_head}；
        取不到就返回 None，调用方退回整块路径。
        """
        try:
            inner = self.model.base_model.model
            return inner.model, inner.lm_head
        except AttributeError:                           # pragma: no cover - 版本相关
            return None, None

    def _supervised_loss(self, ids, labels, chunk=1024, temperature=1.0):
        """完成 token 的平均负对数似然；prompt 位置被 -100 屏蔽。

        分块计算而不是一次性物化 logits：词表 248320，一条 10875 token 的
        DeepMath 样本的 BF16 logits 就是 5.0 GB，再 .float() 一份是 10.1 GB，
        加反向梯度必然撑爆 24GB 卡。实测确实 OOM。

        做法是先拿 hidden states（10875×2048 只有 44MB），按块过输出头并就地
        求出对 hidden 的梯度，累加后再一次性穿过 transformer 反向。任一时刻
        只有一块的 FP32 logits 存在。

        temperature：logits 先除以 T 再求交叉熵。SFT 训练（train_block）用默认
        T=1，即标准的完成 token 负对数似然；本包里没有其它取值的调用方。
        """
        torch = self.torch
        if not temperature > 0:
            raise ValueError("temperature 必须为正")
        target = labels[:, 1:]
        keep = target != -100
        if not keep.any():
            return None, 0
        count = int(keep.sum())
        body, head = self._stack()
        mask = torch.ones_like(ids)
        if body is None or head is None:                 # 退路：整块计算
            out = self.model(input_ids=ids, attention_mask=mask)
            flat = out.logits[:, :-1].float().reshape(-1, out.logits.size(-1)) / temperature
            gold = target.reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                flat[keep.reshape(-1)], gold[keep.reshape(-1)])
            return loss, count

        hidden = body(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
        detached = hidden.detach().requires_grad_(True)
        gold = target.reshape(-1)
        positions = keep.reshape(-1).nonzero(as_tuple=True)[0]
        accumulated = torch.zeros_like(detached)
        total = 0.0
        flat_detached = detached.reshape(-1, detached.size(-1))
        for start in range(0, positions.numel(), chunk):
            index = positions[start:start + chunk]
            logits = head(flat_detached[index]).float() / temperature
            piece = torch.nn.functional.cross_entropy(
                logits, gold[index], reduction="sum")
            grad, = torch.autograd.grad(piece, detached, retain_graph=False)
            accumulated += grad
            total += float(piece.detach())
        # 用 hidden 承接累加的梯度：反向只穿过 transformer 一次
        return _ChunkedLoss(hidden, accumulated / count, total / count), count

    # --- 训练 ---------------------------------------------------------------

    def train_block(self, model_id, records):
        """来源纯净的训练块：只接受 train split，逐条累积后一次更新。"""
        torch = self.torch
        sources = {r["source"] for r in records}
        if not records:
            raise ValueError("records 不能为空")
        if any(r["split"] != "train" for r in records):
            raise ValueError("train_block 只接受 train split")
        if len(sources) != 1:
            raise ValueError("训练块必须是来源纯净的，不能混合多个 source")
        self._activate(model_id)
        # from_pretrained 返回的模型默认是 eval 模式，而 HF 的 gradient
        # checkpointing 只在 training 模式下才应用——不切换就等于没开，
        # 且不会有任何提示（实测：开启前后峰值显存完全相同）。
        self.model.train()
        if self.gradient_checkpointing:
            self.model.config.use_cache = False        # 与重算冲突
        optimizer, scheduler = self.optimizers[model_id], self.schedulers[model_id]
        optimizer.zero_grad(set_to_none=True)

        processed = supervised = 0
        total = 0.0
        for record in records:
            ids, labels = self._encode(record["prompt"], record["completion"])
            loss, count = self._supervised_loss(ids, labels)
            if loss is None:
                raise ValueError(f"记录 {record['id']} 没有可监督 token；"
                                 "训练端与 manifest 的模板或 mask 规则不一致")
            if count != record["completion_tokens"]:
                raise ValueError(
                    f"记录 {record['id']} 的监督 token 数 {count} 与 manifest 的 "
                    f"{record['completion_tokens']} 不符；训练与整备的模板必须一致")
            total += float(loss.detach()) * count
            (loss * count).backward()          # 按 token 加权，等价于整块的 token 平均
            processed += ids.numel()
            supervised += count

        for p in self._adapter_params(self.names[model_id]):
            if p.grad is not None:
                p.grad /= supervised
        norm = float(torch.nn.utils.clip_grad_norm_(
            self._adapter_params(self.names[model_id]), float("inf")))
        optimizer.step()
        scheduler.step()
        self.steps[model_id] += 1
        outcome = {"model": model_id, "source": sources.pop(), "records": len(records),
                   "processed_tokens": processed, "completion_tokens": supervised,
                   "step": self.steps[model_id],
                   "learning_rate": scheduler.get_last_lr()[0],
                   "grad_norm": norm, "loss": total / supervised}
        self.call_log.append({
            "call": "train_block", "model": model_id, "source": outcome["source"],
            "split": "train", "records": len(records), "outputs": 1,
            "completion_tokens": supervised})
        return outcome

    # --- 生成与判分 ---------------------------------------------------------

    def evaluation_rng(self):
        """判分随机流的当前状态。clone 是必须的：torch 的 rng state 是张量，
        直接返回引用会被后续 set_rng_state 就地改写，公共随机数随之失效。"""
        return self._eval_rng.clone()

    def set_evaluation_rng(self, state):
        if state is not None:
            self._eval_rng = state.clone()

    @contextmanager
    def _isolated_rng(self):
        """把生成用的随机数与训练 RNG 隔离开。

        model.generate 会消耗全局 RNG，若不隔离，评估就会改变后续训练的
        随机序列——恢复检查点后轨迹将不可复现，而且这一点不会报错。

        ⚠ 这里存取的是 CPU 生成器（torch.get_rng_state）。在 GPU 上采样用的是
        CUDA 生成器，不受这里控制；评估脚本需要可复现的采样时另行设定 CUDA 种子
        （见 pipeline/score_test_extension.py 的 --eval-seed）。
        """
        torch = self.torch
        saved = torch.get_rng_state()
        torch.set_rng_state(self._eval_rng)
        try:
            yield
        finally:
            self._eval_rng = torch.get_rng_state()
            torch.set_rng_state(saved)

    def _sample_batch(self, prompts, source, greedy=False):
        """批量解码。单条自回归只有约 20 tok/s，一次评估动辄数千条，不批处理不可行。

        decoder-only 模型批量生成必须左填充：右填充会让 pad 落在续写位置之前，
        模型从 pad 开始接着写。

        source 决定生成预算，因此整批必须同源——批量解码同步推进，混源会让
        短来源按长来源的上限陪跑，也无法给它自己的预算。
        """
        torch = self.torch
        if not prompts:
            return []
        side = self.tokenizer.padding_side
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        try:
            batch = self.tokenizer(prompts, add_special_tokens=False, padding=True,
                                   return_tensors="pt")
            batch = {k: v.to(self.device) for k, v in batch.items()}
            options = dict(do_sample=False) if greedy else dict(
                do_sample=True, temperature=self.temperature, top_k=0, top_p=1.0)
            with torch.no_grad(), self._isolated_rng():
                out = self.model.generate(**batch, max_new_tokens=self.max_new_tokens[source],
                                          pad_token_id=pad, **options)
            width = batch["input_ids"].shape[1]
            return [self.tokenizer.decode(row[width:], skip_special_tokens=True)
                    for row in out]
        finally:
            self.tokenizer.padding_side = side

    def _verify(self, record, answer):
        kind = record.get("reference", {}).get("kind")
        verifier = self.verifiers.get(kind)
        if verifier is None:
            raise ValueError(f"没有为 reference.kind={kind!r} 注册判分器")
        score = verifier(answer, record["reference"])
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError("verifier 分数必须落在 [0, 1]")
        return float(score)

    def evaluate(self, model_id, records, *, decoding="stochastic"):
        """生成并判分，返回逐样本分数；不更新任何状态。

        decoding="stochastic" 是本包全部评估（探针与 test 评价）使用的协议；
        "greedy" 是确定性解码，本包的流水线不使用。两者不能混用，也不能按结果
        好坏择一上报。

        records 必须同源：生成预算逐来源取值，混源就无法确定该用哪一个。
        调用方本来就是按来源取 bank 的（_audit_row 取 banks["audit"][k]，test
        评价逐来源调用），这里把那个前提变成显式检查。
        """
        if decoding not in ("stochastic", "greedy"):
            raise ValueError("decoding 只能是 stochastic 或 greedy")
        self._activate(model_id)
        if any(r["split"] == "train" for r in records):
            raise ValueError("评估不能使用 train split")
        sources = {r["source"] for r in records}
        if len(sources) > 1:
            raise ValueError(f"evaluate 的 records 必须同源，收到 {sorted(sources)}")
        source = sources.pop() if sources else 0
        was_training, was_cache = self.model.training, self.model.config.use_cache
        self.model.eval()
        self.model.config.use_cache = True             # 生成要用 KV cache
        greedy = decoding == "greedy"
        try:
            scores = []
            for start in range(0, len(records), self.batch_size):
                chunk = records[start:start + self.batch_size]
                answers = self._sample_batch([r["prompt"] for r in chunk], source,
                                             greedy=greedy)
                scores.extend(self._verify(r, a) for r, a in zip(chunk, answers))
            self.call_log.append({
                "call": "evaluate", "model": model_id, "source": source,
                "split": records[0]["split"] if records else None,
                "records": len(records), "outputs": len(scores),
                "decoding": decoding})
            return scores
        finally:
            self.model.train(was_training)
            self.model.config.use_cache = was_cache

    # --- 状态 ---------------------------------------------------------------

    def _next_blob(self):
        """下一个未被占用的快照文件名。

        序号必须**单调**：discard 会删掉中间的文件，若按"现有文件个数"命名，
        删掉一份之后新快照就会重用一个仍被检查点引用的名字，把它悄悄覆盖掉。
        因此从现有最大序号往后找，且只增不减。
        """
        used = []
        for path in self.state_dir.glob("backend-*.pt"):
            try:
                used.append(int(path.stem.split("-")[1]))
            except (IndexError, ValueError):              # 手工放进来的文件，跳过
                continue
        self._snapshot_seq = max([self._snapshot_seq, *used, -1]) + 1
        return self.state_dir / f"backend-{self._snapshot_seq}.pt"

    def snapshot(self):
        """张量写盘，只返回可 JSON 序列化的元数据与摘要。"""
        torch = self.torch
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "adapters": [{n: v.cpu() for n, v in self.model.state_dict().items()
                          if f".{name}." in n or n.endswith(f".{name}")}
                         for name in self.names],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "schedulers": [s.state_dict() for s in self.schedulers],
            "steps": list(self.steps),
            "torch_rng": torch.get_rng_state(),
            "eval_rng": self._eval_rng,
            "python_rng": self.rng.getstate()}
        blob = self._next_blob()
        torch.save(payload, blob)
        digest = hashlib.sha256(blob.read_bytes()).hexdigest()
        return {"format": "lwts-lora-v1", "path": str(blob), "sha256": digest,
                "steps": list(self.steps)}

    def discard(self, state):
        """删掉一份不再需要的快照文件（区间回滚用完即弃）。

        每份是全部 adapter 的参数加各自的 AdamW 动量（r=32、两个 adapter 时约
        338 MiB）；区间回滚每轮存一份，不回收就会一直堆积。
        """
        if not isinstance(state, dict) or state.get("format") != "lwts-lora-v1":
            return
        Path(state["path"]).unlink(missing_ok=True)

    def restore(self, state):
        """按元数据里的路径恢复全部状态；校验摘要，拒绝被改动的文件。"""
        torch = self.torch
        if not isinstance(state, dict) or state.get("format") != "lwts-lora-v1":
            raise ValueError("不支持的后端状态格式")
        blob = Path(state["path"])
        if not blob.exists():
            raise FileNotFoundError(f"后端状态文件缺失：{blob}")
        if hashlib.sha256(blob.read_bytes()).hexdigest() != state["sha256"]:
            raise ValueError("后端状态文件摘要不匹配：已被修改或损坏")
        payload = torch.load(blob, weights_only=False)
        for adapter in payload["adapters"]:
            self.model.load_state_dict(adapter, strict=False)
        for optimizer, saved in zip(self.optimizers, payload["optimizers"]):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(self.schedulers, payload["schedulers"]):
            scheduler.load_state_dict(saved)
        self.steps = list(payload["steps"])
        torch.set_rng_state(payload["torch_rng"])
        self._eval_rng = payload["eval_rng"]
        self.rng.setstate(payload["python_rng"])
