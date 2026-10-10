"""删减等价性检验的场景：在 CPU 上按流水线脚本的方式重放证据路径。

tests/test_strip_equivalence.py 用这里的函数跑出轨迹，再与 tests/fixtures/ 下的
金标准逐项比对。金标准是用删减前的代码在**同样的场景**上生成的（只存记录 id
序列与数值），所以这里构造 Experiment 的方式必须与流水线脚本一致：

  S1  热身 W        uniform，一个区间（pipeline/run_pilot.py --method uniform
                    --rounds 1 --warmup 1）
  S2  候选 C0–C3    从 W 的检查点 fork，fixed，再训练一个区间（run_pilot.py
                    --method fixed --fork-from W --rounds 2 --warmup 1）
  S3  较早布局      W 的检查点改写成较早的布局（npz 非空、state 与日志带额外的键）
                    后再 fork，候选轨迹必须不变
  S4  单源迁移探针  每个专家测一次 audit 基线，再对每个来源 probe(blocks=16)
                    （pipeline/probe_transfer.py，两个专家，bank 截断）
  S5  test 评价     diagnostic 模式恢复候选检查点，按「每个来源由块数最多的专家
                    服务」的路由评 test（pipeline/score_test_extension.py）

两种后端：
  · SyntheticBackend：纯 Python 浮点，五个场景都跑；跨平台比对容差 1e-12。
  · LoRABackend + 两层微型 Llama：真实训练路径（AdamW、scheduler、分块 loss）。
    本包的测试只拿它跑不含生成的 S1–S3：含采样的场景在另一种 CPU 上可能因
    float32 舍入换掉某个采样 token，金标准就不再可比。分词器是在内存里构造的
    逐字节分词器，不依赖任何下载。

本模块只用本包 lwts 的接口。调用方可以给 Experiment 的设置追加键（extra），
给后端构造追加参数（backend_kwargs）——本包的测试两者都不传。
"""
import hashlib
import json
import random
from pathlib import Path

MODELS, SOURCES = 4, 7
SEED = 101
BLOCKS_PER_INTERVAL = 56            # 每区间 56 块：4 专家 × 14 块，7 来源 × 8 块
PROBE_MODELS, PROBE_BLOCKS, PROBE_BANK = 2, 16, 8

# 与 pipeline/run_candidates.sh 相同的四个整数块矩阵（行 = 专家，列 = 来源）
CANDIDATES = {
    "C0": [[0, 0, 7, 7, 0, 0, 0], [7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7], [1, 1, 1, 1, 8, 1, 1]],
    "C1": [[2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2]],
    "C2": [[0, 0, 0, 6, 8, 0, 0], [0, 8, 2, 2, 0, 0, 2], [0, 0, 0, 0, 0, 8, 6], [8, 0, 6, 0, 0, 0, 0]],
    "C3": [[0, 0, 0, 7, 7, 0, 0], [7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7], [1, 1, 8, 1, 1, 1, 1]],
}

# Experiment 的必填项，取值与 pipeline/run_pilot.py 的 PILOT_SETTINGS 相同
REQUIRED_VALUES = {"tau": 0.05, "eta": 0.5, "gamma": 0.5, "rho": 0.05}

# 日志里与训练有关的字段；其余字段不参与比对
LOG_FIELDS = ("round", "blocks", "processed_tokens", "completion_tokens",
              "block_tokens_realized", "allocation")
# 评估只在这两个切分上发生才属于证据路径（S4 读 audit，S5 读 test）
EVALUATED_SPLITS = ("audit", "test")


def routing(blocks):
    """每个来源由获得它最多块的专家服务，并列取编号小者（同 run_candidates 的规则）。"""
    return [max(range(len(blocks)), key=lambda i: (blocks[i][k], -i))
            for k in range(len(blocks[0]))]


def config(method, models, block_tokens, blocks, extra=None, **settings):
    values = dict(REQUIRED_VALUES, block_tokens=block_tokens,
                  tokens_per_interval=blocks * block_tokens, models=models,
                  sources=SOURCES, method=method, seed=SEED, warmup_rounds=1, **settings)
    values.update(extra or {})
    return {"experiment": values}


def tracing(cls):
    """给后端类加上调用记录：训练块与评估各自看过哪些记录、按什么顺序。"""
    class Tracing(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trained, self.evaluated = [], []

        def train_block(self, model_id, records):
            self.trained.append([model_id, records[0]["source"], [r["id"] for r in records]])
            return super().train_block(model_id, records)

        def evaluate(self, model_id, records, **kwargs):
            self.evaluated.append([model_id, records[0]["source"], records[0]["split"],
                                   [r["id"] for r in records]])
            return super().evaluate(model_id, records, **kwargs)

    Tracing.__name__ = f"Tracing{cls.__name__}"
    return Tracing


# --- 合成后端 -----------------------------------------------------------------

SYNTHETIC_BLOCK_TOKENS = 240
SYNTHETIC_COUNTS = {"train": 300, "feedback": 12, "route": 4, "audit": 12, "test": 10}


def synthetic_manifest():
    """7 来源；记录长度可变（60/80/120），块边界随游标漂移，块大小高于名义值。"""
    rng = random.Random(0)
    records = []
    for k in range(SOURCES):
        for split, count in SYNTHETIC_COUNTS.items():
            for i in range(count):
                completion = rng.choice([60, 80, 120])
                records.append({"id": f"{split}-{k}-{i:04d}", "group_id": f"g{k}-{split}-{i}",
                                "source": k, "split": split, "prompt": "p", "completion": "c",
                                "tokens": completion + 20, "completion_tokens": completion,
                                "reference": {"kind": "x"}, "metadata": {"dataset": "d"}})
    return {"schema_version": 1, "records": records}


def synthetic_factory(**backend_kwargs):
    from lwts.backend import SyntheticBackend
    cls = tracing(SyntheticBackend)

    def make(models, state_dir):                     # state_dir：合成后端不写盘
        return cls(models, SOURCES, seed=7, response=5e-4, **backend_kwargs)
    return make


# --- 微型 LoRA 后端 -------------------------------------------------------------

LORA_BLOCK_TOKENS = 64
LORA_COUNTS = {"train": 80, "feedback": 12, "route": 2, "audit": 8, "test": 6}
LORA_LEARNING_RATE = 0.02


def byte_tokenizer():
    """逐字节的 fast tokenizer（256 个字节符号 + 一个结束符），完全在内存里构造。

    LoRABackend 与数据层需要的是 offsets 与 pad/eos，与具体词表无关；用它代替
    真实分词器，金标准就不随分词器文件的版本漂移，也不需要联网或本地缓存。
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {symbol: index for index, symbol in enumerate(alphabet)}
    vocab["<|endoftext|>"] = len(vocab)
    core = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    core.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>")


def lora_manifest(tokenizer):
    """走一遍数据层（prepare_records）得到 manifest 口径的 token 数，再按 id 顺序分切分。"""
    from lwts.datasets import prepare_records
    per_source = sum(LORA_COUNTS.values())
    raw = [{"prompt": f"Q{k}-{i}: {i}+{k}=?\n\nSolution:\n", "problem": f"q{k}-{i}",
            "source": k, "reference": {"kind": "math", "answer": str(i + k)},
            "completion": f"the answer is {i + k} " + "so " * ((7 * i + k) % 5)}
           for k in range(SOURCES) for i in range(per_source)]
    rows = prepare_records(raw, tokenizer, dataset="tiny")
    records, seen = [], {}
    for row in sorted(rows, key=lambda r: r["id"]):
        k = row["source"]
        n = seen.setdefault(k, 0)
        seen[k] += 1
        bound = 0
        for split, count in LORA_COUNTS.items():
            bound += count
            if n < bound:
                records.append(dict(row, split=split))
                break
    return {"schema_version": 1, "records": records}


def parity_verifier(answer, reference):
    """确定性的 0/1 判分：只要求分数随答案变化，不要求有意义。"""
    return float(sum(map(ord, answer)) % 2)


def lora_factory(tokenizer, **backend_kwargs):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from lwts.lora import LoRABackend
    cls = tracing(LoRABackend)
    # 单线程：CPU 上的归约顺序随线程数变化，固定它金标准才能逐位复现
    torch.set_num_threads(1)

    def make(models, state_dir):
        torch.manual_seed(1234)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4))
        base.requires_grad_(False)
        lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lora, adapter_name="expert_0")
        for i in range(1, models):
            model.add_adapter(f"expert_{i}", lora)
        # 学习率取得大：微型模型上 1e-4 的更新改变不了任何一次采样，探针与评价的
        # 比对就只是在比两份全零的翻转。
        return cls(model, tokenizer, models=models, sources=SOURCES, state_dir=state_dir,
                   learning_rate=LORA_LEARNING_RATE, max_new_tokens=4, seed=0, batch_size=4,
                   verifiers={"math": parity_verifier}, **backend_kwargs)
    return make


def seed_evaluation(backend, seed):
    """评估前设定采样随机流：GPU 上的 test 评价用 --eval-seed 设 CUDA 种子，这里是
    它在 CPU 上的对应。合成后端的评估不消耗随机数，无需设定。"""
    if hasattr(backend, "_eval_rng"):
        import torch
        backend.set_evaluation_rng(torch.Generator().manual_seed(seed).get_state())


# --- 状态视图 -------------------------------------------------------------------

def _is_tensor(value):
    return type(value).__module__.split(".")[0] == "torch" and hasattr(value, "detach")


def summarize(value):
    """把后端快照里的张量换成可存进 JSON 的摘要：dtype、形状、字节 sha256、和与平方和。

    不含张量的值原样返回，因此合成后端的场景不需要安装 torch。
    """
    if _is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        wide = tensor.double()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                "sum": float(wide.sum()), "sumsq": float((wide * wide).sum())}
    if isinstance(value, float):                      # 含 numpy.float64
        return float(value)
    if isinstance(value, dict):
        return {str(k): summarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(v) for v in value]
    return value


def backend_view(backend):
    """后端的完整训练状态。LoRA 的快照文件读完即删（它不被任何检查点引用）。"""
    snapshot = backend.snapshot()
    if snapshot.get("format") == "lwts-synthetic-v1":
        return {k: snapshot[k] for k in ("params", "velocity", "steps", "rng")}
    import torch
    try:
        blob = torch.load(snapshot["path"], map_location="cpu", weights_only=False)
    finally:
        backend.discard(snapshot)
    return {k: summarize(blob[k]) for k in ("adapters", "optimizers", "schedulers", "steps",
                                            "torch_rng", "python_rng", "eval_rng")}


def training_logs(logs):
    return [{k: entry[k] for k in LOG_FIELDS} for entry in logs if "blocks" in entry]


def live_view(experiment):
    return {"round": experiment.round,
            "cursors": {str(k): v for k, v in sorted(experiment.cursors.items())},
            "allocation": experiment.allocation,
            "logs": training_logs(experiment.logs),
            "backend": backend_view(experiment.backend)}


def checkpoint_view(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    state = payload["state"]
    return {"round": state["round"], "cursors": state["cursors"],
            "allocation": state["allocation"],
            "identity": {k: payload["identity"][k]
                         for k in ("manifest", "method", "models", "sources")},
            "logs": training_logs(state["logs"])}


def evaluated(backend):
    return [entry for entry in backend.evaluated if entry[2] in EVALUATED_SPLITS]


def fork_consistent(w_path, echo_path):
    """fork 起点一致性：LoRA 用 lwts.guards（与 run_pilot 相同）；合成后端直接比快照。"""
    w = json.loads(Path(w_path).read_text(encoding="utf-8"))["state"]
    echo = json.loads(Path(echo_path).read_text(encoding="utf-8"))["state"]
    if w["backend"].get("format") == "lwts-synthetic-v1":
        keys = ("params", "velocity", "steps", "rng")
        return ({k: w["backend"][k] for k in keys} == {k: echo["backend"][k] for k in keys}
                and w["cursors"] == echo["cursors"])
    from lwts.guards import check_fork_consistency, common_training_state
    return check_fork_consistency(common_training_state(w_path),
                                  {"fork": common_training_state(echo_path)})


# --- 场景 ---------------------------------------------------------------------

def warmup(make_backend, manifest, root, block_tokens, extra=None, experiment_class=None):
    """S1：热身 W。返回 (视图, 检查点路径)。"""
    from lwts.experiment import Experiment
    root = Path(root)
    backend = make_backend(MODELS, str(root / "W" / "state"))
    experiment = (experiment_class or Experiment)(
        config("uniform", MODELS, block_tokens, BLOCKS_PER_INTERVAL, extra, fixed_blocks=None),
        manifest, backend)
    experiment.run(1)
    path = root / "W" / "checkpoint.json"
    experiment.checkpoint(str(path))
    return {"trace": backend.trained, "evaluated": evaluated(backend),
            "state": live_view(experiment), "checkpoint": checkpoint_view(path)}, path


def candidates(make_backend, manifest, w_path, root, block_tokens, extra=None):
    """S2：从 W fork 出 C0–C3，各训练一个区间并存 checkpoint-r2.json。"""
    from lwts.experiment import Experiment
    root = Path(root)
    out = {}
    for name, blocks in CANDIDATES.items():
        backend = make_backend(MODELS, str(root / name / "state"))
        experiment = Experiment(
            config("fixed", MODELS, block_tokens, BLOCKS_PER_INTERVAL, extra, fixed_blocks=blocks),
            manifest, backend)
        resumed = experiment.resume(str(w_path), mode="fork")
        echo = root / name / "checkpoint-resumed.json"
        experiment.checkpoint(str(echo))
        consistent = fork_consistent(w_path, echo)
        experiment.run(2)
        path = root / name / "checkpoint-r2.json"
        experiment.checkpoint(str(path))
        out[name] = {"resumed_round": resumed, "fork_consistent": consistent,
                     "trace": backend.trained, "evaluated": evaluated(backend),
                     "state": live_view(experiment), "checkpoint": checkpoint_view(path)}
    return out


def older_layout(w_path, dest):
    """把 W 的检查点改写成较早的布局：npz 里放数组，state 与日志带额外的键。

    恢复时 npz 只按索引核对存在性、sha256 与键集，不载入；state 里只读后端、
    游标、分配、轮次与日志。所以额外内容无论叫什么、是什么值，候选轨迹都不能变。
    """
    import numpy
    payload = json.loads(Path(w_path).read_text(encoding="utf-8"))
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = Path(str(dest) + ".older.npz")
    arrays = {f"{i}.{field}": numpy.full(9, 0.25 * (i + 1), dtype=numpy.float32)
              for i in range(MODELS) for field in ("a", "b")}
    numpy.savez(blob, **arrays)
    payload["vectors"] = {"path": blob.name, "keys": sorted(arrays),
                          "sha256": hashlib.sha256(blob.read_bytes()).hexdigest()}
    state = payload["state"]
    state.update({"extra_table": [[0.5] * SOURCES for _ in range(MODELS)],
                  "extra_counts": {"generated": 7}, "extra_list": [1, 2, 3]})
    for entry in state["logs"]:
        if "blocks" in entry:
            entry.update({"extra_value": 0.125, "extra_table": [[0.5] * SOURCES] * MODELS})
    state["logs"].append({"round": state["round"], "extra_interval": {"tokens": 1}})
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return dest


def probes(make_backend, manifest, root, block_tokens, method="uniform", extra=None,
           bank=PROBE_BANK, blocks=PROBE_BLOCKS):
    """S4：两个专家，每个测一次 audit 基线，再逐来源 probe；最后状态必须完全回滚。"""
    from lwts.experiment import Experiment
    backend = make_backend(PROBE_MODELS, str(Path(root) / "probe" / "state"))
    experiment = Experiment(config(method, PROBE_MODELS, block_tokens, PROBE_BLOCKS, extra),
                            manifest, backend)
    for split in ("feedback", "audit"):              # 同 probe_transfer.py 的 --bank 截断
        experiment.banks[split] = {k: rows[:bank] for k, rows in experiment.banks[split].items()}
    before = live_view(experiment)
    baselines, cells = [], []
    for model_id in range(PROBE_MODELS):
        baseline = experiment.audit_baseline(model_id)
        baselines.append({"model": baseline["model"], "row": baseline["row"],
                          "detail": baseline["detail"]})
        for k in range(SOURCES):
            cells.append(experiment.probe(model_id, k, blocks=blocks, baseline=baseline))
    return {"baselines": baselines, "probes": summarize(cells), "trace": backend.trained,
            "evaluated": evaluated(backend), "state_before": before,
            "state_after": live_view(experiment)}


def evaluation(make_backend, manifest, candidates_root, root, block_tokens, extra=None,
               eval_seed=None):
    """S5：diagnostic 恢复每个候选的 checkpoint-r2.json，按路由评 test 的全部题。"""
    from lwts.experiment import Experiment
    test = {k: sorted([r for r in manifest["records"]
                       if r["split"] == "test" and r["source"] == k], key=lambda r: r["id"])
            for k in range(SOURCES)}
    out = {}
    for name, blocks in CANDIDATES.items():
        backend = make_backend(MODELS, str(Path(root) / f"eval-{name}" / "state"))
        experiment = Experiment(
            config("fixed", MODELS, block_tokens, BLOCKS_PER_INTERVAL, extra, fixed_blocks=blocks),
            manifest, backend)
        experiment.resume(str(Path(candidates_root) / name / "checkpoint-r2.json"),
                          mode="diagnostic")
        # 评价的对象就是恢复出来的状态：先记下它，再评
        restored = {"round": experiment.round,
                    "cursors": {str(k): v for k, v in sorted(experiment.cursors.items())},
                    "backend": backend_view(backend)}
        if eval_seed is not None:
            seed_evaluation(backend, eval_seed)
        route = routing(blocks)
        rows = {str(k): [float(s) for s in backend.evaluate(route[k], test[k],
                                                            decoding="stochastic")]
                for k in range(SOURCES)}
        out[name] = {"restored": restored, "routing": route,
                     "test_ids": {str(k): [r["id"] for r in test[k]] for k in range(SOURCES)},
                     "test_rows": rows}
    return out
