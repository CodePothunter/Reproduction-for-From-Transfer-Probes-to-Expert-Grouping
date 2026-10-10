#!/usr/bin/env python
"""在单卡上训练若干区间：读 manifest → 起 LoRA 后端 → 跑区间 → 存检查点。

本报告用它做两件事（命令见 run_warmup.sh 与 run_candidates.sh）：

  · 热身 W：  --method uniform --rounds 1 --warmup 1，从基座 + 新初始化的 LoRA
              出发，每个专家在每个来源上训练相同块数。
  · 候选 C*： --method fixed --fixed-blocks <整数块矩阵> --rounds 2 --warmup 1
              --fork-from <W 的 checkpoint.json>，从 W 分叉，训练一个区间。

PILOT_SETTINGS 的全部键都保留，以便原始命令里的 --set 原样可用（--set 遇到
未知键会退出）。uniform 与 fixed 只读取 block_tokens 与 tokens_per_interval；
tau/eta/gamma/rho 只做必填校验，其余键不被读取。

⚠️ 判分会执行模型生成的代码。子进程 + 超时 + 资源上限只是健壮性隔离，
不是安全沙箱：只在可丢弃的机器上跑。
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# block_tokens 是单块 completion token 的下界：块至少是一条记录，DeepMath 解答
# 中位约 4500-5500 token。16384 时最差来源的实测块是名义值的 1.22-1.24 倍，紧贴
# 1.25 容限；32768 给出 1.12。本报告的命令都用 --set 显式给出 block_tokens 与
# tokens_per_interval（56 块/区间）。
BLOCK_TOKENS = 32768
BLOCKS_PER_INTERVAL = 16
PILOT_SETTINGS = {
    "block_tokens": BLOCK_TOKENS,   # 单块 completion token：下界，实测块见 plan
    "tokens_per_interval": BLOCK_TOKENS * BLOCKS_PER_INTERVAL,
    # 以下各项 uniform / fixed 都不读取（tau 只在探针的 portfolio_return 里用）。
    # 保留它们是为了让 Experiment 的必填校验与原始命令的 --set 原样通过。
    "tau": 0.05,
    "eta": 0.5,
    "gamma": 0.5,                   # 必须落在 (0, 1)，见 Experiment.__init__
    "rho": 0.05,
    "beta": 2.0,
    "scale_floor": 1.4e-6,
    "anchor_interval_tokens": BLOCK_TOKENS * BLOCKS_PER_INTERVAL,
}


def cap_banks(manifest, per_source, report):
    """把 feedback/audit 每来源截到 per_source 条，并**据实报告缺口**。

    bank 大小不是省时间的旋钮而是分辨率本身：二元判分下量化步长就是 1/bank，
    要分辨 0.05 量级的能力差距需要每来源 80 条。

    截断还决定 manifest 摘要（Experiment._identity 对记录 id 求摘要），所以
    fork / diagnostic 恢复时必须与产生检查点的那次运行用同一个 per_source。

    截断是静默的，缺口不是：某来源凑不满时必须显式报出来，否则那一列的
    分辨率悄悄低于其余来源，而诊断量只会显示"分辨不出"。
    """
    kept, dropped = [], 0
    seen = {}
    for row in sorted(manifest["records"], key=lambda r: r["id"]):
        key = (row["split"], row["source"])
        if row["split"] in ("feedback", "audit"):
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > per_source:
                dropped += 1
                continue
        kept.append(row)
    manifest["records"] = kept
    report(f"  feedback/audit 每来源目标 {per_source} 条，丢弃多余 {dropped} 条")
    short = {key: count for key, count in sorted(seen.items()) if count < per_source}
    for (split, source), count in short.items():
        step = 1.0 / count
        report(f"  ⚠ {split} 来源 {source} 只有 {count} 条（缺 {per_source - count}）："
               f"二元判分的量化步长 {step:.3f}，该来源的分辨率低于其余来源")
    if not short:
        report(f"  每来源均已满额，量化步长 {1.0 / per_source:.4f}")
    return manifest


def write_manifest(path, manifest):
    """按 JSONL 落盘：首行是头部，其余每行一条记录。

    manifest 的体积几乎全是 prompt/completion 文本（7 来源版本约 1.4 GB）；
    一次性 json.loads 的峰值内存是文件的两倍多，逐行解析则只留最终结构。
    """
    header = {k: v for k, v in manifest.items() if k != "records"}
    with path.open("w", encoding="utf-8") as out:
        out.write(json.dumps(header, ensure_ascii=False) + "\n")
        for record in manifest["records"]:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path.stat().st_size


def read_manifest(path):
    """读回 write_manifest 写下的 JSONL。"""
    with path.open(encoding="utf-8") as source:
        manifest = json.loads(next(source))
        manifest["records"] = [json.loads(line) for line in source]
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1,
                   help="前几个区间强制均匀热身")
    p.add_argument("--method", default="uniform", help="uniform 或 fixed")
    p.add_argument("--models", type=int, default=2)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=16384)
    # 逐来源取值：给 1 个则各来源相同，给 K 个则按来源 0..K-1 对号入座。
    # 数学侧 8192、代码侧 1024：训练会把数学生成长度放大数倍，而代码侧训练
    # 目标通常只有一两百 token（见 lwts.lora._budgets）。
    p.add_argument("--max-new-tokens", type=int, nargs="+",
                   default=[8192, 8192, 1024, 1024],
                   help="生成预算；给 1 个值则各来源相同，给 K 个则逐来源")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--bank", type=int, default=80,
                   help="feedback/audit 每来源条数。二元判分的量化步长是 1/bank，"
                        "80 对应 0.0125，才够分辨 0.05 量级的能力差距")
    p.add_argument("--manifest", required=True,
                   help="已建好的 manifest.jsonl（见 build_manifest_7src.py）")
    p.add_argument("--out", default="runs/pilot")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="覆盖 PILOT_SETTINGS 里的某项")
    p.add_argument("--fixed-blocks", default=None, metavar="JSON",
                   help="method=fixed 必填：整数块矩阵的 JSON，例如 [[0,0,7,7,0,0,0],...]")
    p.add_argument("--checkpoint-each-round", action="store_true",
                   help="每个区间末各存一份完整状态（候选评价读 checkpoint-r2.json）")
    p.add_argument("--resume", default=None, metavar="CHECKPOINT",
                   help="从同方法的检查点续跑（断点恢复）。恢复 adapter、optimizer、"
                        "scheduler、数据游标与 RNG")
    p.add_argument("--fork-from", default=None, metavar="CHECKPOINT",
                   help="从**共同热身**的检查点分叉出本方法。只比对 manifest/models/"
                        "sources，允许 method 不同；但拒绝从热身之后的检查点分叉")
    p.add_argument("--seed", type=int, default=17,
                   help="训练流洗牌种子；LoRA 初始化也由它派生（derive(seed, 'init')）")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def report(text=""):
        print(text, flush=True)
        lines.append(text)

    def section(title):
        report(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lwts.datasets import validate_manifest
    from lwts.experiment import Experiment
    from lwts.lora import LoRABackend
    from lwts.verifiers import default_verifiers

    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA 设备")

    settings = dict(PILOT_SETTINGS)
    for item in a.set:
        key, _, value = item.partition("=")
        if key not in settings:
            raise SystemExit(f"未知设置 {key}，可选：{sorted(settings)}")
        settings[key] = float(value) if "." in value or "e" in value.lower() else int(value)

    section("0. 环境与设置")
    report(f"  设备 {torch.cuda.get_device_name(0)}  "
           f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    report(f"  方法 {a.method}  专家数 {a.models}  区间数 {a.rounds}")
    report("  设置（PILOT_SETTINGS 与 --set；uniform/fixed 只读块尺度两项）：")
    for key, value in settings.items():
        report(f"       {key:26} {value}")

    section("1. 读 manifest")
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    target = out / "manifest.jsonl"
    source_manifest = Path(a.manifest)
    manifest = read_manifest(source_manifest)
    report(f"  读入 manifest：{source_manifest}")
    # 截断决定 manifest 摘要：fork / diagnostic 恢复要求它与产生检查点的运行一致。
    manifest = cap_banks(manifest, a.bank, report)
    audit = validate_manifest(manifest)
    report(f"  记录 {audit['records']}  题组 {audit['groups']}  状态 {audit['status']}")
    for split in ("train", "feedback", "audit"):
        counts = {k: v["records"] for k, v in audit["counts"][split].items()}
        report(f"    {split:9} {counts}")
    if source_manifest != target:
        size = write_manifest(target, manifest)
        report(f"  写入 manifest.jsonl（{size / 1024**2:.0f} MB）")

    section("2. 起 LoRA 后端")
    started = time.time()
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    base.requires_grad_(False)
    config = LoraConfig(r=a.rank, lora_alpha=2 * a.rank,
                        target_modules=list(LoRABackend.DEFAULT_TARGETS),
                        task_type="CAUSAL_LM")
    from lwts.seeds import derive, isolated_torch_rng
    with isolated_torch_rng(derive(a.seed, "init")):
        model = get_peft_model(base, config, adapter_name="expert_0")
        for i in range(1, a.models):
            model.add_adapter(f"expert_{i}", config)
    model.cuda()
    # sources 取自 manifest 头部，不写死：写死时喂 7 来源 manifest 不报错，多出来
    # 的来源会被静默忽略。
    # LoRA 初始化在隔离 RNG 上下文里按 derive(seed,'init') 播种：不播种时两次
    # nominally identical 的运行会拿到完全不同的 lora_A。
    backend = LoRABackend(model, tokenizer, models=a.models,
                          sources=len(manifest.get('sources') or [0, 1, 2, 3]),
                          state_dir=str(out / "state"), max_length=a.seq_len,
                          max_new_tokens=(a.max_new_tokens[0] if len(a.max_new_tokens) == 1
                                          else a.max_new_tokens),
                          batch_size=a.batch_size,
                          verifiers=default_verifiers())
    report(f"  加载 {time.time() - started:.0f}s   LoRA 覆盖 {backend.check_coverage()}")
    report(f"  权重显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    section("3. 初始化 Experiment")
    started = time.time()
    sources = len(manifest.get("sources") or [0, 1, 2, 3])
    report(f"  来源数 {sources}（取自 manifest 头部）  训练流种子 {a.seed}")
    experiment = Experiment(
        {"experiment": dict(settings, models=a.models, sources=sources, method=a.method,
                            seed=a.seed,
                            fixed_blocks=(json.loads(a.fixed_blocks) if a.fixed_blocks else None),
                            warmup_rounds=a.warmup)},
        manifest, backend)
    report(f"  初始化 {time.time() - started:.0f}s")

    section("3b. 块计划（实测，不是名义值）")
    plan = experiment.plan
    report(f"  block_tokens={plan['block_tokens']}（下界）  "
           f"每区间 {plan['blocks_per_interval']} 块 / {plan['cells']} 个 cell  "
           f"分配粒度 {plan['granularity']:.2%}")
    report(f"  {'来源':>4}{'记录':>7}{'token':>11}{'整流块数':>9}"
           f"{'块均值':>9}{'比名义':>8}{'条/块':>8}{'区间遍数':>9}")
    for source, entry in sorted(plan["measured"]["by_source"].items()):
        report(f"  {source:>4}{entry['records']:>7}{entry['tokens']:>11}"
               f"{entry['blocks_per_full_pass']:>9}{entry['mean']:>9.0f}"
               f"{entry['ratio_to_nominal']:>8.2f}{entry['records_per_block']:>8.1f}"
               f"{entry['passes_per_interval']:>9.2f}")
    for warning in plan["warnings"]:
        report(f"  ⚠ {warning}")

    if a.resume or a.fork_from:
        source = a.resume or a.fork_from
        section("3c. 恢复状态" + ("（分叉）" if a.fork_from else "（续跑）"))
        experiment.resume(source, mode="fork" if a.fork_from else "strict")
        report(f"  从 {source} 恢复到第 {experiment.round} 轮")
        report(f"  游标 {dict(sorted(experiment.cursors.items()))}")
        # 恢复保真度：**在跑任何区间之前**立刻再存一份。它与来源检查点逐项比对，
        # 就能证明 optimizer step/moments、scheduler、RNG、游标与当前分配都被
        # 正确恢复——不需要另跑一条长轨迹来对照。
        echo = out / "checkpoint-resumed.json"
        experiment.checkpoint(str(echo))
        report(f"  恢复后即时快照 → {echo.name}（与来源检查点逐项比对）")
        if a.fork_from:
            # fork 起点一致性：加载完成、执行任何新计算**之前**与共同 checkpoint 比对。
            # 只比共同训练状态。
            from lwts.guards import check_fork_consistency, common_training_state
            check_fork_consistency(common_training_state(a.fork_from),
                                   {a.method: common_training_state(str(echo))})
            report(f"  ✅ fork 起点与共同 checkpoint 逐位一致（adapters/optimizer/"
                   f"scheduler/训练 RNG/游标/步数）")

    section(f"4. 跑 {a.rounds} 个区间")
    for target in range(experiment.round + 1, a.rounds + 1):
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        experiment.run(target)
        log = experiment.logs[-1]
        report(f"\n  ── 区间 {target}  {time.time() - started:.0f}s  "
               f"峰值 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB ──")
        report(f"    分配块数 {log['blocks']}")
        report(f"    处理 token {log['processed_tokens']}  "
               f"监督 token {log['completion_tokens']}")
        realized = log["block_tokens_realized"]
        report(f"    实测块大小 均值 {realized['mean']:.0f}  中位 {realized['median']:.0f}"
               f"  区间 [{realized['min']}, {realized['max']}]"
               f"  （名义 {settings['block_tokens']}）")
        if experiment.allocation:
            report("    分配 q（行=专家）：")
            for i, row in enumerate(experiment.allocation):
                report(f"      expert_{i}  " + "  ".join(f"{q:.4f}" for q in row))
        if a.checkpoint_each_round:
            # 每个区间末存一份完整状态：候选评价读的是 checkpoint-r2.json。
            each = out / f"checkpoint-r{target}.json"
            experiment.checkpoint(str(each))
            report(f"    区间末状态 → {each.name}")

    # 产物先落盘，再打印体积：诊断代码不该有能力毁掉数据。
    experiment.checkpoint(str(out / "checkpoint.json"))
    (out / "logs.json").write_text(
        json.dumps(experiment.logs, ensure_ascii=False, indent=1), encoding="utf-8")
    # 调用流水：证明每个来源都真的进入了执行路径——配置里出现来源名字不算通过。
    (out / "call-log.json").write_text(
        json.dumps(backend.call_log, ensure_ascii=False), encoding="utf-8")
    (out / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    section("5. 产物体积")
    for name in ("manifest.jsonl", "logs.json", "checkpoint.json",
                 "checkpoint.json" + Experiment.VECTOR_SUFFIX):
        path = out / name
        if path.exists():
            report(f"  {name:32}{path.stat().st_size / 1024**2:>9.1f} MB")
    state = sum(f.stat().st_size for f in (out / "state").glob("*")) / 1024**2
    report(f"  {'state/（后端快照）':32}{state:>9.1f} MB")
    report(f"\n  产物写入 {out}/")
    (out / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
