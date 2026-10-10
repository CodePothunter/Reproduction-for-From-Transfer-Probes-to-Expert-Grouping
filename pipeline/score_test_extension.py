"""在固定的检查点与固定路由下，对 test 题做一次随机解码评价。

本报告的候选评价（命令见 run_eval.sh）：

  · 模型：候选训练完的 checkpoint-r2.json，不重训。
  · 路由：每个来源交给获得它最多块的专家（并列取编号小者），由 --routing 直接给定，
    在看到任何 test 结果之前确定。
  · 解码：stochastic（T=0.8，不截断），与探针相同。
  · 题目：test split 每来源按 id 排序后的第 [offset, offset+n) 条（本报告用 [0, 400)）。
    各候选用同一批题、同一个 --eval-seed。固定数量，跑完统一分析，不按中途结果
    停止或加题。

输出逐题结果与题目 ID，供按题配对的 bootstrap 分析使用。
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
NAMES = ["Integral", "Differential", "Abstract", "Linear", "Statistics",
         "algorithmic", "generic"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--routing", required=True, metavar="i,i,...",
                    help="每个来源由哪个专家服务（候选实验按预先登记的规则固定路由）")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--models", type=int, default=4)
    ap.add_argument("--offset", type=int, default=80)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, nargs="+",
                    default=[8192, 8192, 8192, 8192, 8192, 1024, 1024])
    ap.add_argument("--block-tokens", type=int, default=32768)
    ap.add_argument("--blocks-per-interval", type=int, default=56)
    ap.add_argument("--method", required=True)
    ap.add_argument("--fixed-blocks", default=None,
                    help="--method fixed 时必填：构造器要求它（评分本身不用）")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="评估前设定 CUDA RNG。采样在 GPU 上进行，用的是 CUDA 发生器；后端的"
                         "评估 RNG 只隔离 CPU 状态，控制不了采样。缺省时沿用进程默认的 CUDA "
                         "种子——同一检查点、同一批题、同一顺序重跑很可能得到完全相同的采样")
    ap.add_argument("--dump-answer-hashes", action="store_true",
                    help="记录每个生成答案的 sha256 前缀，用于判断两次采样是否相同（二元分数会碰巧相等）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from lwts.build import build_backend
    from lwts.experiment import Experiment
    from run_pilot import PILOT_SETTINGS, cap_banks, read_manifest

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    routing = [int(x) for x in a.routing.split(",")]
    # 与训练时的 --bank 80 一致：feedback/audit 截断决定 manifest 摘要，
    # diagnostic 模式的 identity 比对才能通过
    manifest = cap_banks(read_manifest(Path(a.manifest)), 80, lambda *_: None)
    sources = len(manifest["sources"])
    test = {k: sorted([r for r in manifest["records"]
                       if r["split"] == "test" and r["source"] == k], key=lambda r: r["id"])
            for k in range(sources)}
    bank = {k: rows[a.offset:a.offset + a.n] for k, rows in test.items()}
    short = {NAMES[k]: len(v) for k, v in bank.items() if len(v) < a.n}
    if short:
        raise SystemExit(f"test 不足 {a.n} 条：{short}")

    backend, _ = build_backend(a.model, models=a.models, sources=sources,
                               state_dir=str(out / "state"), max_new_tokens=a.max_new_tokens,
                               batch_size=a.batch_size)
    settings = dict(PILOT_SETTINGS, models=a.models, sources=sources, seed=a.seed,
                    tokens_per_interval=a.blocks_per_interval * a.block_tokens,
                    method=a.method, warmup_rounds=1,
                    fixed_blocks=(json.loads(a.fixed_blocks) if a.fixed_blocks else None))
    Experiment({"experiment": settings}, manifest, backend).resume(a.checkpoint,
                                                                   mode="diagnostic")
    hashes = []
    if a.dump_answer_hashes:
        import hashlib
        original_verify = backend._verify

        def recording_verify(record, answer):
            hashes.append(hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16])
            return original_verify(record, answer)
        backend._verify = recording_verify
    if a.eval_seed is not None:
        import torch
        torch.cuda.manual_seed_all(a.eval_seed)
    rows, per_source, t0 = {}, {}, time.time()
    for k in range(sources):
        scores = list(backend.evaluate(routing[k], bank[k], decoding="stochastic"))
        rows[str(k)] = scores
        per_source[NAMES[k]] = sum(scores) / len(scores)
        print(f"  {NAMES[k]:<13} 专家 {routing[k]}  新增 {len(scores)} 题  均分 "
              f"{per_source[NAMES[k]]:.4f}", flush=True)
    # 输出键与本报告随包的评价文件一致（"repeat" 恒为 False：本包不含重复测量模式）。
    payload = {"checkpoint": a.checkpoint, "method": a.method, "routing": routing,
               "offset": a.offset, "n": a.n, "decoding": "stochastic",
               "eval_seed": a.eval_seed, "repeat": False, "answer_hashes": hashes,
               "test_rows": rows, "test_ids": {str(k): [r["id"] for r in bank[k]]
                                                for k in range(sources)},
               "per_source": per_source, "seconds": round(time.time() - t0, 1),
               "generated": sum(len(v) for v in rows.values())}
    (out / "test_extension.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"写入 {out / 'test_extension.json'}  用时 {payload['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
