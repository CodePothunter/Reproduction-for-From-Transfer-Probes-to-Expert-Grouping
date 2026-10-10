"""两个候选之间专家参数的距离（为本报告新写；CPU，只读）。

事后诊断：C0 与 C3（附录候选）里有两个专家拿到完全相同的块、服务相同的来源。如果训练是逐位
确定的，这两个专家在两次运行里的参数应当完全相同；它们之间的距离因此量出了"同样的数据、同样的
起点"下由 GPU 非确定性造成的差异，用训练位移（相对共同起点 W）作尺度。

距离：每个专家取后端快照里该专家的全部 LoRA 张量（adapters[e] 的所有键），转 fp32 后
拼成一个向量求 L2 范数之差。

输入是三个后端快照（W、C0 训练区间结束、C3 训练区间结束），每个约 0.7 GB，**不随包**；
本脚本记录它们的 sha256 以便核对。块矩阵取自 results/candidates/training_summary.json，
用来标出两次运行里块完全相同的专家。

用法：python analysis/param_distance.py --w <W 快照.pt> --c0 <C0 快照.pt> --c3 <C3 快照.pt> \
          [--summary results/candidates/training_summary.json] [--out results/analysis/PARAM-distance.json]
"""
import argparse, hashlib, json
from pathlib import Path

import torch


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def l2(a, b):
    """两组同键张量之差的 L2 范数（fp32，逐键平方和再开方）。"""
    if list(a) != list(b):
        raise SystemExit("两份快照的 LoRA 张量键不一致")
    total = 0.0
    for key in a:
        total += float(((a[key].float() - b[key].float()) ** 2).sum())
    return total ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w", required=True, help="共同起点 W 的后端快照（不随包）")
    ap.add_argument("--c0", required=True, help="C0 训练区间结束时的后端快照（不随包）")
    ap.add_argument("--c3", required=True, help="C3 训练区间结束时的后端快照（不随包）")
    ap.add_argument("--summary", default="results/candidates/training_summary.json")
    ap.add_argument("--out", default="results/analysis/PARAM-distance.json")
    a = ap.parse_args()

    paths = {"W": a.w, "C0": a.c0, "C3": a.c3}
    digests = {k: sha256(p) for k, p in paths.items()}
    adapters = {k: torch.load(p, map_location="cpu", weights_only=False)["adapters"]
                for k, p in paths.items()}
    blocks = {c: v["blocks"] for c, v in
              json.loads(Path(a.summary).read_text(encoding="utf-8"))["candidates"].items()}
    experts = len(adapters["C0"])
    assert len(adapters["C3"]) == len(adapters["W"]) == experts

    rows = []
    for e in range(experts):
        d_c0_c3 = l2(adapters["C0"][e], adapters["C3"][e])
        d_c0_w = l2(adapters["C0"][e], adapters["W"][e])
        d_c3_w = l2(adapters["C3"][e], adapters["W"][e])
        same = blocks["C0"][e] == blocks["C3"][e]
        rows.append({"expert": e, "identical_blocks_in_C0_and_C3": same,
                     "blocks_C0": blocks["C0"][e], "blocks_C3": blocks["C3"][e],
                     "n_tensors": len(adapters["C0"][e]),
                     "dist_C0_C3": d_c0_c3, "dist_C0_W": d_c0_w, "dist_C3_W": d_c3_w,
                     "ratio_C0_C3_over_C0_W": d_c0_c3 / d_c0_w})
        print(f"e{e}  块相同 {'是' if same else '否'}  ‖C0−C3‖ {d_c0_c3:.4f}  "
              f"‖C0−W‖ {d_c0_w:.4f}  ‖C3−W‖ {d_c3_w:.4f}  比值 {d_c0_c3 / d_c0_w:.1%}")

    matched = [r for r in rows if r["identical_blocks_in_C0_and_C3"]]
    out = {"note": ("Post hoc diagnostic. L2 distance between expert LoRA parameters (all adapter "
                    "tensors of the expert, fp32) at the end of the C0 and C3 training intervals, "
                    "and each run's displacement from the shared warm-up state W. Experts with "
                    "identical blocks in C0 and C3 measure run-to-run differences from GPU "
                    "nondeterminism. The three input snapshots are not included in this package."),
           "inputs_sha256": digests,
           "experts": rows,
           "matched_experts": [r["expert"] for r in matched],
           "matched_dist_C0_C3_range": [min(r["dist_C0_C3"] for r in matched),
                                        max(r["dist_C0_C3"] for r in matched)],
           "matched_dist_C0_W_range": [min(r["dist_C0_W"] for r in matched),
                                       max(r["dist_C0_W"] for r in matched)],
           "matched_ratio_range": [min(r["ratio_C0_C3_over_C0_W"] for r in matched),
                                   max(r["ratio_C0_C3_over_C0_W"] for r in matched)],
           "unmatched_dist_C0_C3_range": [min(r["dist_C0_C3"] for r in rows if r not in matched),
                                          max(r["dist_C0_C3"] for r in rows if r not in matched)]}
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {dst}")


if __name__ == "__main__":
    main()
