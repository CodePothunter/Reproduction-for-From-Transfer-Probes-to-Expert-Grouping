"""对照来源上的逐题一致率（为本报告新写；事后诊断，只用 results/candidates 的评价文件）。

C0 与 C3 里专家 e1、e2 拿到相同的块，并服务相同的四个来源（Integral、Differential、
algorithmic、generic）。在这四个来源上逐题比较两份评价的 0/1 分数，报告一致比例：

- same_seed：C0 与 C3，同一评价种子（2001 对 2001、2002 对 2002）；
- cross_seed：C0 与 C3，不同评价种子（2001 对 2002、2002 对 2001）；
- within_candidate：同一候选的两个种子之间（C0、C3 各自）。

用法：python analysis/control_agreement.py [--run results/candidates] [--out results/analysis]
"""
import argparse, json
from pathlib import Path

NAMES = ["Integral", "Differential", "Abstract", "Linear", "Statistics",
         "algorithmic", "generic"]
SEEDS = (2001, 2002)


def agreement(a, b):
    if len(a) != len(b):
        raise ValueError("题数不同，无法逐题比较")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/candidates")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    run = Path(a.run)
    ev = {(c, s): json.loads((run / f"{c}-eval-s{s}.json").read_text(encoding="utf-8"))
          for c in ("C0", "C3") for s in SEEDS}
    for (c, s), d in ev.items():
        assert d["eval_seed"] == s and d["test_ids"] == ev[("C0", 2001)]["test_ids"]
    # 对照来源：两次运行中块相同、路由到同一专家的来源
    summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))["candidates"]
    b0, b3 = summary["C0"]["blocks"], summary["C3"]["blocks"]
    r0, r3 = summary["C0"]["routing"], summary["C3"]["routing"]
    controls = [j for j in range(len(NAMES)) if r0[j] == r3[j] and b0[r0[j]] == b3[r3[j]]]
    experts = sorted({r0[j] for j in controls})

    def rows(c, s, j):
        return ev[(c, s)]["test_rows"][str(j)]

    sets = {
        "same_seed": [(f"C0 s{s} vs C3 s{s}", rows("C0", s, j), rows("C3", s, j), j)
                      for s in SEEDS for j in controls],
        "cross_seed": [(f"C0 s{s} vs C3 s{t}", rows("C0", s, j), rows("C3", t, j), j)
                       for s, t in ((2001, 2002), (2002, 2001)) for j in controls],
        "within_candidate": [(f"{c} s2001 vs {c} s2002", rows(c, 2001, j), rows(c, 2002, j), j)
                             for c in ("C0", "C3") for j in controls],
    }
    out = {"note": ("Post hoc diagnostic: per-item agreement of 0/1 scores on the control sources "
                    "(identical blocks and the same serving expert in C0 and C3)."),
           "control_sources": [NAMES[j] for j in controls], "control_experts": experts,
           "sets": {}}
    print(f"对照来源 {[NAMES[j] for j in controls]}（专家 {experts}）")
    for name, items in sets.items():
        vals = [{"pair": p, "source": NAMES[j], "agreement": agreement(x, y)}
                for p, x, y, j in items]
        lo = min(v["agreement"] for v in vals)
        hi = max(v["agreement"] for v in vals)
        out["sets"][name] = {"values": vals, "range": [lo, hi]}
        print(f"  {name:<17}一致率 {lo:.2%} – {hi:.2%}")
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "CONTROL-agreement.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"写入 {dst / 'CONTROL-agreement.json'}")


if __name__ == "__main__":
    main()
