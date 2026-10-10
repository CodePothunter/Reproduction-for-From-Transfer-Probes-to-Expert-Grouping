"""候选实际收益表（预先指定的主分析）。

每个候选每题的分数 = k 次采样（种子 2001..2000+k）的均值。对每个已完成的候选 C 报告
ΔU_all(C − C0)、ΔU_LS、ΔU_other5 与逐来源差，全部来自同一次联合配对 bootstrap：
每来源内按题目重采样，所有候选共用同一组索引。按预先指定的三档判读给出结论（δ_min = 2 pp）；
verdict() 另有第四档「区间完全低于 −δ_min」，不在预先指定的判读表中，本轮没有候选落入。

只分析 k 次采样全部完成的候选；未完成的跳过并打印说明。
⚠ 区间只含 test 抽样（题目 + 解码），条件于这一次训练，不含训练 seed 变化。
"""
import argparse, json, random
from pathlib import Path

NAMES = ["Integral", "Differential", "Abstract", "Linear", "Statistics",
         "algorithmic", "generic"]
LS = (3, 4)
DELTA_MIN = 0.02


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def summarize(per):
    return (sum(per) / 7, (per[LS[0]] + per[LS[1]]) / 2,
            sum(v for k, v in enumerate(per) if k not in LS) / 5)


def verdict(lo, hi):
    if lo > DELTA_MIN:
        return "存在值得利用的机会（下界 > +δ_min）"
    if lo > -DELTA_MIN and hi < DELTA_MIN:
        return "区间完全落在 (−δ_min, +δ_min)：此状态下难以提供设定的收益"
    if hi < -DELTA_MIN:
        return "区间完全低于 −δ_min：明显差于 C0"
    return "区间跨越 ±δ_min：精度不足，不定位算法问题"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/candidates")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    run = Path(a.run)
    seeds = [2000 + s for s in range(1, a.k + 1)]

    rows, meta = {}, {}
    for c in ("C0", "C1", "C2", "C3"):
        paths = [run / f"{c}-eval-s{s}.json" for s in seeds]
        if not all(p.exists() for p in paths):
            print(f"⚠ {c}：{sum(p.exists() for p in paths)}/{a.k} 次采样完成，跳过")
            continue
        res = [load(p) for p in paths]
        for r, s in zip(res, seeds):
            assert r["eval_seed"] == s and r["offset"] == 0 and r["n"] == 400, f"{c} 评价参数不符"
            assert r["routing"] == res[0]["routing"] and r["test_ids"] == res[0]["test_ids"]
        rows[c] = [[sum(r["test_rows"][str(k)][t] for r in res) / a.k
                    for t in range(len(res[0]["test_rows"][str(k)]))] for k in range(7)]
        flips = [sum(res[0]["test_rows"][str(k)][t] != res[1]["test_rows"][str(k)][t]
                     for t in range(len(res[0]["test_rows"][str(k)]))) / len(res[0]["test_rows"][str(k)])
                 for k in range(7)] if a.k >= 2 else None
        meta[c] = {"routing": res[0]["routing"], "test_ids": res[0]["test_ids"],
                   "seed_flip_rate": flips}
    assert "C0" in rows, "C0 未完成，无法比较"
    for c in rows:
        assert meta[c]["test_ids"] == meta["C0"]["test_ids"], f"{c} 与 C0 题目不一致"

    n = [len(rows["C0"][k]) for k in range(7)]
    mean = lambda v, idx: sum(v[t] for t in idx) / len(idx)            # noqa: E731
    point = {c: [sum(rows[c][k]) / n[k] for k in range(7)] for c in rows}
    rng = random.Random(20261007)
    others = [c for c in rows if c != "C0"]
    samples = {c: [] for c in others}
    for _ in range(a.draws):
        idx = [[rng.randrange(n[k]) for _ in range(n[k])] for k in range(7)]
        per = {c: [mean(rows[c][k], idx[k]) for k in range(7)] for c in rows}
        for c in others:
            samples[c].append([per[c][k] - per["C0"][k] for k in range(7)])

    def ci(vals):
        v = sorted(vals)
        return v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]

    out = {"k": a.k, "seeds": seeds, "n_per_source": n, "delta_min": DELTA_MIN,
           "point": {c: dict(zip(("U_all", "U_LS", "U_other5"), summarize(point[c])))
                     for c in rows},
           "per_source": point, "meta": {c: {kk: v for kk, v in m.items() if kk != "test_ids"}
                                          for c, m in meta.items()},
           "contrasts": {}}
    print(f"每来源 {n[0]} 题，每题 {a.k} 次采样（种子 {seeds}）")
    for c in rows:
        p = out["point"][c]
        print(f"  {c}  路由 {meta[c]['routing']}  U_all {p['U_all']:.4f}  "
              f"U_LS {p['U_LS']:.4f}  U_other5 {p['U_other5']:.4f}")
        if meta[c]["seed_flip_rate"]:
            print("      两次采样翻转率 " + " ".join(f"{x:.0%}" for x in meta[c]["seed_flip_rate"]))
    for c in others:
        d = [point[c][k] - point["C0"][k] for k in range(7)]
        agg = [summarize(s) for s in samples[c]]
        res = {}
        for j, name in enumerate(("dU_all", "dU_LS", "dU_other5")):
            res[name] = summarize(d)[j]
            res[name + "_ci95"] = ci([x[j] for x in agg])
        res["per_source"] = {NAMES[k]: {"d": d[k], "ci95": ci([s[k] for s in samples[c]])}
                             for k in range(7)}
        res["verdict"] = verdict(*res["dU_all_ci95"])
        out["contrasts"][f"{c}-C0"] = res
        print(f"\n  {c} − C0")
        for name in ("dU_all", "dU_LS", "dU_other5"):
            lo, hi = res[name + "_ci95"]
            print(f"    {name:<10}{res[name] * 100:+6.2f} pp  [{lo * 100:+.2f}, {hi * 100:+.2f}]")
        for nm, v in res["per_source"].items():
            lo, hi = v["ci95"]
            print(f"      {nm:<13}{v['d'] * 100:+6.2f}  [{lo * 100:+.2f}, {hi * 100:+.2f}]")
        print(f"    判读：{res['verdict']}")
    dst = Path(a.out) / "CANDIDATES-analysis.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {dst}")


if __name__ == "__main__":
    main()
