"""标签参照比较的补充分析——事后分析，非事先指定。

⚠ 本脚本的全部分析都是在看到候选实际收益表（`analysis/analyze_candidates.py` 的输出
`results/analysis/CANDIDATES-analysis.txt`）**之后**才加的，不属于预先指定协议中的
比较；论文中一律标为 "added after observing results"。事先指定的只有 Cj − C0 的 ΔU_all 及其 δ_min 判读。

角色：C0 是标签参照（label reference），由标签树与预算配平唯一决定，方法拿不到；
C1 是不用任何信息的默认分配（每格 2 块，块矩阵路由下全部来源由 expert 0 服务）；
C2 由 state0 探针张量（s17）上的运输 LP 导出；C3 是在 C0 上对调 Abstract 与 Statistics。

三部分（数据与重采样方式同 analyze_candidates.py：每题分数 = k 次采样均值）：

(a) 配对差 C2 − C1、C3 − C1（U_all、U_LS、U_other5）：点估计 + 95% 联合配对 bootstrap
    区间。2000 次，每来源内按题重采样，四个候选共用同一组索引；分位数取法与
    analyze_candidates.py 相同（第 ⌊0.025B⌋ 与 ⌊0.975B⌋ 个次序统计量）。
(b) 不依赖权重的流量敏感性：任意流量比例 w（非负、和为 1）下 ΔU_w = Σ_j w_j Δ_j 是 Δ 的
    线性函数，在单纯形上于顶点取极值，可达集合恰为 [min_j Δ_j, max_j Δ_j]。报告各候选
    相对 C0 的逐来源 Δ 与该范围。只是点估计，不带区间。
(c) 3 候选 × 7 来源共 21 个逐线差 Cj − C0 的同时 95% 区间（max-|t| bootstrap）：
    t_b = max_{c,j} |D*_{b,cj} − D̂_cj| / se_cj，se 取 bootstrap 标准差；临界值 q 取 t_b 的
    第 ⌊0.95B⌋ 个次序统计量，区间 D̂ ± q·se。报告同时区间整段 < 0 的线。

蒙特卡洛稳定性：主结果固定用种子 20261009。另用主分析
种子 20261007（与主表完全相同的重采样索引）以及种子 1..N 重跑 (a)(c)，报告临界值与
"整段 < 0"的判定是否随 bootstrap 种子改变——上界贴近 0 的线不能只凭一个种子下结论。

自检：用种子 20261007 按 analyze_candidates.py 的算术顺序复算 Cj − C0 的区间，须与
CANDIDATES-analysis.json 逐项一致（容差 1e-12），否则说明重采样实现与主分析不同，直接报错。

⚠ 区间只含 test 抽样（题目 + 解码），条件于每个候选的那一次训练，不含训练随机性。
"""
import argparse, json, math, random, statistics
from fractions import Fraction
from pathlib import Path

NAMES = ["Integral", "Differential", "Abstract", "Linear", "Statistics",
         "algorithmic", "generic"]
SHORT = ["Int", "Diff", "Abs", "Lin", "Stat", "alg", "gen"]
K = len(NAMES)
LS = (3, 4)
DELTA_MIN = 0.02
CANDS = ("C0", "C1", "C2", "C3")
VS_REF = ("C1", "C2", "C3")                 # 与标签参照 C0 比较
PAIRS = (("C2", "C1"), ("C3", "C1"))        # 与不用信息的默认 C1 比较
PRIMARY_SEED = 20261009                     # 本事后分析的主种子
MAIN_SEED = 20261007                        # analyze_candidates.py 的种子
LEVEL = 0.95


# ---------------------------------------------------------------- 纯函数

def summarize(per):
    """(U_all, U_LS, U_other5)；算术顺序与 analyze_candidates.summarize 完全相同。"""
    return (sum(per) / K, (per[LS[0]] + per[LS[1]]) / 2,
            sum(v for k, v in enumerate(per) if k not in LS) / 5)


def percentile_ci(vals, level=LEVEL):
    """百分位区间；取法与 analyze_candidates.ci 相同。"""
    v = sorted(vals)
    a = (1 - level) / 2
    return v[int(a * len(v))], v[min(len(v) - 1, int((1 - a) * len(v)))]


def weighted_delta(deltas, weights):
    """流量比例 weights 下的效用差 Σ_j w_j Δ_j。weights 须非负且和为 1。"""
    if len(weights) != len(deltas):
        raise ValueError("权重与来源数不符")
    if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("权重须非负且和为 1")
    return sum(w * d for w, d in zip(weights, deltas))


def reachable_range(deltas):
    """任意流量比例下 Σ_j w_j Δ_j 的可达集合 [min Δ, max Δ]。

    线性函数在单纯形上的像是其顶点像的凸包，故可达集合恰为逐项最小到最大；两端由把全部
    流量放在单个来源上取到。返回 (lo, argmin, hi, argmax)，并列取编号小者。
    """
    if not deltas:
        raise ValueError("deltas 不能为空")
    j_lo = min(range(len(deltas)), key=lambda j: (deltas[j], j))
    j_hi = max(range(len(deltas)), key=lambda j: (deltas[j], -j))
    return deltas[j_lo], j_lo, deltas[j_hi], j_hi


def joint_bootstrap(rows, draws, seed):
    """联合配对 bootstrap：每次每来源内按题有放回重采样，所有候选共用同一组索引。

    rows[c][j][t] 为候选 c、来源 j、第 t 题的分数。返回 per[c][b][j]（第 b 次重采样下的
    来源均值）。随机数的消耗顺序与 analyze_candidates.py 相同（逐次、逐来源、逐题），
    同一种子给出逐位相同的索引。
    """
    names = list(rows)
    ref = rows[names[0]]
    n = [len(ref[j]) for j in range(len(ref))]
    for c in names:
        if [len(col) for col in rows[c]] != n:
            raise ValueError(f"{c} 的题数与 {names[0]} 不一致，无法配对")
    rng = random.Random(seed)
    per = {c: [] for c in names}
    for _ in range(draws):
        idx = [[rng.randrange(n[j]) for _ in range(n[j])] for j in range(len(n))]
        for c in names:
            per[c].append([sum(rows[c][j][t] for t in idx[j]) / len(idx[j])
                           for j in range(len(n))])
    return per


def contrast_samples(per, a, b):
    """逐次重采样的逐来源差 a − b（与主分析相同：先逐来源相减，再汇总）。"""
    return [[x - y for x, y in zip(pa, pb)] for pa, pb in zip(per[a], per[b])]


def bootstrap_se(samples):
    """samples[b][m] → 每个统计量的 bootstrap 标准差（总体标准差）。"""
    m = len(samples[0])
    return [statistics.pstdev([s[i] for s in samples]) for i in range(m)]


def simultaneous_critical(samples, point, se, level=LEVEL):
    """max-|t| 同时临界值。

    samples[b][m] 为第 b 次重采样的 m 个统计量，point[m] 为点估计，se[m] 为标准差。
    t_b = max_i |samples[b][i] − point[i]| / se[i]；返回 t_b 的第 ⌊level·B⌋ 个次序统计量。
    se 为 0 的统计量：偏差也为 0 时记 0，否则记 +∞。
    """
    def t(dev, s):
        if s > 0:
            return abs(dev) / s
        return 0.0 if dev == 0 else math.inf
    maxt = sorted(max(t(row[i] - point[i], se[i]) for i in range(len(point)))
                  for row in samples)
    return maxt[min(len(maxt) - 1, int(level * len(maxt)))]


def simultaneous_intervals(point, se, q):
    """D̂ ± q·se。"""
    return [(p - q * s, p + q * s) for p, s in zip(point, se)]


def exact_point(rows_c, rows_ref):
    """逐来源点估计差的精确值（分数为 0/1 的 k 次采样均值，均值差是有理数）。"""
    return [Fraction(sum(Fraction(x) for x in a) - sum(Fraction(x) for x in b), len(a))
            for a, b in zip(rows_c, rows_ref)]


# ---------------------------------------------------------------- 数据与分析

def load_rows(run, k):
    seeds = [2000 + s for s in range(1, k + 1)]
    rows, meta = {}, {}
    for c in CANDS:
        paths = [run / f"{c}-eval-s{s}.json" for s in seeds]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{c} 缺评价文件：{missing}")
        res = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        for r, s in zip(res, seeds):
            assert r["eval_seed"] == s and r["offset"] == 0 and r["n"] == 400, f"{c} 评价参数不符"
            assert r["routing"] == res[0]["routing"] and r["test_ids"] == res[0]["test_ids"]
        rows[c] = [[sum(r["test_rows"][str(j)][t] for r in res) / k
                    for t in range(len(res[0]["test_rows"][str(j)]))] for j in range(K)]
        meta[c] = {"routing": res[0]["routing"], "test_ids": res[0]["test_ids"]}
    for c in CANDS:
        assert meta[c]["test_ids"] == meta["C0"]["test_ids"], f"{c} 与 C0 题目不一致"
    return rows, {c: m["routing"] for c, m in meta.items()}, seeds


def pair_block(per, rows, a, b):
    """a − b 的 U_all / U_LS / U_other5：点估计 + 百分位区间。"""
    n = [len(col) for col in rows["C0"]]
    point = [sum(rows[a][j]) / n[j] - sum(rows[b][j]) / n[j] for j in range(K)]
    agg = [summarize(s) for s in contrast_samples(per, a, b)]
    out = {}
    for i, name in enumerate(("dU_all", "dU_LS", "dU_other5")):
        out[name] = summarize(point)[i]
        out[name + "_ci95"] = list(percentile_ci([x[i] for x in agg]))
    lo, hi = out["dU_all_ci95"]
    out["dU_all_ci_within_pm_delta_min"] = lo > -DELTA_MIN and hi < DELTA_MIN
    return out


def simultaneous_block(per, rows):
    """21 个逐线差 Cj − C0 的同时区间。"""
    point = [float(d) for c in VS_REF for d in exact_point(rows[c], rows["C0"])]
    samples = [sum(rows_b, []) for rows_b in zip(*(contrast_samples(per, c, "C0")
                                                   for c in VS_REF))]
    se = bootstrap_se(samples)
    q = simultaneous_critical(samples, point, se)
    ivs = simultaneous_intervals(point, se, q)
    lines = {}
    for m, (c, j) in enumerate((c, j) for c in VS_REF for j in range(K)):
        lines.setdefault(c, {})[NAMES[j]] = {"d": point[m], "se": se[m],
                                             "lo": ivs[m][0], "hi": ivs[m][1]}
    below = [f"{c} {nm}" for c in VS_REF for nm, v in lines[c].items() if v["hi"] < 0]
    above = [f"{c} {nm}" for c in VS_REF for nm, v in lines[c].items() if v["lo"] > 0]
    max_hi = {c: max(((v["hi"], nm) for nm, v in lines[c].items())) for c in VS_REF}
    return {"critical": q, "lines": lines, "entirely_below_zero": below,
            "entirely_above_zero": above,
            "max_upper": {c: {"hi": h, "source": nm} for c, (h, nm) in max_hi.items()}}


def self_check(per, rows, reference):
    """种子 20261007 的复算须与 CANDIDATES-analysis.json 逐项一致。"""
    n = [len(col) for col in rows["C0"]]
    point = {c: [sum(rows[c][j]) / n[j] for j in range(K)] for c in CANDS}
    worst = 0.0
    for c in VS_REF:
        ref = reference["contrasts"][f"{c}-C0"]
        samples = contrast_samples(per, c, "C0")
        d = [point[c][j] - point["C0"][j] for j in range(K)]
        agg = [summarize(s) for s in samples]
        for i, name in enumerate(("dU_all", "dU_LS", "dU_other5")):
            got = (summarize(d)[i], *percentile_ci([x[i] for x in agg]))
            want = (ref[name], *ref[name + "_ci95"])
            worst = max(worst, *(abs(g - w) for g, w in zip(got, want)))
        for j in range(K):
            got = (d[j], *percentile_ci([s[j] for s in samples]))
            want = (ref["per_source"][NAMES[j]]["d"], *ref["per_source"][NAMES[j]]["ci95"])
            worst = max(worst, *(abs(g - w) for g, w in zip(got, want)))
    if worst > 1e-12:
        raise SystemExit(f"自检失败：种子 {MAIN_SEED} 的复算与 CANDIDATES-analysis.json "
                         f"最大偏差 {worst:.3g}")
    return worst


def pp(x, nd=2):
    return f"{100 * x:+.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="results/candidates")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=PRIMARY_SEED)
    ap.add_argument("--stability-seeds", type=int, default=20,
                    help="另用种子 1..N 重跑 (a)(c) 检查蒙特卡洛稳定性；0 表示不跑")
    a = ap.parse_args()
    run = Path(a.run)
    rows, routing, seeds = load_rows(run, a.k)
    n = [len(col) for col in rows["C0"]]
    out_dir = Path(a.out)
    reference = json.loads((out_dir / "CANDIDATES-analysis.json").read_text(encoding="utf-8"))

    out_lines = []

    def say(s=""):
        print(s)
        out_lines.append(s)

    say("标签参照比较的补充分析——事后分析，非事先指定（added after observing results）")
    say(f"数据：{run}，每来源 {n[0]} 题，每题 {a.k} 次采样（种子 {seeds}）")
    say(f"联合配对 bootstrap：{a.draws} 次，来源内按题重采样，四候选共用索引，主种子 {a.seed}")
    say("C0 = 标签参照（方法拿不到）；C1 = 不用任何信息的默认分配")
    say("⚠ 区间只含 test 抽样（题目 + 解码），条件于每个候选的一次训练，不含训练随机性。")
    for c in CANDS:
        u = summarize([sum(rows[c][j]) / n[j] for j in range(K)])
        say(f"  {c}  路由 {routing[c]}  U_all {u[0]:.4f}  U_LS {u[1]:.4f}  U_other5 {u[2]:.4f}")

    # 自检：主分析种子
    per_main = joint_bootstrap(rows, a.draws, MAIN_SEED)
    worst = self_check(per_main, rows, reference)
    say(f"\n自检：种子 {MAIN_SEED} 复算 Cj − C0 的点估计与区间，与 CANDIDATES-analysis.json "
        f"最大偏差 {worst:.1e}（一致）")

    per = per_main if a.seed == MAIN_SEED else joint_bootstrap(rows, a.draws, a.seed)

    # (a)
    say("\n(a) 配对差（相对不用信息的默认 C1），pp，95% 联合配对 bootstrap 区间")
    pairs = {}
    for x, y in PAIRS:
        res = pair_block(per, rows, x, y)
        pairs[f"{x}-{y}"] = res
        say(f"  {x} − {y}")
        for name in ("dU_all", "dU_LS", "dU_other5"):
            lo, hi = res[name + "_ci95"]
            say(f"    {name:<10}{pp(res[name])}  [{pp(lo)}, {pp(hi)}]")
        say(f"    ΔU_all 区间整段落在 (−δ_min, +δ_min) = (−2, +2) pp 内："
            f"{'是' if res['dU_all_ci_within_pm_delta_min'] else '否'}")

    # (b)
    say("\n(b) 不依赖权重的流量敏感性：逐来源 Δ_j = Cj − C0（pp，精确值，均为 0.125 的倍数）")
    say("    任意流量比例 w 下 ΔU_w = Σ_j w_j Δ_j ∈ [min_j Δ_j, max_j Δ_j]（线性，顶点取极值）")
    say("    " + " " * 4 + "".join(f"{nm:>12}" for nm in SHORT) + "      可达范围")
    flow = {}
    for c in VS_REF:
        exact = exact_point(rows[c], rows["C0"])
        d = [float(x) for x in exact]
        lo, j_lo, hi, j_hi = reachable_range(d)
        flow[c] = {"per_source": dict(zip(NAMES, d)),
                   "per_source_exact": {nm: str(x) for nm, x in zip(NAMES, exact)},
                   "range": [lo, hi], "argmin": NAMES[j_lo], "argmax": NAMES[j_hi],
                   "equal_weight_dU_all": weighted_delta(d, [1 / K] * K)}
        say(f"    {c}" + "".join(f"{pp(v, 3):>12}" for v in d)
            + f"   [{pp(lo, 3)}, {pp(hi, 3)}]（最差 {NAMES[j_lo]}，最好 {NAMES[j_hi]}）")
    best = max(VS_REF, key=lambda c: flow[c]["range"][1])
    max_adv = flow[best]["range"][1]
    say(f"    任何流量比例下，相对 C0 的点估计优势最大为 {pp(max_adv, 3)} pp"
        f"（{best} 全部流量在 {flow[best]['argmax']}）"
        f"{'，< δ_min = 2 pp' if max_adv < DELTA_MIN else '，≥ δ_min'}")

    # (c)
    sim = simultaneous_block(per, rows)
    say(f"\n(c) 21 个逐线差 Cj − C0 的同时 95% 区间（max-|t| bootstrap）")
    say(f"    同时临界值 q = {sim['critical']:.3f}（逐个 95% 为 1.96）")
    for c in VS_REF:
        say(f"  {c} − C0")
        for nm, v in sim["lines"][c].items():
            mark = "  整段 < 0" if v["hi"] < 0 else ("  整段 > 0" if v["lo"] > 0 else "")
            say(f"      {nm:<13}{pp(v['d'], 3)}  [{pp(v['lo'])}, {pp(v['hi'])}]{mark}")
        mh = sim["max_upper"][c]
        say(f"    同时上界最大 {pp(mh['hi'])} pp（{mh['source']}）")
    say(f"    同时区间整段 < 0 的线：{sim['entirely_below_zero'] or '无'}")
    say(f"    同时区间整段 > 0 的线：{sim['entirely_above_zero'] or '无'}")

    # 蒙特卡洛稳定性
    stability = []
    if a.stability_seeds > 0:
        say(f"\n蒙特卡洛稳定性：同一数据、不同 bootstrap 种子（各 {a.draws} 次）")
        say(f"    {'种子':>9}{'q':>8}{'C2−C1 ΔU_all 区间':>22}{'C3−C1 ΔU_all 区间':>22}"
            f"{'最大同时上界':>10}   整段 < 0 的线")
        extra = [MAIN_SEED] + [s for s in range(1, a.stability_seeds + 1)]
        for s in [a.seed] + [s for s in extra if s != a.seed]:
            p = per if s == a.seed else (per_main if s == MAIN_SEED
                                         else joint_bootstrap(rows, a.draws, s))
            si = sim if s == a.seed else simultaneous_block(p, rows)
            pr = pairs if s == a.seed else {f"{x}-{y}": pair_block(p, rows, x, y)
                                            for x, y in PAIRS}
            mx = max(v["hi"] for v in si["max_upper"].values())
            rec = {"seed": s, "critical": si["critical"],
                   "C2-C1_ci95": pr["C2-C1"]["dU_all_ci95"],
                   "C3-C1_ci95": pr["C3-C1"]["dU_all_ci95"],
                   "max_upper": mx, "entirely_below_zero": si["entirely_below_zero"],
                   "upper": {f"{c} {nm}": v["hi"] for c in VS_REF
                             for nm, v in si["lines"][c].items()}}
            stability.append(rec)
            tag = "（主）" if s == a.seed else ("（主表）" if s == MAIN_SEED else "")
            c21, c31 = rec["C2-C1_ci95"], rec["C3-C1_ci95"]
            say(f"    {str(s) + tag:>9}{rec['critical']:>8.3f}"
                f"{'[' + pp(c21[0]) + ', ' + pp(c21[1]) + ']':>22}"
                f"{'[' + pp(c31[0]) + ', ' + pp(c31[1]) + ']':>22}"
                f"{pp(mx):>12}   {', '.join(rec['entirely_below_zero']) or '无'}")
        qs = [r["critical"] for r in stability]
        say(f"    临界值范围 [{min(qs):.3f}, {max(qs):.3f}]（{len(stability)} 个种子）")
        for line in sorted({x for r in stability for x in r["entirely_below_zero"]}):
            ups = [r["upper"][line] for r in stability]
            cnt = sum(u < 0 for u in ups)
            say(f"    {line}：同时上界 < 0 的种子 {cnt}/{len(stability)}，"
                f"上界范围 [{pp(min(ups))}, {pp(max(ups))}] pp"
                + ("" if cnt == len(stability) else "  ← 贴近 0，判定随种子改变，不宜作确定结论"))

    out = {"posthoc": True,
           "note": "事后分析，非事先指定（added after observing results）",
           "k": a.k, "eval_seeds": seeds, "n_per_source": n, "delta_min": DELTA_MIN,
           "bootstrap": {"draws": a.draws, "seed": a.seed, "main_analysis_seed": MAIN_SEED,
                         "scheme": "joint paired; resample items within source; shared indices"},
           "routing": routing,
           "point": {c: dict(zip(("U_all", "U_LS", "U_other5"),
                                 summarize([sum(rows[c][j]) / n[j] for j in range(K)])))
                     for c in CANDS},
           "self_check_max_abs_dev": worst,
           "pairs": pairs,
           "flow_sensitivity": flow,
           "max_point_advantage_any_weights": {"value": max_adv, "candidate": best,
                                               "source": flow[best]["argmax"]},
           "simultaneous": sim,
           "stability": stability}
    dst = out_dir / "LABELREF-analysis.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"\n写入 {dst} 与 {out_dir / 'LABELREF-analysis.txt'}")
    (out_dir / "LABELREF-analysis.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
