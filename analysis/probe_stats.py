"""state0 单来源迁移探针的统计（为本报告新写）。

输入是两次训练洗牌（s17 共 7 行、s23 共 6 行，缺 generic 行）的翻转计数：每格（训练来源 k
→ 打分来源 j）有 +g（错→对）与 −l（对→错）两个数，都是 audit 集每来源 80 题上的配对结果。
全部 p 值都从翻转计数用**精确整数/有理数算术**重算，不使用输入文件里四舍五入过的 pvalues 字段
（s23 的 pvalues 只保留三位小数）。

输出：
1. 逐格双侧精确 McNemar（不一致题上的二项检验，p = min(1, 2·Σ_{i≤min(g,l)} C(n,i)/2^n)）。
2. 每次洗牌各自的 49 格 Bonferroni：p_Bonf = min(1, 49p)。s23 只测了 42 格，仍按 49 格族
   校正（对 s23 偏保守）。列出名义显著（p < 0.05）与 Bonferroni 显著的格，以及两次洗牌都
   通过 Bonferroni 的边。
3. 完全零假设下名义显著格数的期望：上界 0.05 × 格数；以及精确检验在实测不一致数下的期望
   Σ_cells P_{g~Bin(n,1/2)}(p(g, n−g) < 0.05)（离散检验的拒绝概率低于 0.05）。
4. 按打分来源聚合：Statistics 列与 generic 列，供体为另外四个数学来源（Integral、
   Differential、Abstract、Linear），在两次洗牌上合计；给出逐供体、逐洗牌计数与"最强供体"
   （该洗牌内净增 g − l 最大者，并列取 p 小者）。合计计数上的 McNemar p 假设八次测量可交换，未经验证，只作描述。
5. 标签对计数：两个方向、两次洗牌合计（s23 缺 generic 行，algorithmic–generic 只含
   s17 两个方向与 s23 的 algorithmic→generic）。Linear–Statistics 对与三个标签树指定的对
   （Abstract–Linear、Integral–Differential、algorithmic–generic）逐一比较正向比例：
   2×2 表 [[g1, l1], [g2, l2]] 上的双侧 Fisher 精确检验（概率不超过观测表的所有表求和，
   比较在整数上做，无浮点容差）。

用法：python analysis/probe_stats.py [--s17 ...] [--s23 ...] [--out results/analysis]
"""
import argparse, json
from fractions import Fraction
from math import comb
from pathlib import Path

NAMES = ["Integral", "Differential", "Abstract", "Linear", "Statistics",
         "algorithmic", "generic"]
K = len(NAMES)
FAMILY = 49
ALPHA = Fraction(1, 20)
STAT, GEN = 4, 6
MATH_DONORS = (0, 1, 2, 3)
LS_PAIR = ("Linear-Statistics", (3, 4))
LABEL_PAIRS = (("Abstract-Linear", (2, 3)), ("Integral-Differential", (0, 1)),
               ("algorithmic-generic", (5, 6)))
# 附录统计表中的格（任一洗牌名义显著，外加边界格 Integral→Abstract）
TABLE_CELLS = ((3, 4), (4, 4), (3, 0), (4, 3), (2, 2), (0, 2), (1, 4), (3, 3), (4, 2))


# ---------------------------------------------------------------- 精确检验

def mcnemar(gained, lost):
    """双侧精确 McNemar：不一致题 n = g + l 上 Bin(n, 1/2) 的双侧二项检验，返回 Fraction。"""
    n = gained + lost
    if n == 0:
        return Fraction(1)
    k = min(gained, lost)
    return min(Fraction(1), Fraction(2 * sum(comb(n, i) for i in range(k + 1)), 2 ** n))


def fisher(a, b, c, d):
    """2×2 表 [[a, b], [c, d]] 的双侧 Fisher 精确检验，返回 Fraction。

    固定行和与列和，超几何分布下概率不超过观测表的所有表概率之和。各表概率共享分母
    C(n, a+c)，故直接比较整数分子，没有浮点容差问题。
    """
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    lo, hi = max(0, c1 - r2), min(r1, c1)
    observed = comb(r1, a) * comb(r2, c1 - a)
    tail = sum(w for w in (comb(r1, x) * comb(r2, c1 - x) for x in range(lo, hi + 1))
               if w <= observed)
    return Fraction(tail, comb(n, c1))


def rejection_probability(n, alpha=ALPHA):
    """零假设 g ~ Bin(n, 1/2) 下精确 McNemar 给出 p < alpha 的概率。"""
    if n == 0:
        return Fraction(0)
    return sum((Fraction(comb(n, g), 2 ** n) for g in range(n + 1)
                if mcnemar(g, n - g) < alpha), Fraction(0))


def bonferroni(p, family=FAMILY):
    return min(Fraction(1), family * p)


# ---------------------------------------------------------------- 数据

def load_flips(s17_path, s23_path):
    """返回 ({shuffle: {训练来源: [(g, l)] * 7}}, S0 准确率)。"""
    s17_rows = json.loads(Path(s17_path).read_text(encoding="utf-8"))
    s23 = json.loads(Path(s23_path).read_text(encoding="utf-8"))
    flips = {}
    for name, rows in (("s17", s17_rows), ("s23", s23["rows"])):
        flips[name] = {r["source"]: [(f["gained"], f["lost"]) for f in r["flips"]]
                       for r in rows}
    before = s17_rows[0]["before"]
    for rows in (s17_rows, s23["rows"]):
        for r in rows:
            if r["before"] != before:
                raise SystemExit("两次洗牌的 S0 准确率不一致")
    return flips, before


def pool(flips, cells):
    """在两次洗牌上合计若干格的 (g, l)；缺行的洗牌跳过该格。返回 (g, l, 用到的测量列表)。"""
    g = l = 0
    used = []
    for shuffle, rows in flips.items():
        for k, j in cells:
            if k in rows:
                g += rows[k][j][0]
                l += rows[k][j][1]
                used.append(f"{shuffle} {NAMES[k]}->{NAMES[j]}")
    return g, l, used


def label(k, j):
    return f"{NAMES[k]}->{NAMES[j]}"


def pfloat(p):
    return float(p)


def pexact(p):
    return f"{p.numerator}/{p.denominator}"


# ---------------------------------------------------------------- 分析

def shuffle_block(rows):
    cells, nominal, bonf = [], [], []
    expected = Fraction(0)
    for k in sorted(rows):
        for j in range(K):
            g, l = rows[k][j]
            p = mcnemar(g, l)
            pb = bonferroni(p)
            cells.append({"train": NAMES[k], "scored": NAMES[j], "gained": g, "lost": l,
                          "p": pfloat(p), "p_exact": pexact(p), "p_bonf": pfloat(pb),
                          "nominal": p < ALPHA, "bonferroni": pb < ALPHA})
            if p < ALPHA:
                nominal.append(label(k, j))
            if pb < ALPHA:
                bonf.append(label(k, j))
            expected += rejection_probability(g + l)
    measured = len(rows) * K
    return {"rows": [NAMES[k] for k in sorted(rows)], "cells_measured": measured,
            "cells": cells, "nominal_cells": nominal, "n_nominal": len(nominal),
            "bonferroni_cells": bonf, "n_bonferroni": len(bonf),
            "expected_nominal_under_null": {
                "upper_bound_alpha_times_cells": pfloat(ALPHA * measured),
                "upper_bound_alpha_times_family": pfloat(ALPHA * FAMILY),
                "exact_test_at_observed_discordant_counts": pfloat(expected)}}


def column_block(flips, target, before):
    per_donor = {}
    for k in MATH_DONORS:
        entry = {}
        for shuffle, rows in flips.items():
            if k in rows:
                g, l = rows[k][target]
                entry[shuffle] = [g, l]
        g, l, _ = pool(flips, [(k, target)])
        entry["pooled"] = [g, l]
        entry["gains_exceed_losses_in_every_shuffle"] = all(
            v[0] > v[1] for s, v in entry.items() if s in flips)
        per_donor[NAMES[k]] = entry
    g, l, used = pool(flips, [(k, target) for k in MATH_DONORS])
    p = mcnemar(g, l)
    strongest = {}
    for shuffle, rows in flips.items():
        k_best = min(MATH_DONORS, key=lambda k: (rows[k][target][1] - rows[k][target][0],
                                                 mcnemar(*rows[k][target]), k))
        gb, lb = rows[k_best][target]
        strongest[shuffle] = {"donor": NAMES[k_best], "gained": gb, "lost": lb,
                              "p": pfloat(mcnemar(gb, lb))}
    return {"target": NAMES[target], "s0_accuracy": before[target],
            "donors": [NAMES[k] for k in MATH_DONORS], "measurements": used,
            "pooled": [g, l], "p_nominal_if_exchangeable": pfloat(p), "p_exact": pexact(p),
            "per_donor": per_donor, "strongest_donor_by_net_gain": strongest}


def pair_block(flips, a, b):
    g, l, used = pool(flips, [(a, b), (b, a)])
    return {"sources": [NAMES[a], NAMES[b]], "measurements": used, "gained": g, "lost": l,
            "positive_fraction": g / (g + l)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s17", default="results/probes/s17_transfer.json")
    ap.add_argument("--s23", default="results/probes/s23_transfer_partial.json")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    flips, before = load_flips(a.s17, a.s23)

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("state0 单来源迁移探针统计（精确整数/有理数算术，p 值从翻转计数重算）")
    say(f"输入：s17 = {a.s17}（{len(flips['s17'])} 行）；s23 = {a.s23}（{len(flips['s23'])} 行）")
    say("S0 准确率（audit 每来源 80 题）：" + "  ".join(f"{n} {v:.4f}" for n, v in zip(NAMES, before)))

    shuffles = {s: shuffle_block(rows) for s, rows in flips.items()}
    say(f"\n1–3. 逐格双侧精确 McNemar；Bonferroni 按 {FAMILY} 格族（p_Bonf = min(1, {FAMILY}p)）")
    for s, blk in shuffles.items():
        e = blk["expected_nominal_under_null"]
        say(f"  {s}：测了 {blk['cells_measured']} 格；名义 p<0.05 共 {blk['n_nominal']} 格："
            f"{', '.join(blk['nominal_cells'])}")
        say(f"      完全零假设下名义显著格数的期望：上界 {e['upper_bound_alpha_times_cells']:.2f}"
            f"（0.05×{blk['cells_measured']}）；精确检验在实测不一致数下 "
            f"{e['exact_test_at_observed_discordant_counts']:.3f}")
        bc = [c for c in blk["cells"] if c["bonferroni"]]
        say(f"      Bonferroni 通过：" + ("; ".join(
            f"{c['train']}->{c['scored']} +{c['gained']}/-{c['lost']} p={c['p']:.3g} "
            f"p_Bonf={c['p_bonf']:.5f}" for c in bc) or "无"))
    both = sorted(set(shuffles["s17"]["bonferroni_cells"]) & set(shuffles["s23"]["bonferroni_cells"]))
    say(f"  两次洗牌都通过 Bonferroni 的边：{', '.join(both) or '无'}")

    say("\n  附录表中的格（任一洗牌名义显著，外加边界格 Integral->Abstract）")
    say(f"    {'格':<28}{'s17 +g/-l':>11}{'p':>10}{'s23 +g/-l':>11}{'p':>11}{'p_Bonf(s23)':>13}")
    table = []
    for k, j in TABLE_CELLS:
        row = {"cell": label(k, j)}
        for s in ("s17", "s23"):
            g, l = flips[s][k][j]
            p = mcnemar(g, l)
            row[s] = {"gained": g, "lost": l, "p": pfloat(p), "p_bonf": pfloat(bonferroni(p))}
        table.append(row)
        r17, r23 = row["s17"], row["s23"]
        say(f"    {row['cell']:<28}{'+%d/-%d' % (r17['gained'], r17['lost']):>11}{r17['p']:>10.4g}"
            f"{'+%d/-%d' % (r23['gained'], r23['lost']):>11}{r23['p']:>11.3g}{r23['p_bonf']:>13.5f}")

    stat = column_block(flips, STAT, before)
    gen = column_block(flips, GEN, before)
    say("\n4. 按打分来源聚合（供体 = 另外四个数学来源，两次洗牌合计；可交换性未验证，只作描述）")
    for blk in (stat, gen):
        g, l = blk["pooled"]
        say(f"  {blk['target']} 列（S0 准确率 {blk['s0_accuracy']:.4f}）：合计 +{g}/-{l}，"
            f"名义 p = {blk['p_nominal_if_exchangeable']:.3g}（{len(blk['measurements'])} 次测量）")
        for nm, v in blk["per_donor"].items():
            parts = "  ".join(f"{s} +{v[s][0]}/-{v[s][1]}" for s in flips if s in v)
            say(f"      {nm:<13}合计 +{v['pooled'][0]}/-{v['pooled'][1]}   {parts}"
                f"   每次洗牌增多于减：{'是' if v['gains_exceed_losses_in_every_shuffle'] else '否'}")
        say("      最强供体（洗牌内 g−l 最大）：" + "；".join(
            f"{s} {v['donor']} +{v['gained']}/-{v['lost']} p={v['p']:.3g}"
            for s, v in blk["strongest_donor_by_net_gain"].items()))

    pairs = {name: pair_block(flips, *cells) for name, cells in (LS_PAIR,) + LABEL_PAIRS}
    ls = pairs[LS_PAIR[0]]
    say("\n5. 标签对计数（两个方向、两次洗牌合计）与 Linear-Statistics 对的 Fisher 精确比较")
    for name, v in pairs.items():
        say(f"  {name:<24}+{v['gained']}/-{v['lost']}  正向比例 {v['positive_fraction']:.3f}"
            f"  （{len(v['measurements'])} 次测量）")
    fisher_out = {}
    for name, _ in LABEL_PAIRS:
        o = pairs[name]
        p = fisher(ls["gained"], ls["lost"], o["gained"], o["lost"])
        fisher_out[name] = {"table": [[ls["gained"], ls["lost"]], [o["gained"], o["lost"]]],
                            "p": pfloat(p), "p_exact": pexact(p)}
        say(f"  Linear-Statistics vs {name:<24}双侧 Fisher p = {pfloat(p):.6f}")

    out = {"note": ("Exact two-sided McNemar per cell from flip counts; Bonferroni over the "
                    f"{FAMILY}-cell family per shuffle; pooled counts are descriptive "
                    "(exchangeability of measurements not verified); exact two-sided Fisher "
                    "tests in integer arithmetic. The Linear-Statistics pair was selected after "
                    "viewing the tensor."),
           "inputs": {"s17": a.s17, "s23": a.s23},
           "sources": NAMES, "s0_accuracy": before, "family_size": FAMILY, "alpha": 0.05,
           "shuffles": shuffles, "bonferroni_in_both_shuffles": both,
           "table_cells": table,
           "statistics_column": stat, "generic_column": gen,
           "pairs": pairs, "fisher_vs_linear_statistics": fisher_out}
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    say(f"\n写入 {dst / 'PROBE-stats.json'} 与 {dst / 'PROBE-stats.txt'}")
    (dst / "PROBE-stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (dst / "PROBE-stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
