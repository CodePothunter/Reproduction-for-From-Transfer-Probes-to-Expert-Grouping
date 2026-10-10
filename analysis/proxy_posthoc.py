"""加性代理的事后诊断——事后分析，非事先指定。

⚠ 本诊断是在看到候选实际收益表（`results/analysis/CANDIDATES-analysis.txt`）**之后**
才加的，只作诊断，不作结论；论文中标为 "added after observing results"（附录 app:proxy）。

问题：开发期的加性代理对 C1/C2/C3 − C0 的预测，与 W 上实测的差有多大出入；"迁移在 W 处
消失"这一解释在代理里能否解释"比 C0 差"。

代理（与 `analysis/allocation_sweep.py::utility` 同一个公式，直接复用其常数与实现）：

    c_i(j) = clip_[0,1]( base_j + Σ_k T[k][j] · min(n_ik, 16) / 16 )

T 为 state0 单来源探针张量（行 = 训练来源 k，列 = audit 来源 j，值为 80 题上的准确率变化），
base 为 state0 的 audit 准确率，n_ik 为专家 i 在本区间获得来源 k 的块数。U = 7 个来源上
被路由专家能力的等权均值；U_LS 为 Linear、Statistics 两列，U_other5 为其余五列。

张量：
- s17：`results/probes/s17_transfer.json`（7/7 行）。
- s23*：`results/probes/s23_transfer_partial.json`（6/7 行），缺的 generic 行借用 s17。

路由三种口径：
1. 块矩阵路由：每来源由获得其块数最多的专家服务，并列取编号小者（与
   pipeline/run_candidates.sh 及实测评价的路由规则相同；本脚本逐一核对实测评价文件里的路由）。
2. 能力（argmax）路由：每来源由代理能力最高的专家服务（= allocation_sweep.utility）。
3. 零迁移：T 只保留对角项（训练 k 只影响 k），块矩阵路由。

另报告 C3 的 ΔU_LS 预测，并与实测 ΔU_LS 对照。

⚠ 解读边界：
- 三重错配：状态（探针在 state0 = 基座 + 新 LoRA 上测，候选从 W 出发训练）；题库（探针在
  audit 集每来源 80 题上测，实测在 test 集每来源前 400 题上测）；剂量（探针每来源 16 块，
  候选里每专家每来源 0–8 块，代理按 min(n,16)/16 线性外推）。
- C2 是同一代理（s17 张量 + argmax 路由）的运输 LP 最优点（本脚本核对 C2 = Q*·56），
  所以它在 s17 代理下的预测带有选择偏高（optimizer's curse），不是独立预测。
- 只有 3 个候选点；只能说"迁移在 W 处消失"在该代理里不足以解释"比 C0 差"，**不能**据此
  证明真实差距来自联合训练干扰，也不能把这 3 个点概括成代理与实测之间的一般规律。
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import allocation_sweep as sweep  # noqa: E402  utility 公式、块总数、饱和点、LP 最优

NAMES = sweep.SOURCES
K = sweep.K
M = 4
LS = (3, 4)
ROW_BLOCKS, COL_BLOCKS = 14, 8
# 块矩阵（行 = 专家，列 = Int, Diff, Abs, Lin, Stat, alg, gen），与 run_candidates.sh 相同。
CANDIDATES = {
    "C0": [[0, 0, 7, 7, 0, 0, 0], [7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7],
           [1, 1, 1, 1, 8, 1, 1]],
    "C1": [[2] * K for _ in range(M)],
    "C2": [[0, 0, 0, 6, 8, 0, 0], [0, 8, 2, 2, 0, 0, 2], [0, 0, 0, 0, 0, 8, 6],
           [8, 0, 6, 0, 0, 0, 0]],
    "C3": [[0, 0, 0, 7, 7, 0, 0], [7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7],
           [1, 1, 8, 1, 1, 1, 1]],
}
OTHERS = ("C1", "C2", "C3")


# ---------------------------------------------------------------- 纯函数

def check_blocks(blocks):
    """整数块矩阵须满足双边际：每专家 14 块，每来源 8 块。"""
    if any(sum(row) != ROW_BLOCKS for row in blocks):
        raise ValueError("行边际不是每专家 14 块")
    if any(sum(row[k] for row in blocks) != COL_BLOCKS for k in range(K)):
        raise ValueError("列边际不是每来源 8 块")


def block_routing(blocks):
    """每来源由获得其块数最多的专家服务，并列取编号小者。"""
    return [max(range(len(blocks)), key=lambda i: (blocks[i][k], -i))
            for k in range(len(blocks[0]))]


def capability(row, tensor, base, j):
    """专家（块向量 row）在来源 j 上的代理能力；公式与 allocation_sweep.utility 相同。"""
    value = base[j]
    for k in range(len(row)):
        value += tensor[k][j] * min(row[k], sweep.SATURATION) / sweep.SATURATION
    return min(1.0, max(0.0, value))


def argmax_routing(blocks, tensor, base):
    """每来源由代理能力最高的专家服务，并列取编号小者。"""
    return [max(range(len(blocks)),
                key=lambda i: (capability(blocks[i], tensor, base, j), -i))
            for j in range(len(base))]


def routed_utility(blocks, tensor, base, routing, cols=None):
    """固定路由下的代理效用：cols 上被路由专家能力的等权均值。"""
    cols = list(range(len(base))) if cols is None else list(cols)
    return sum(capability(blocks[routing[j]], tensor, base, j) for j in cols) / len(cols)


def zero_transfer(tensor):
    """只保留对角项：训练来源 k 只影响 audit 来源 k。"""
    return [[v if k == j else 0.0 for j, v in enumerate(row)] for k, row in enumerate(tensor)]


def summarize(blocks, tensor, base, routing):
    other = [j for j in range(len(base)) if j not in LS]
    return {"U_all": routed_utility(blocks, tensor, base, routing),
            "U_LS": routed_utility(blocks, tensor, base, routing, LS),
            "U_other5": routed_utility(blocks, tensor, base, routing, other)}


def borrow_rows(partial_rows, fallback):
    """部分张量补齐：缺的训练来源行借用 fallback。返回 (张量, 借用的来源编号)。"""
    have = {r["source"]: r["delta"] for r in partial_rows}
    tensor, borrowed = [], []
    for k in range(len(fallback)):
        if k in have:
            tensor.append(list(have[k]))
        else:
            tensor.append(list(fallback[k]))
            borrowed.append(k)
    return tensor, borrowed


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s17", default="results/probes/s17_transfer.json")
    ap.add_argument("--s23", default="results/probes/s23_transfer_partial.json")
    ap.add_argument("--run", default="results/candidates")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    run = Path(a.run)

    t17, base, rows = sweep.load_tensor(a.s17)
    assert t17 is not None and rows == K, f"{a.s17} 不是完整的 7 行张量"
    partial = json.loads(Path(a.s23).read_text(encoding="utf-8"))
    for r in partial["rows"]:
        assert r["before"] == base, "s23 的 state0 基线与 s17 不同，不能拼接"
    t23, borrowed = borrow_rows(partial["rows"], t17)
    tensors = {"s17": t17, "s23*": t23}

    for c, b in CANDIDATES.items():
        check_blocks(b)

    out_lines = []

    def say(s=""):
        print(s)
        out_lines.append(s)

    say("加性代理的事后诊断——事后分析，非事先指定（added after observing results）")
    say("代理 c_i(j) = clip(base_j + Σ_k T[k][j]·min(n_ik,16)/16)，与 allocation_sweep.utility 同式")
    say(f"张量 s17 = {a.s17}；s23* = {a.s23}（{len(partial['rows'])}/7 行，"
        f"借用 s17 的行：{[NAMES[k] for k in borrowed] or '无'}）")
    say("⚠ 状态（state0 vs W）、题库（audit 80 题 vs test 400 题）、剂量（16 块 vs 0–8 块）三重错配；")
    say("  C2 是 s17 代理的 LP 最优点（optimizer's curse）；只有 3 个候选点。只作诊断。")

    # 自检 1：块矩阵路由与实测评价文件一致
    routes = {c: block_routing(b) for c, b in CANDIDATES.items()}
    route_check = {}
    for c in CANDIDATES:
        p = run / f"{c}-eval-s2001.json"
        if p.exists():
            measured = json.loads(p.read_text(encoding="utf-8"))["routing"]
            assert measured == routes[c], f"{c} 块矩阵路由 {routes[c]} ≠ 实测评价路由 {measured}"
            route_check[c] = True
    say(f"\n自检：块矩阵路由与实测评价文件一致 {sorted(route_check)}")
    for c in CANDIDATES:
        say(f"  {c}  块矩阵路由 {routes[c]}")

    # 自检 2：argmax 路由下与 allocation_sweep.utility 相同
    worst = 0.0
    for t in list(tensors.values()) + [zero_transfer(t) for t in tensors.values()]:
        for c, b in CANDIDATES.items():
            q = [[x / sweep.BLOCKS_TOTAL for x in row] for row in b]
            mine = routed_utility(b, t, base, argmax_routing(b, t, base))
            worst = max(worst, abs(mine - sweep.utility(q, t, base)))
    assert worst < 1e-12, f"与 allocation_sweep.utility 不一致：{worst}"
    say(f"自检：argmax 路由效用与 allocation_sweep.utility 最大偏差 {worst:.1e}")

    # 自检 3：C2 = s17 代理的运输 LP 最优点 × 56
    u_opt, q_opt, lp_routing = sweep.best_allocation(M, t17, base)
    lp_blocks = [[round(x * sweep.BLOCKS_TOTAL, 9) for x in row] for row in q_opt]
    c2_is_lp = lp_blocks == [[float(x) for x in row] for row in CANDIDATES["C2"]]
    say(f"自检：C2 {'等于' if c2_is_lp else '不等于'} s17 代理（argmax 路由）的运输 LP 最优点 × 56"
        f"（U* = {u_opt:.4f}，LP 枚举路由 {list(lp_routing)}）")

    out_dir = Path(a.out)
    measured = json.loads((out_dir / "CANDIDATES-analysis.json").read_text(encoding="utf-8"))
    meas = {c: {k: measured["contrasts"][f"{c}-C0"][k] for k in ("dU_all", "dU_LS", "dU_other5")}
            for c in OTHERS}

    modes = {
        "block": ("块矩阵路由", lambda t: t, lambda b, t: block_routing(b)),
        "argmax": ("能力（argmax）路由", lambda t: t, lambda b, t: argmax_routing(b, t, base)),
        "zero_transfer": ("零迁移（只留对角，块矩阵路由）", zero_transfer, lambda b, t: block_routing(b)),
    }

    results = {}
    say("\n代理预测的 Cj − C0（pp）；实测见末行")
    say(f"  {'口径':<30}{'张量':<7}" + "".join(f"{c + '−C0':>9}" for c in OTHERS)
        + f"{'C3 ΔU_LS':>10}")
    for mode, (label, transform, router) in modes.items():
        results[mode] = {}
        for tn, t in tensors.items():
            tt = transform(t)
            u = {c: summarize(b, tt, base, router(b, tt)) for c, b in CANDIDATES.items()}
            rt = {c: router(b, tt) for c, b in CANDIDATES.items()}
            d = {c: {k: u[c][k] - u["C0"][k] for k in ("U_all", "U_LS", "U_other5")}
                 for c in OTHERS}
            results[mode][tn] = {"U": u, "routing": rt, "delta_vs_C0": d}
            say(f"  {label:<30}{tn:<7}" + "".join(f"{100 * d[c]['U_all']:>+9.2f}" for c in OTHERS)
                + f"{100 * d['C3']['U_LS']:>+10.2f}")
    say(f"  {'实测（W，test 400 题，k=2）':<30}{'':<7}"
        + "".join(f"{100 * meas[c]['dU_all']:>+9.2f}" for c in OTHERS)
        + f"{100 * meas['C3']['dU_LS']:>+10.2f}")

    c0_switch = {tn: results["argmax"][tn]["U"]["C0"]["U_all"]
                 - results["block"][tn]["U"]["C0"]["U_all"] for tn in tensors}
    say("\nC0 自身由块矩阵路由换成能力路由的代理增益（pp）："
        + "，".join(f"{tn} {100 * v:+.2f}" for tn, v in c0_switch.items()))
    for tn in tensors:
        say(f"  {tn} 能力路由：" + "  ".join(f"{c} {results['argmax'][tn]['routing'][c]}"
                                          for c in CANDIDATES))

    say("\n逐候选 U_LS / U_other5 差（pp，代理 vs 实测）")
    for c in OTHERS:
        parts = [f"{mode}/{tn} {100 * results[mode][tn]['delta_vs_C0'][c]['U_LS']:+.2f}"
                 f"/{100 * results[mode][tn]['delta_vs_C0'][c]['U_other5']:+.2f}"
                 for mode in modes for tn in tensors]
        say(f"  {c}: " + "；".join(parts)
            + f"；实测 {100 * meas[c]['dU_LS']:+.2f}/{100 * meas[c]['dU_other5']:+.2f}")

    say("\n解读（只作诊断）：零迁移代理下 C2、C3 − C0 接近 0，而实测为负；故"
        "\"迁移在 W 处消失\"这一解释在该代理里不足以解释\"比 C0 差\"。")
    say("这不能证明真实差距来自联合训练干扰：状态、题库、剂量三重错配；C2 是同一代理的 LP 最优点；"
        "只有 3 个点。")

    out = {"posthoc": True,
           "note": "事后分析，非事先指定（added after observing results）；只作诊断",
           "proxy": "c_i(j)=clip(base_j+sum_k T[k][j]*min(n_ik,16)/16), allocation_sweep.utility",
           "tensors": {"s17": a.s17, "s23*": a.s23,
                       "s23*_borrowed_rows_from_s17": [NAMES[k] for k in borrowed]},
           "base": base, "blocks": CANDIDATES, "block_routing": routes,
           "self_checks": {"block_routing_matches_eval": route_check,
                           "argmax_vs_allocation_sweep_max_abs_dev": worst,
                           "C2_equals_s17_LP_optimum": c2_is_lp,
                           "s17_LP_optimum_U": u_opt, "s17_LP_routing": list(lp_routing)},
           "results": results,
           "C0_argmax_minus_block": c0_switch,
           "measured": meas,
           "caveats": ["state mismatch: state0 vs W", "bank mismatch: audit 80 vs test 400",
                       "dose mismatch: 16 blocks vs 0-8 blocks, linear saturation",
                       "C2 is the LP optimum of the same proxy (optimizer's curse)",
                       "only 3 candidate points"]}
    dst = out_dir / "PROXY-posthoc.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"\n写入 {dst} 与 {out_dir / 'PROXY-posthoc.txt'}")
    (out_dir / "PROXY-posthoc.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
