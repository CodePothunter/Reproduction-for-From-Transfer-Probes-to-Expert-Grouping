"""标签参照 C0 的构造核对（为本报告新写；CPU，几秒）。

只用标签树、预算边际与包内的 lwts.allocation / analysis/allocation_sweep.py，核对论文中关于
C0 的四件事：

1. 标签树在每个专家数 M = 2..7 下的覆盖切分（互不包含、并集覆盖全部叶子的 M 个树节点）
   个数；M = 4 时只有一个切分 {calculus, algebra, Statistics, code}，因此 C0 的分组不受任何
   开发期响应数据影响。
2. C0 的块矩阵 = 硬标签映射（来源 → 专家，其余格取 1e-6 的正下限）经 Sinkhorn 配平
   （lwts.allocation.mirror，η = 0，即只做边际配平）到行边际 1/4、列边际 1/7，再按整数预算
   （每专家 14 块、每来源 8 块）用 lwts.allocation.integer_counts 取整。
3. C0/56 是树距离运输问题 min Σ Q[i][k]·d(k, 组 i) 的并列最优解之一：其代价等于
   _transport_lp 求得的最优代价，而 LP 顶点本身是另一个块矩阵。d 为标签树上叶到组的边数。
4. 开发期加性代理（s17 张量，代理内按能力路由，allocation_sweep.utility）下，树距离 LP
   顶点与 C0 的得分。

用法：python analysis/c0_construction.py [--s17 results/probes/s17_transfer.json] [--out results/analysis]
"""
import argparse, json, sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import allocation_sweep as sweep  # noqa: E402

from lwts.allocation import _transport_lp, check_marginals, integer_counts, mirror  # noqa: E402

M = 4
K = sweep.K
ROW_BLOCKS, COL_BLOCKS = 14, 8
C0 = [[0, 0, 7, 7, 0, 0, 0], [7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7],
      [1, 1, 1, 1, 8, 1, 1]]
# C0 的专家编号：e0 = algebra，e1 = calculus，e2 = code，e3 = Statistics
C0_GROUPS = ("algebra", "calculus", "code", "leaf4")
FLOOR = 1e-6


def hard_map(groups):
    """来源 → 专家：来源所在的组的编号。"""
    return [next(i for i, g in enumerate(groups) if k in sweep.TREE[g]) for k in range(K)]


def tree_cost(groups):
    return [[min(sweep.DIST[k][m] for m in sweep.TREE[groups[i]]) for k in range(K)]
            for i in range(len(groups))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s17", default="results/probes/s17_transfer.json")
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # 1. 覆盖切分
    cuts = {}
    say("1. 标签树的覆盖切分")
    for m in range(2, K + 1):
        groups = sweep.covering_antichains(m)
        cuts[m] = [sweep.describe(g, m) for g in groups]
        say(f"  M={m}：{len(groups)} 个  " + "；".join(cuts[m]))
    four = sweep.covering_antichains(M)
    assert len(four) == 1 and set(four[0]) == set(C0_GROUPS), four

    # 2. Sinkhorn 配平 + 整数取整
    route = hard_map(C0_GROUPS)
    base = [[FLOOR] * K for _ in range(M)]
    for k, i in enumerate(route):
        base[i][k] = 1.0
    r, p = [1.0 / M] * M, [1.0 / K] * K
    q = mirror([[0.0] * K for _ in range(M)], base, r, p, eta=0.0)
    check_marginals(q, r, p, 1e-7)
    blocks = integer_counts(q, [ROW_BLOCKS] * M, [COL_BLOCKS] * K)
    derived_equals_c0 = blocks == C0
    say("\n2. 硬标签映射 " + str(route) + " 经 Sinkhorn 配平（η=0）与整数取整")
    say("  连续目标 ×56：" + str([[round(x * 56, 3) for x in row] for row in q]))
    say(f"  整数块矩阵：{blocks}  与 C0 {'相同' if derived_equals_c0 else '不同'}")

    # 3. 树距离运输问题
    cost = tree_cost(C0_GROUPS)
    q_lp = _transport_lp(cost, r, p)
    lp_blocks = [[round(x * 56, 9) for x in row] for row in q_lp]
    lp_cost = Fraction(round(sum(cost[i][k] * q_lp[i][k] for i in range(M) for k in range(K)) * 56, 9))
    c0_cost = sum(cost[i][k] * C0[i][k] for i in range(M) for k in range(K))
    tied = lp_cost == c0_cost
    say("\n3. 树距离运输问题（代价 × 56 块）")
    say(f"  LP 最优代价 {lp_cost}；C0 的代价 {c0_cost}；{'并列最优' if tied else '不并列'}")
    say(f"  LP 顶点 ×56：{lp_blocks}  {'≠' if lp_blocks != [[float(x) for x in row] for row in C0] else '='} C0")

    # 4. 开发期代理得分
    tensor, base_acc, rows = sweep.load_tensor(a.s17)
    assert rows == K
    u_lp = sweep.utility(q_lp, tensor, base_acc)
    u_c0 = sweep.utility([[x / 56 for x in row] for row in C0], tensor, base_acc)
    u_sem = sweep.semantic_allocation(M, tensor, base_acc)[0]
    say("\n4. 开发期加性代理（s17，代理内按能力路由）")
    say(f"  树距离 LP 顶点 {u_lp:.4f}（= allocation_sweep 的语义基线 {u_sem:.4f}）；C0 {u_c0:.4f}")

    out = {"note": ("Label-tree cuts per number of experts, derivation of C0 from the label map and "
                    "the budget margins, and its tie in the tree-distance transportation problem."),
           "inputs": {"s17": a.s17},
           "covering_cuts": {str(m): {"count": len(v), "cuts": v} for m, v in cuts.items()},
           "covering_cut_counts_M2_to_M7": [len(cuts[m]) for m in range(2, K + 1)],
           "C0_blocks": C0,
           "C0_groups": [sweep.describe((g,), 1) for g in C0_GROUPS],
           "hard_label_map": route,
           "sinkhorn_floor": FLOOR,
           "sinkhorn_target_times_56": [[x * 56 for x in row] for row in q],
           "derived_blocks": blocks,
           "derived_equals_C0": derived_equals_c0,
           "tree_distance_cost_times_56": {"lp_optimum": str(lp_cost), "C0": c0_cost,
                                           "C0_is_tied_optimum": tied},
           "tree_distance_lp_vertex_times_56": lp_blocks,
           "proxy_s17_competence_routing": {"tree_distance_lp_vertex": u_lp, "C0": u_c0,
                                            "allocation_sweep_semantic_M4": u_sem}}
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    say(f"\n写入 {dst / 'C0-construction.json'} 与 {dst / 'C0-construction.txt'}")
    (dst / "C0-construction.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    (dst / "C0-construction.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
