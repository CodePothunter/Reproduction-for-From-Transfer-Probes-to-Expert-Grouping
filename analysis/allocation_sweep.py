"""开发期扫描：在已测响应张量上，对各专家数 M 精确算均匀、最佳语义（树切分）与最优分配的效用。

对每个 M 报告 U(均匀)、U(Q_sem,M)、U(Q*_M) 与 Δ_M = U(Q*_M) − U(Q_sem,M)，并标出 Q*_M 的
路由结构是否与最佳语义划分不同（相同时，该 M 下语义划分已是代理最优）；再用自助重采样
给出 Δ_M > 0 的比例。

本报告的 M=4 不由此脚本选出：选择 M 不在报告范围内，而 M=4 时标签树只有一种四组切分
（见报告 Setting 一节）。报告只引用 M=4 一行，作为开发期代理下的对照。

---

为什么能精确求最优
------------------
目标是**凸**的，不能靠局部爬山求最大值：

    U(Q) = (1/K) Σ_j max_i c_i(j)       c_i(j) = base_j + Σ_k A[k][j]·b(Q[i][k])/16

max 是线性函数的上包络 ⇒ U 在 Q 上是**凸**的 ⇒ 在多胞形上求**最大**值，最优点必在
顶点，而爬山会卡在任意一个局部顶点。

但凸性同时给了出路。引入路由 ρ: 来源 j → 专家 ρ(j)，记 U_ρ(Q) = (1/K) Σ_j c_ρ(j)(j)，
则 U(Q) = max_ρ U_ρ(Q)，于是

    max_{Q∈U(r,π)} U(Q) = max_ρ  max_{Q∈U(r,π)} U_ρ(Q)

内层在固定 ρ 下是**线性**的，就是运输问题，`_transport_lp` 精确求解（单纯形 + Bland
规则，并列时有确定选择）。外层的 ρ 在专家可交换下只需枚举**受限增长串**（集合划分），
M=5 时只有 855 个。**所以这是穷举，不是启发式。**

`U(r,π)` 用连续边际：r_i = 1/M、π_k = 1/K。整数块配额是下游的舍入（`integer_counts`），
在开发期扫 M 时强行取整会让 M ∤ H 的配置无法比较，反而引入人为选择。

三个口径固定为与既有实验一致：总预算 H = 56 块（每来源 8 块，与 state0 探针
（results/probes/）同尺度）、饱和点 16 块、能力裁剪到 [0,1]。列边际锁死 ⇒ **每个来源被训练的块数在
所有分配下完全相同**，U 的差异只来自「同样的块落在哪个专家里」，不可能靠给某个来源
多训练来刷高——这是双边际约束在科学上的作用，不只是工程约束。

语义基线的定义
------------
标签树（`pipeline/build_manifest_7src.py` 的设计）：

    root ├ math ├ calculus ├ 0 Integral      ┘ 标签说同属 Calculus
         │      │          └ 1 Differential
         │      ├ algebra  ├ 2 Abstract      ┘ 标签说同属 Algebra
         │      │          └ 3 Linear
         │      └ 4 Statistics
         └ code ├ 5 algorithmic
                └ 6 generic

给定 M，枚举**所有**覆盖全部叶子的大小为 M 的反链（= 按树切出的 M 个组），每个组合都
求其最忠实的可行分配，再取 U 最大者。**故意给语义基线最强的一把**：任何剩下的 gap
都不能用「你选了个弱的语义划分」来解释。

「最忠实的可行分配」= argmin_{Q∈U(r,π)} Σ_{i,k} Q[i][k]·d(k, 组 i)，d 是标签树上的
叶到组距离（组内为 0）。这是无参数的：不用 Sinkhorn 的温度，也不用先硬分组再投影
（硬分组在组大小不等时根本不可行）。
"""
import argparse, json, math, random
from itertools import product
from pathlib import Path

from lwts.allocation import _transport_lp, check_marginals

SOURCES = ["Integral", "Differential", "Abstract", "Linear", "Statistics",
           "algorithmic", "generic"]
K = len(SOURCES)
BLOCKS_TOTAL = 56          # 与 state0 探针同尺度：每来源 8 块
SATURATION = 16            # 饱和点：探针剂量为 16 块，代理不外推到更大剂量

# 标签树：内部节点 → 其叶子集合。
TREE = {
    "root": frozenset(range(7)),
    "math": frozenset({0, 1, 2, 3, 4}),
    "calculus": frozenset({0, 1}),
    "algebra": frozenset({2, 3}),
    "code": frozenset({5, 6}),
    **{f"leaf{i}": frozenset({i}) for i in range(7)},
}
CHILDREN = {"root": ["math", "code"],
            "math": ["calculus", "algebra", "leaf4"],
            "calculus": ["leaf0", "leaf1"],
            "algebra": ["leaf2", "leaf3"],
            "code": ["leaf5", "leaf6"],
            **{f"leaf{i}": [] for i in range(7)}}


def leaf_distance():
    """叶到叶的树上边数；用来定义「标签有多近」。"""
    depth, parent = {}, {}

    def walk(node, d, par):
        depth[node], parent[node] = d, par
        for child in CHILDREN[node]:
            walk(child, d + 1, node)

    walk("root", 0, None)

    def ancestors(node):
        chain = []
        while node is not None:
            chain.append(node)
            node = parent[node]
        return chain

    dist = [[0] * K for _ in range(K)]
    for a in range(K):
        chain_a = ancestors(f"leaf{a}")
        for b in range(K):
            common = next(x for x in chain_a if x in set(ancestors(f"leaf{b}")))
            dist[a][b] = (depth[f"leaf{a}"] - depth[common]) + (depth[f"leaf{b}"] - depth[common])
    return dist


DIST = leaf_distance()


def covering_antichains(size):
    """枚举所有「恰好 size 个、互不包含、并集覆盖全部叶子」的树节点组合。"""
    results = []

    def expand(front):
        if len(front) == size:
            results.append(tuple(sorted(front)))
            return
        if len(front) > size:
            return
        for node in front:
            if CHILDREN[node]:
                expand([n for n in front if n != node] + CHILDREN[node])

    expand(["root"])
    return sorted(set(results))


def utility(q, tensor, base):
    """U(Q) = (1/K) Σ_j max_i c_i(j)；c 裁剪到 [0,1]，块数按 16 块饱和。"""
    models = len(q)
    total = 0.0
    for j in range(K):
        best = -1.0
        for i in range(models):
            value = base[j]
            for k in range(K):
                blocks = min(q[i][k] * BLOCKS_TOTAL, SATURATION)
                value += tensor[k][j] * blocks / SATURATION
            best = max(best, min(1.0, max(0.0, value)))
        total += best
    return total / K


def restricted_growth(length, ceiling):
    """受限增长串：枚举「至多 ceiling 块」的集合划分，专家可交换下不重不漏。"""
    out, current = [], [0] * length
    def rec(pos, top):
        if pos == length:
            out.append(tuple(current))
            return
        for value in range(min(top + 1, ceiling - 1) + 1):
            current[pos] = value
            rec(pos + 1, max(top, value))
    rec(0, -1)
    return out


def best_allocation(models, tensor, base):
    """穷举路由 × 精确运输 LP，返回全局最优 (U, Q, 路由)。"""
    r = [1.0 / models] * models
    p = [1.0 / K] * K
    best = (-1.0, None, None)
    for routing in restricted_growth(K, models):
        # 固定路由后目标线性：权重 w[i][k] = (1/K)·Σ_{j:ρ(j)=i} A[k][j]·H/16。
        cost = [[0.0] * K for _ in range(models)]
        for i in range(models):
            targets = [j for j in range(K) if routing[j] == i]
            for k in range(K):
                gain = sum(tensor[k][j] for j in targets) * BLOCKS_TOTAL / SATURATION / K
                cost[i][k] = -gain                      # LP 求最小，故取负
        q = _transport_lp(cost, r, p)
        value = utility(q, tensor, base)                # 用真实 argmax 路由评估
        if value > best[0]:
            best = (value, q, routing)
    return best


def semantic_allocation(models, tensor, base):
    """在所有大小为 models 的树切分里，取 U 最大的那个（给语义最强的一把）。"""
    r = [1.0 / models] * models
    p = [1.0 / K] * K
    best = (-1.0, None, None)
    for groups in covering_antichains(models):
        cost = [[min(DIST[k][m] for m in TREE[groups[i]]) for k in range(K)]
                for i in range(models)]
        q = _transport_lp(cost, r, p)
        value = utility(q, tensor, base)
        if value > best[0]:
            best = (value, q, groups)
    return best


def load_tensor(path):
    rows = json.load(open(path, encoding="utf-8"))
    if len(rows) != K:
        return None, None, len(rows)
    order = {row["source"]: row for row in rows}
    return ([order[k]["delta"] for k in range(K)], order[0]["before"], K)


def load_flips(path):
    rows = json.load(open(path, encoding="utf-8"))
    order = {row["source"]: row for row in rows}
    return [[(order[k]["flips"][j]["gained"], order[k]["flips"][j]["lost"])
             for j in range(K)] for k in sorted(order)]


def resample(flips, rng, questions=80):
    """按 McNemar 的条件分布重抽：不一致数 n 固定，只重抽其中「变好」的个数。"""
    tensor = []
    for row in flips:
        line = []
        for gained, lost in row:
            n = gained + lost
            if n == 0:
                line.append(0.0)
                continue
            draw = sum(1 for _ in range(n) if rng.random() < gained / n)
            line.append((2 * draw - n) / questions)
        tensor.append(line)
    return tensor


def partition(routing, models):
    """路由 → 无序划分。专家编号可交换，比较结构时必须先规范化，否则同一个划分
    换个专家下标就会被判成「不同」——criterion (c) 会因此误判。"""
    blocks = [frozenset(j for j in range(K) if routing[j] == i) for i in range(models)]
    return frozenset(b for b in blocks if b)


def describe(routing_or_groups, models):
    if isinstance(routing_or_groups, tuple) and isinstance(routing_or_groups[0], str):
        return " | ".join("+".join(SOURCES[i] for i in sorted(TREE[g]))
                          for g in routing_or_groups)
    return " | ".join("+".join(SOURCES[j] for j in range(K) if routing_or_groups[j] == i)
                      or "（空）" for i in range(models))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors", nargs="+",
                    default=["results/probes/s17_transfer.json"])
    ap.add_argument("--experts", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    available = []
    for path in args.tensors:
        if not Path(path).exists():
            print(f"⚠ 缺 {path}（未跑完），跳过")
            continue
        tensor, base, rows = load_tensor(path)
        if tensor is None:
            print(f"⚠ {path} 只有 {rows}/{K} 行，跳过（部分张量会让 M 扫描不可比）")
            continue
        available.append((path, tensor, base))
    if not available:
        raise SystemExit("没有完整的响应张量可用")

    for path, tensor, base in available:
        print(f"\n{'='*78}\n张量 {path}\n{'='*78}")
        r_uniform = None
        print(f"{'M':>2} {'均匀':>8} {'语义':>8} {'最优':>8} {'Δ=最优−语义':>12} "
              f"{'语义−均匀':>10}  结构是否不同")
        for models in args.experts:
            if not covering_antichains(models):
                print(f"{models:>2}   标签树切不出 {models} 组，跳过")
                continue
            q_uniform = [[1.0 / (models * K)] * K for _ in range(models)]
            check_marginals(q_uniform, [1.0 / models] * models, [1.0 / K] * K, 1e-9)
            u_uniform = utility(q_uniform, tensor, base)
            u_sem, q_sem, groups = semantic_allocation(models, tensor, base)
            u_opt, q_opt, routing = best_allocation(models, tensor, base)
            check_marginals(q_sem, [1.0 / models] * models, [1.0 / K] * K, 1e-7)
            check_marginals(q_opt, [1.0 / models] * models, [1.0 / K] * K, 1e-7)
            sem_routing = tuple(next(i for i in range(models) if k in TREE[groups[i]])
                                for k in range(K))
            differs = partition(routing, models) != partition(sem_routing, models)
            print(f"{models:>2} {u_uniform:>8.4f} {u_sem:>8.4f} {u_opt:>8.4f} "
                  f"{u_opt-u_sem:>+12.4f} {u_sem-u_uniform:>+10.4f}  "
                  f"{'不同 ✅' if differs else '相同 ❌ 无判别力'}")
            print(f"     语义 {describe(groups, models)}")
            print(f"     最优 {describe(routing, models)}")

    print(f"\n{'='*78}\n自助重采样：Δ_M > 0 的比例（{args.bootstrap} 次）\n{'='*78}")
    for path, _, base in available:
        flips = load_flips(path)
        rng = random.Random(args.seed)
        print(f"\n{path}")
        for models in args.experts:
            if not covering_antichains(models):
                continue
            positives, deltas = 0, []
            for _ in range(args.bootstrap):
                boot = resample(flips, rng)
                u_sem = semantic_allocation(models, boot, base)[0]
                u_opt = best_allocation(models, boot, base)[0]
                deltas.append(u_opt - u_sem)
                positives += u_opt - u_sem > 0
            if not deltas:
                continue
            deltas.sort()
            lo, hi = deltas[int(0.05 * len(deltas))], deltas[min(len(deltas) - 1,
                                                                int(0.95 * len(deltas)))]
            print(f"  M={models}  Δ>0 比例 {positives/args.bootstrap:>6.1%}   "
                  f"中位 {deltas[len(deltas)//2]:+.4f}   5–95% {lo:+.4f} … {hi:+.4f}")


if __name__ == "__main__":
    main()
