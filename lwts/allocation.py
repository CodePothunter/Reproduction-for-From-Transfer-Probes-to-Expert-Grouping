"""分配矩阵：可行集、整数化与精确运输问题。

Q[i][k] 是本轮预算中分给「模型 i × 来源 k」的比例，可行集为

    U(r, π) = { Q ≥ 0 : Q·1_K = r, Qᵀ·1_M = π }

行和保住每个模型的训练预算，列和保住来源曝光。本模块提供：

    check_marginals / uniform   可行性检查与均匀分配
    mirror                      Sinkhorn 配平：Sinkhorn(Q̄ ⊙ exp(η·S); r, π)，η=0 时
                                就是把一个正矩阵按行列边际配平（C0 的推导用它）
    integer_counts              连续分配 → 精确满足整数行列预算的块配额
    _transport_lp               min ⟨Q, cost⟩ s.t. Q ∈ U(r, π) 的精确解（分配扫描用）

Sinkhorn 全程在对数域做，避免 exp(η·score) 溢出；不收敛时明确报错而不是
返回一个不满足边际的矩阵——静默违反边际等于偷偷改动训练预算。
"""
import math

TOLERANCE = 1e-9


def dimensions(matrix):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("矩阵不能为空或参差不齐")
    if any(not math.isfinite(x) for row in matrix for x in row):
        raise ValueError("矩阵存在非有限值")
    return len(matrix), len(matrix[0])


def check_marginals(q, r, p, tolerance=1e-9):
    m, k = dimensions(q)
    if len(r) != m or len(p) != k or any(x < 0 for row in q for x in row):
        raise ValueError("分配维度或非负性错误")
    errors = [abs(sum(q[i])-r[i]) for i in range(m)]
    errors += [abs(sum(q[i][j] for i in range(m))-p[j]) for j in range(k)]
    if max(errors) > tolerance:
        raise ValueError("分配不满足行列预算")


def uniform(r, p):
    """均匀分配 Q = r πᵀ：r 和 p 是概率边际。"""
    return [[ri * pj for pj in p] for ri in r]


def _positive_marginals(r, p):
    """行列边际必须为正且总量相等，否则运输可行集为空。"""
    for name, vector in (("r", r), ("p", p)):
        if not vector:
            raise ValueError(f"{name} 不能为空")
        for value in vector:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 的每一项都必须是正的有限数")
    if not math.isclose(sum(r), sum(p), rel_tol=0, abs_tol=1e-9):
        raise ValueError("r 与 p 的总量必须相等，否则不存在可行分配")


def _logsumexp(values):
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(v - top) for v in values))


def mirror(score, reference, r, p, eta, tolerance=1e-9, max_iter=10000):
    """通用 Sinkhorn 配平：返回满足边际的 Q = Sinkhorn(reference ⊙ exp(η·score); r, p)。

    η=0 时 score 不起作用，结果就是把 reference 按行列边际配平。语义划分 C0
    的连续目标正是这样得到的：reference 在「来源 → 专家」映射命中的格子取 1，
    其余取 1e-6，配平后再交给 integer_counts（见 tests/test_c0_derivation.py）。

    reference 必须逐项为正：有零元的格子在配平中会被永久钉死为零。

    迭代在对数域进行，exp(η·score) 不会溢出。达到 max_iter 仍不满足 tolerance
    时抛 RuntimeError——返回一个违反边际的矩阵等于悄悄改变训练预算。
    """
    models, sources = dimensions(score)
    if dimensions(reference) != (models, sources):
        raise ValueError("reference 与 score 的形状必须一致")
    if any(value <= 0 for row in reference for value in row):
        raise ValueError("reference 必须逐项为正")
    if len(r) != models or len(p) != sources:
        raise ValueError("边际长度与矩阵形状不一致")
    _positive_marginals(r, p)
    if not isinstance(eta, (int, float)) or isinstance(eta, bool) or not math.isfinite(eta):
        raise ValueError("eta 必须是有限实数")
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter 必须是正整数")

    kernel = [[math.log(reference[i][k]) + eta * score[i][k] for k in range(sources)]
              for i in range(models)]
    log_r = [math.log(value) for value in r]
    log_p = [math.log(value) for value in p]
    f = [0.0] * models
    g = [0.0] * sources
    for _ in range(max_iter):
        for i in range(models):
            f[i] = log_r[i] - _logsumexp([kernel[i][k] + g[k] for k in range(sources)])
        for k in range(sources):
            g[k] = log_p[k] - _logsumexp([kernel[i][k] + f[i] for i in range(models)])
        # 刚更新过 g，列和已精确；只需检查行和。
        rows = [math.exp(_logsumexp([f[i] + kernel[i][k] + g[k] for k in range(sources)]))
                for i in range(models)]
        if max(abs(rows[i] - r[i]) for i in range(models)) <= tolerance:
            break
    else:
        raise RuntimeError(f"Sinkhorn 在 {max_iter} 次迭代内未收敛到 {tolerance}")
    q = [[math.exp(f[i] + kernel[i][k] + g[k]) for k in range(sources)] for i in range(models)]
    check_marginals(q, r, p, tolerance=max(tolerance, TOLERANCE) * 10)
    return q


def integer_counts(q, row_counts, col_counts):
    """把 Q 转成整数块配额：精确满足整数行列边际，并使与连续目标 T = total·Q/mass
    的平方误差最小。

    整数预算是权威的：即使与 Q 的边际不完全一致（例如 _apportion 的余数分配），也必须
    严格满足。做法是凸费用的最小费用流：每格可取任意整数，第 u 个单位的边际平方误差
    (u−t)² − (u−1−t)² = 2u − 1 − 2t 随 u 递增，因此把每格展开成逐单位的平行边
    （行→列），连续最短路逐单位增广即得全局最优。目标与边际一致时，最优解每格只取
    floor 或 ceil。
    """
    models, sources = dimensions(q)
    if any(value < 0 for row in q for value in row):
        raise ValueError("Q 不能有负项")
    if len(row_counts) != models or len(col_counts) != sources:
        raise ValueError("整数预算长度与矩阵形状不一致")
    for name, counts in (("row_counts", row_counts), ("col_counts", col_counts)):
        if any(type(c) is not int or c < 0 for c in counts):
            raise ValueError(f"{name} 必须是非负整数")
    total = sum(row_counts)
    if total != sum(col_counts):
        raise ValueError("行列整数总量必须相等")
    if total == 0:
        return [[0] * sources for _ in range(models)]
    mass = sum(sum(row) for row in q)
    if mass <= 0:
        raise ValueError("Q 的总质量必须为正")
    target = [[q[i][k] / mass * total for k in range(sources)] for i in range(models)]

    # 源 → 行（容量 row_counts）→ 格的逐单位边（容量 1，费用 2u−1−2t）→ 列 → 汇
    S, T = 0, 1 + models + sources
    n = T + 1
    graph = [[] for _ in range(n)]

    def add(u, v, cap, cost):
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])

    for i in range(models):
        add(S, 1 + i, row_counts[i], 0.0)
        for k in range(sources):
            for unit in range(1, min(row_counts[i], col_counts[k]) + 1):
                add(1 + i, 1 + models + k, 1, 2 * unit - 1 - 2 * target[i][k])
    for k in range(sources):
        add(1 + models + k, T, col_counts[k], 0.0)

    for _ in range(total):
        dist, prev = [math.inf] * n, [None] * n
        dist[S] = 0.0
        for _ in range(n):                           # Bellman-Ford（有负费用边）
            changed = False
            for u in range(n):
                if dist[u] == math.inf:
                    continue
                for idx, (v, cap, cost, _) in enumerate(graph[u]):
                    if cap > 0 and dist[u] + cost < dist[v] - 1e-12:
                        dist[v], prev[v], changed = dist[u] + cost, (u, idx), True
            if not changed:
                break
        if dist[T] == math.inf:                      # pragma: no cover - 总量相等时总可行
            raise RuntimeError("整数配额无可行解")
        v = T
        while v != S:
            u, idx = prev[v]
            graph[u][idx][1] -= 1
            graph[v][graph[u][idx][3]][1] += 1
            v = u

    counts = [[0] * sources for _ in range(models)]
    for i in range(models):
        for v, cap, _, _ in graph[1 + i]:
            if 1 + models <= v < T and cap == 0:
                counts[i][v - 1 - models] += 1
    return counts


def _transport_lp(cost, r, p, tolerance=1e-12):
    """精确运输问题：min ⟨Q,cost⟩ s.t. Q ∈ U(r,p)。

    运输单纯形（MODI）。进基与离基都取 Bland 规则的最小下标，既保证不循环，
    也让最优解并列时有确定的选择（并列按固定规则决定）。
    """
    models, sources = dimensions(cost)
    q = [[0.0] * sources for _ in range(models)]

    # 西北角法构造初始基本可行解；用并查集保证基是一棵生成树（含退化的零基元）。
    parent = list(range(models + sources))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i, k):
        a, b = find(i), find(models + k)
        if a == b:
            return False
        parent[a] = b
        return True

    basis = set()
    supply, demand = list(r), list(p)
    i = k = 0
    while i < models and k < sources:
        amount = min(supply[i], demand[k])
        q[i][k] = amount
        supply[i] -= amount
        demand[k] -= amount
        if union(i, k):
            basis.add((i, k))
        if supply[i] <= tolerance and i < models - 1:
            i += 1
        elif demand[k] <= tolerance and k < sources - 1:
            k += 1
        else:
            i += 1 if supply[i] <= tolerance else 0
            k += 1 if demand[k] <= tolerance else 0
            if supply[i - 1 if i else 0] > tolerance and demand[k - 1 if k else 0] > tolerance:
                break
    for a in range(models):                     # 补零基元使基连通且无环
        for b in range(sources):
            if len(basis) >= models + sources - 1:
                break
            if (a, b) not in basis and union(a, b):
                basis.add((a, b))

    for _ in range(10000):
        # 由基元解对偶：u_i + v_k = cost_ik，沿基构成的树传播。
        u = [None] * models
        v = [None] * sources
        u[0] = 0.0
        pending = True
        while pending:
            pending = False
            for a, b in basis:
                if u[a] is not None and v[b] is None:
                    v[b] = cost[a][b] - u[a]
                    pending = True
                elif v[b] is not None and u[a] is None:
                    u[a] = cost[a][b] - v[b]
                    pending = True
        entering = None
        for a in range(models):                 # Bland：取下标最小的负 reduced cost
            for b in range(sources):
                if (a, b) in basis or u[a] is None or v[b] is None:
                    continue
                if cost[a][b] - u[a] - v[b] < -1e-10:
                    entering = (a, b)
                    break
            if entering:
                break
        if entering is None:
            return q

        # 在基构成的树里找 entering 两端之间的唯一回路。
        adjacency = {}
        for a, b in basis:
            adjacency.setdefault(("r", a), []).append(("c", b))
            adjacency.setdefault(("c", b), []).append(("r", a))
        start, goal = ("r", entering[0]), ("c", entering[1])
        stack, came = [start], {start: None}
        while stack:
            node = stack.pop()
            if node == goal:
                break
            for nxt in adjacency.get(node, ()):
                if nxt not in came:
                    came[nxt] = node
                    stack.append(nxt)
        path, node = [], goal
        while node is not None:
            path.append(node)
            node = came[node]
        path.reverse()
        cycle = [entering]
        for x, y in zip(path, path[1:]):
            cycle.append((x[1], y[1]) if x[0] == "r" else (y[1], x[1]))

        minus = cycle[1::2]
        theta, leaving = min(((q[a][b], (a, b)) for a, b in minus), key=lambda t: (t[0], t[1]))
        for index, (a, b) in enumerate(cycle):
            q[a][b] += theta if index % 2 == 0 else -theta
        basis.add(entering)
        basis.discard(leaving)
        q[leaving[0]][leaving[1]] = 0.0
    raise RuntimeError("运输单纯形未在迭代上限内收敛")
