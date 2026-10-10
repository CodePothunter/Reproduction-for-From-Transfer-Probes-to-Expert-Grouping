"""分配矩阵的离线测试：Sinkhorn 配平、整数化与精确运输问题。"""
import math
import random
import unittest

from lwts.allocation import _transport_lp, check_marginals, integer_counts, mirror, uniform

R = [0.5, 0.5]
P = [0.4, 0.3, 0.2, 0.1]
S = [[0.9, 0.1, -0.4, 0.2],
     [0.1, 0.7, 0.3, -0.2]]


def objective(q, cost):
    return sum(q[i][k] * cost[i][k] for i in range(len(q)) for k in range(len(q[0])))


class MirrorTests(unittest.TestCase):
    def test_preserves_both_marginals(self):
        q = mirror(S, uniform(R, P), R, P, eta=2.0)
        check_marginals(q, R, P, tolerance=1e-8)

    def test_zero_step_returns_reference(self):
        """η=0 时核就是参考本身，参考已满足边际则原样返回。"""
        reference = uniform(R, P)
        q = mirror(S, reference, R, P, eta=0.0)
        for i in range(len(R)):
            for k in range(len(P)):
                self.assertAlmostEqual(q[i][k], reference[i][k], places=9)

    def test_higher_score_gets_more_mass(self):
        """在固定边际下，分数高的 cell 相对参考应获得更多质量。"""
        reference = uniform(R, P)
        q = mirror(S, reference, R, P, eta=3.0)
        # 模型 0 在来源 0 上分数更高，模型 1 在来源 1 上更高
        self.assertGreater(q[0][0], reference[0][0])
        self.assertGreater(q[1][1], reference[1][1])

    def test_large_eta_does_not_overflow(self):
        """对数域实现：η·score 很大时 exp 不应溢出。"""
        q = mirror(S, uniform(R, P), R, P, eta=500.0)
        check_marginals(q, R, P, tolerance=1e-8)
        self.assertTrue(all(math.isfinite(v) for row in q for v in row))

    def test_asymmetric_shape(self):
        r, p = [0.2, 0.3, 0.5], [0.25, 0.25, 0.25, 0.25]
        score = [[0.3, -0.2, 0.1, 0.0], [0.0, 0.4, -0.1, 0.2], [-0.3, 0.1, 0.5, -0.4]]
        check_marginals(mirror(score, uniform(r, p), r, p, 1.5), r, p, tolerance=1e-8)

    def test_rejects_nonpositive_reference(self):
        bad = [[0.25, 0.25, 0.25, 0.25], [0.0, 0.5, 0.25, 0.25]]
        with self.assertRaises(ValueError):
            mirror(S, bad, R, P, eta=1.0)

    def test_reports_nonconvergence(self):
        """迭代上限太低时必须明确失败，而不是返回违反边际的矩阵。"""
        with self.assertRaises(RuntimeError):
            mirror(S, uniform(R, P), R, P, eta=50.0, tolerance=1e-15, max_iter=1)

    def test_rejects_mismatched_marginals(self):
        with self.assertRaises(ValueError):
            mirror(S, uniform(R, P), [0.5, 0.6], P, eta=1.0)   # 总量不等
        with self.assertRaises(ValueError):
            mirror(S, uniform(R, P), [0.0, 1.0], P, eta=1.0)   # 非正边际


class IntegerCountsTests(unittest.TestCase):
    def test_exact_row_and_column_totals(self):
        q = mirror(S, uniform(R, P), R, P, eta=2.0)
        rows, cols = [50, 50], [40, 30, 20, 10]
        counts = integer_counts(q, rows, cols)
        self.assertEqual([sum(row) for row in counts], rows)
        self.assertEqual([sum(row[k] for row in counts) for k in range(4)], cols)
        self.assertTrue(all(isinstance(v, int) and v >= 0 for row in counts for v in row))

    def test_already_integral_is_preserved(self):
        q = [[2.0, 1.0], [1.0, 3.0]]
        self.assertEqual(integer_counts(q, [3, 4], [3, 4]), [[2, 1], [1, 3]])

    def test_deterministic(self):
        q = mirror(S, uniform(R, P), R, P, eta=1.7)
        a = integer_counts(q, [7, 7], [4, 4, 3, 3])
        self.assertEqual(a, integer_counts(q, [7, 7], [4, 4, 3, 3]))

    def test_totals_may_disagree_with_q_marginals(self):
        """整数预算是权威：即使与 Q 的边际不完全一致也必须严格满足。"""
        q = uniform(R, P)
        counts = integer_counts(q, [10, 5], [5, 5, 3, 2])
        self.assertEqual([sum(row) for row in counts], [10, 5])
        self.assertEqual([sum(row[k] for row in counts) for k in range(4)], [5, 5, 3, 2])

    def test_zero_budget(self):
        self.assertEqual(integer_counts(uniform(R, P), [0, 0], [0, 0, 0, 0]),
                         [[0, 0, 0, 0], [0, 0, 0, 0]])

    def test_rejects_inconsistent_totals(self):
        with self.assertRaises(ValueError):
            integer_counts(uniform(R, P), [5, 5], [1, 1, 1, 1])
        with self.assertRaises(ValueError):
            integer_counts(uniform(R, P), [5, 5], [3, 3, 2, 2.0])


class TransportLPTests(unittest.TestCase):
    """min ⟨Q, cost⟩ s.t. Q ∈ U(r, p) 的精确解（分配扫描与事后代理分析用它）。"""
    LOSS = [[1.0, 3.0, 2.0, 4.0],
            [2.5, 1.0, 3.0, 1.5]]

    def test_solves_exact_lp(self):
        q = _transport_lp(self.LOSS, R, P)
        check_marginals(q, R, P, tolerance=1e-9)
        best = objective(q, self.LOSS)
        # 与大量随机可行解比较：LP 解不应被任何一个超过
        rng = random.Random(17)
        for _ in range(300):
            score = [[rng.uniform(-3, 3) for _ in P] for _ in R]
            other = mirror(score, uniform(R, P), R, P, eta=1.0)
            self.assertLessEqual(best, objective(other, self.LOSS) + 1e-9)

    def test_known_optimum(self):
        """2×2 对角代价：最优解就是对角分配，目标值为 0。"""
        r = p = [0.5, 0.5]
        q = _transport_lp([[0.0, 1.0], [1.0, 0.0]], r, p)
        self.assertAlmostEqual(objective(q, [[0.0, 1.0], [1.0, 0.0]]), 0.0, places=12)
        self.assertAlmostEqual(q[0][0], 0.5, places=12)
        q = _transport_lp([[1.0, 0.0], [0.0, 1.0]], r, p)
        self.assertAlmostEqual(q[0][1], 0.5, places=12)

    def test_degenerate_marginals(self):
        """行列在西北角法中同时耗尽的退化情形。"""
        r, p = [0.5, 0.5], [0.5, 0.5]
        q = _transport_lp([[1.0, 2.0], [2.0, 1.0]], r, p)
        check_marginals(q, r, p, tolerance=1e-9)
        self.assertAlmostEqual(objective(q, [[1.0, 2.0], [2.0, 1.0]]), 1.0, places=12)

    def test_rectangular(self):
        r, p = [0.2, 0.3, 0.5], [0.25, 0.25, 0.25, 0.25]
        cost = [[3.0, 1.0, 2.0, 5.0], [2.0, 4.0, 1.0, 3.0], [1.0, 2.0, 4.0, 2.0]]
        q = _transport_lp(cost, r, p)
        check_marginals(q, r, p, tolerance=1e-9)
        best = objective(q, cost)
        rng = random.Random(5)
        for _ in range(200):
            score = [[rng.uniform(-3, 3) for _ in p] for _ in r]
            self.assertLessEqual(best, objective(mirror(score, uniform(r, p), r, p, 1.0), cost) + 1e-9)


if __name__ == "__main__":
    unittest.main()
