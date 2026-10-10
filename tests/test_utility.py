"""组合效用与配对收益的离线测试；全部用手工小矩阵，不需要模型。"""
import math
import unittest

from lwts.utility import paired_return, portfolio_utility

# 两个模型、三个来源。模型 0 擅长来源 0，模型 1 擅长来源 1、2。
C = [[0.8, 0.2, 0.4],
     [0.3, 0.9, 0.5]]
W = [0.5, 0.25, 0.25]


class HardUtilityTests(unittest.TestCase):
    def test_matches_hand_computation(self):
        """U = Σ_j w_j max_i c_ij。"""
        expected = 0.5 * 0.8 + 0.25 * 0.9 + 0.25 * 0.5
        self.assertAlmostEqual(portfolio_utility(C, W), expected)

    def test_only_the_best_model_per_source_counts(self):
        """降低非承担专家的能力不改变硬路由效用。"""
        worse = [[0.8, 0.2, 0.4], [0.0, 0.9, 0.5]]
        self.assertAlmostEqual(portfolio_utility(C, W), portfolio_utility(worse, W))

    def test_weights_select_deployment_importance(self):
        only_first = portfolio_utility(C, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(only_first, 0.8)


class SoftUtilityTests(unittest.TestCase):
    def test_matches_direct_formula(self):
        """Ũ_τ = Σ_j w_j τ log Σ_i exp(c_ij/τ)，与未平移的直接算法一致。"""
        tau = 0.5
        direct = sum(w * tau * math.log(sum(math.exp(c / tau) for c in col))
                     for w, col in zip(W, zip(*C)))
        self.assertAlmostEqual(portfolio_utility(C, W, tau), direct, places=12)

    def test_softbound_holds(self):
        """U ≤ Ũ_τ ≤ U + τ log M。"""
        hard = portfolio_utility(C, W)
        for tau in (0.01, 0.1, 0.5, 1.0, 5.0):
            soft = portfolio_utility(C, W, tau)
            gap = tau * math.log(len(C))
            self.assertLessEqual(hard - 1e-12, soft, f"τ={tau} 下界被突破")
            self.assertLessEqual(soft, hard + gap + 1e-12, f"τ={tau} 上界被突破")

    def test_bound_is_tight_when_models_tie(self):
        """所有模型能力相同时，软化效用正好等于 U + τ log M。"""
        tied = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        weights, tau = [0.5, 0.5], 0.3
        soft = portfolio_utility(tied, weights, tau)
        self.assertAlmostEqual(soft, 0.5 + tau * math.log(3), places=12)

    def test_approaches_hard_as_tau_shrinks(self):
        hard = portfolio_utility(C, W)
        previous = float("inf")
        for tau in (1.0, 0.1, 0.01, 1e-3):
            excess = portfolio_utility(C, W, tau) - hard
            self.assertLess(excess, previous, "τ 变小时软化偏移应单调下降")
            previous = excess
        self.assertAlmostEqual(portfolio_utility(C, W, 1e-9), hard, places=9)

    def test_small_tau_does_not_overflow(self):
        """未做最大值平移时 exp(1.0/1e-6) 会直接 OverflowError。"""
        value = portfolio_utility(C, W, 1e-6)
        self.assertTrue(math.isfinite(value))
        self.assertAlmostEqual(value, portfolio_utility(C, W), places=6)


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_inputs(self):
        for bad in ([], [[]], [[0.5, 0.5], [0.5]], [[1.5, 0.5], [0.5, 0.5]],
                    [[-0.1, 0.5], [0.5, 0.5]], [[float("nan"), 0.5], [0.5, 0.5]]):
            with self.assertRaises(ValueError):
                portfolio_utility(bad, [0.5, 0.5])

    def test_rejects_bad_weights(self):
        for bad in ([0.5, 0.5, 0.5], [0.5], [1.0, 0.5], [-0.5, 1.5]):
            with self.assertRaises(ValueError):
                portfolio_utility([[0.5, 0.5], [0.5, 0.5]], bad)

    def test_rejects_nonpositive_tau(self):
        for bad in (0, -0.5, float("inf")):
            with self.assertRaises(ValueError):
                portfolio_utility(C, W, bad)


class PairedReturnTests(unittest.TestCase):
    def test_gain_per_token(self):
        """g = (Ũ(Θ⁺) − Ũ(Θ⁻)) / n。"""
        after = [[0.8, 0.2, 0.4], [0.3, 0.95, 0.5]]
        expected = (portfolio_utility(after, W, 0.5) - portfolio_utility(C, W, 0.5)) / 1000
        self.assertAlmostEqual(paired_return(C, after, 1000, W, 0.5), expected)

    def test_no_change_is_zero(self):
        self.assertEqual(paired_return(C, C, 500, W, 0.5), 0.0)

    def test_regression_gives_negative_return(self):
        """效用下降必须得到负收益，不能截断到零。"""
        worse = [[0.8, 0.2, 0.4], [0.3, 0.1, 0.5]]
        self.assertLess(paired_return(C, worse, 100, W, 0.5), 0)

    def test_hard_routing_variant(self):
        after = [[0.8, 0.2, 0.4], [0.3, 0.95, 0.5]]
        self.assertAlmostEqual(paired_return(C, after, 100, W, None),
                               (0.25 * 0.05) / 100)

    def test_rejects_bad_tokens_and_shape(self):
        for tokens in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                paired_return(C, C, tokens, W, 0.5)
        with self.assertRaises(ValueError):
            paired_return(C, [[0.5, 0.5], [0.5, 0.5]], 10, W, 0.5)


if __name__ == "__main__":
    unittest.main()
