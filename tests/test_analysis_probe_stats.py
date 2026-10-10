"""analysis/probe_stats.py：重跑须与 results/analysis/PROBE-stats.* 一致，并核对论文引用的关键数字。"""
import unittest
from fractions import Fraction
from math import comb

from _analysis_common import EXPECTED, load, run, tmpdir, without_last_line, import_analysis

ps = import_analysis("probe_stats")


def cell(blk, train, scored):
    return next(c for c in blk["cells"] if c["train"] == train and c["scored"] == scored)


class ReproduceTests(unittest.TestCase):
    def test_outputs_reproduce(self):
        out = tmpdir(self)
        text = run("probe_stats.py", "--out", out)
        self.assertEqual(load(out / "PROBE-stats.json"), load(EXPECTED / "PROBE-stats.json"))
        self.assertEqual((out / "PROBE-stats.txt").read_text(encoding="utf-8").rstrip("\n"),
                         text.rstrip("\n"))
        self.assertEqual(without_last_line(text),
                         without_last_line((EXPECTED / "PROBE-stats.txt").read_text(encoding="utf-8")))


class ExactTestTests(unittest.TestCase):
    def test_mcnemar_exact_values(self):
        self.assertEqual(ps.mcnemar(0, 0), 1)
        self.assertEqual(ps.mcnemar(5, 5), 1)                      # 截到 1
        self.assertEqual(ps.mcnemar(20, 1), Fraction(2 * (1 + 21), 2 ** 21))
        self.assertEqual(ps.mcnemar(1, 20), ps.mcnemar(20, 1))     # 双侧对称
        n = 13
        for g in range(n + 1):
            want = min(Fraction(1), Fraction(2 * sum(comb(n, i) for i in range(min(g, n - g) + 1)), 2 ** n))
            self.assertEqual(ps.mcnemar(g, n - g), want)

    def test_fisher_matches_enumeration(self):
        # 逐表枚举：超几何概率不超过观测表的所有表求和（有理数，无容差）
        def brute(a, b, c, d):
            r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
            prob = lambda x: Fraction(comb(r1, x) * comb(r2, c1 - x), comb(n, c1))  # noqa: E731
            obs = prob(a)
            return sum(prob(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if prob(x) <= obs)
        for table in ((3, 1, 1, 3), (71, 23, 66, 42), (0, 5, 5, 0), (2, 2, 2, 2), (10, 0, 0, 10)):
            self.assertEqual(ps.fisher(*table), brute(*table))
        self.assertEqual(ps.fisher(2, 2, 2, 2), 1)

    def test_fisher_has_no_float_tolerance_defect(self):
        # 浮点容差 `<= observed + 1e-9` 会把观测表本身漏出尾部，给出偏小的 p（约 0.0236 / 0.0149）
        self.assertAlmostEqual(float(ps.fisher(71, 23, 66, 42)), 0.034633, places=6)
        self.assertAlmostEqual(float(ps.fisher(71, 23, 60, 41)), 0.021831, places=6)


class PaperNumberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load(EXPECTED / "PROBE-stats.json")

    def test_s0_accuracy(self):
        self.assertEqual(self.d["s0_accuracy"], [0.375, 0.4625, 0.35, 0.3875, 0.2, 0.3625, 0.2])

    def test_s17_cells_and_multiplicity(self):
        s17 = self.d["shuffles"]["s17"]
        self.assertEqual(s17["cells_measured"], 49)
        self.assertEqual(sorted(s17["nominal_cells"]),
                         sorted(["Abstract->Abstract", "Linear->Integral", "Linear->Statistics",
                                 "Statistics->Linear", "Statistics->Statistics"]))
        self.assertEqual(s17["bonferroni_cells"], [])
        e = s17["expected_nominal_under_null"]
        self.assertAlmostEqual(e["upper_bound_alpha_times_family"], 2.45)
        self.assertEqual(round(e["exact_test_at_observed_discordant_counts"], 2), 1.35)
        int_abs = cell(s17, "Integral", "Abstract")
        self.assertEqual((int_abs["gained"], int_abs["lost"]), (22, 10))
        self.assertEqual(round(int_abs["p"], 4), 0.0501)
        self.assertFalse(int_abs["nominal"])
        for (tr, sc), (g, l, p) in {("Abstract", "Abstract"): (21, 9, 0.043),
                                    ("Linear", "Integral"): (21, 6, 0.006),
                                    ("Linear", "Statistics"): (16, 3, 0.004),
                                    ("Statistics", "Linear"): (22, 8, 0.016),
                                    ("Statistics", "Statistics"): (18, 4, 0.004)}.items():
            c = cell(s17, tr, sc)
            self.assertEqual((c["gained"], c["lost"], round(c["p"], 3)), (g, l, p))

    def test_s23_cells_and_bonferroni(self):
        s23 = self.d["shuffles"]["s23"]
        self.assertEqual(s23["cells_measured"], 42)
        self.assertEqual(s23["n_nominal"], 4)
        self.assertEqual(s23["bonferroni_cells"], ["Differential->Statistics"])
        ds = cell(s23, "Differential", "Statistics")
        self.assertEqual((ds["gained"], ds["lost"]), (20, 1))
        self.assertEqual(ds["p_exact"], "11/524288")
        self.assertEqual(round(ds["p"], 6), 2.1e-05)
        self.assertEqual(round(ds["p_bonf"], 5), 0.00103)
        ls = cell(s23, "Linear", "Statistics")
        self.assertEqual((ls["gained"], ls["lost"], round(ls["p"], 3)), (14, 4, 0.031))
        sl = cell(s23, "Statistics", "Linear")
        self.assertEqual((sl["gained"], sl["lost"], round(sl["p"], 3)), (19, 8, 0.052))
        self.assertEqual(self.d["bonferroni_in_both_shuffles"], [])

    def test_statistics_and_generic_columns(self):
        st, gen = self.d["statistics_column"], self.d["generic_column"]
        self.assertEqual(st["pooled"], [111, 36])
        self.assertEqual({k: v["pooled"] for k, v in st["per_donor"].items()},
                         {"Integral": [22, 9], "Differential": [32, 8], "Abstract": [27, 12],
                          "Linear": [30, 7]})
        self.assertTrue(all(v["gains_exceed_losses_in_every_shuffle"] for v in st["per_donor"].values()))
        self.assertEqual(f"{st['p_nominal_if_exchangeable']:.2e}", "4.32e-10")
        self.assertEqual(st["strongest_donor_by_net_gain"]["s17"]["donor"], "Linear")
        self.assertEqual(st["strongest_donor_by_net_gain"]["s23"]["donor"], "Differential")
        self.assertEqual(gen["pooled"], [55, 60])
        self.assertEqual(round(gen["p_nominal_if_exchangeable"], 2), 0.71)
        self.assertEqual(st["s0_accuracy"], gen["s0_accuracy"])
        self.assertEqual(st["s0_accuracy"], 0.2)

    def test_label_pairs_and_fisher(self):
        pairs = self.d["pairs"]
        self.assertEqual({k: [v["gained"], v["lost"]] for k, v in pairs.items()},
                         {"Linear-Statistics": [71, 23], "Abstract-Linear": [66, 42],
                          "Integral-Differential": [60, 41], "algorithmic-generic": [44, 22]})
        self.assertEqual(round(pairs["Linear-Statistics"]["positive_fraction"], 3), 0.755)
        self.assertEqual(len(pairs["algorithmic-generic"]["measurements"]), 3)   # s23 缺 generic 行
        f = self.d["fisher_vs_linear_statistics"]
        self.assertEqual({k: round(v["p"], 3) for k, v in f.items()},
                         {"Abstract-Linear": 0.035, "Integral-Differential": 0.022,
                          "algorithmic-generic": 0.284})


if __name__ == "__main__":
    unittest.main()
