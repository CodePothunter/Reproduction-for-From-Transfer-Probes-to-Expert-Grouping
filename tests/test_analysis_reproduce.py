"""重跑 analysis/ 下的分析脚本，输出须与 results/analysis/ 中随包的文件逐项一致（CPU）。

除记录输出路径的末行外，文本输出逐字一致；JSON 逐项相等。LABELREF 与分配扫描各需约
1–1.5 分钟（bootstrap 与 LP 穷举）。
"""
import shutil
import unittest

from _analysis_common import (EXPECTED, RESULTS, assert_package_lwts, import_analysis, load, run,
                              tmpdir, without_last_line)


class PackageImportTests(unittest.TestCase):
    def test_subprocess_uses_package_lwts(self):
        assert_package_lwts()


class CandidatesTests(unittest.TestCase):
    def test_primary_analysis_reproduces(self):
        out = tmpdir(self)
        text = run("analyze_candidates.py", "--out", out)
        self.assertEqual(load(out / "CANDIDATES-analysis.json"),
                         load(EXPECTED / "CANDIDATES-analysis.json"))
        self.assertEqual(without_last_line(text),
                         without_last_line((EXPECTED / "CANDIDATES-analysis.txt").read_text(encoding="utf-8")))


class LabelReferenceTests(unittest.TestCase):
    def test_posthoc_label_reference_reproduces(self):
        out = tmpdir(self)
        shutil.copy(EXPECTED / "CANDIDATES-analysis.json", out)   # 自检读取主分析结果
        run("analyze_label_reference.py", "--out", out)
        self.assertEqual(load(out / "LABELREF-analysis.json"), load(EXPECTED / "LABELREF-analysis.json"))
        self.assertEqual(without_last_line((out / "LABELREF-analysis.txt").read_text(encoding="utf-8")),
                         without_last_line((EXPECTED / "LABELREF-analysis.txt").read_text(encoding="utf-8")))


class ProxyTests(unittest.TestCase):
    def test_proxy_diagnostic_reproduces(self):
        out = tmpdir(self)
        shutil.copy(EXPECTED / "CANDIDATES-analysis.json", out)
        run("proxy_posthoc.py", "--out", out)
        self.assertEqual(load(out / "PROXY-posthoc.json"), load(EXPECTED / "PROXY-posthoc.json"))
        self.assertEqual(without_last_line((out / "PROXY-posthoc.txt").read_text(encoding="utf-8")),
                         without_last_line((EXPECTED / "PROXY-posthoc.txt").read_text(encoding="utf-8")))


class AllocationSweepTests(unittest.TestCase):
    def test_sweep_reproduces(self):
        text = run("allocation_sweep.py", "--experts", 2, 3, 4, 5)
        self.assertEqual(text, (EXPECTED / "ALLOC-sweep.txt").read_text(encoding="utf-8"))

    def test_label_tree_cut_counts(self):
        sweep = import_analysis("allocation_sweep")
        self.assertEqual([len(sweep.covering_antichains(m)) for m in range(2, 8)], [1, 1, 1, 3, 3, 1])
        self.assertEqual(sweep.covering_antichains(4), [("algebra", "calculus", "code", "leaf4")])


class C0ConstructionTests(unittest.TestCase):
    def test_c0_construction_reproduces(self):
        out = tmpdir(self)
        run("c0_construction.py", "--out", out)
        got = load(out / "C0-construction.json")
        self.assertEqual(got, load(EXPECTED / "C0-construction.json"))
        self.assertTrue(got["derived_equals_C0"])
        self.assertTrue(got["tree_distance_cost_times_56"]["C0_is_tied_optimum"])
        self.assertEqual(got["covering_cut_counts_M2_to_M7"], [1, 1, 1, 3, 3, 1])
        summary = load(RESULTS / "candidates" / "training_summary.json")
        self.assertEqual(got["C0_blocks"], summary["candidates"]["C0"]["blocks"])


class ControlAgreementTests(unittest.TestCase):
    def test_control_agreement_reproduces(self):
        out = tmpdir(self)
        run("control_agreement.py", "--out", out)
        got = load(out / "CONTROL-agreement.json")
        self.assertEqual(got, load(EXPECTED / "CONTROL-agreement.json"))
        self.assertEqual(got["control_sources"], ["Integral", "Differential", "algorithmic", "generic"])
        self.assertEqual([round(x, 4) for x in got["sets"]["same_seed"]["range"]], [0.6925, 0.875])
        self.assertEqual([round(x, 4) for x in got["sets"]["cross_seed"]["range"]], [0.66, 0.8225])


if __name__ == "__main__":
    unittest.main()
