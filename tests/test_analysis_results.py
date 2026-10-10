"""results/ 下随包文件的一致性，以及论文引用数字与这些文件的对应（CPU，只读）。"""
import re
import unittest

from _analysis_common import EXPECTED, PKG, RESULTS, load

CANDS = ("C0", "C1", "C2", "C3")
SEEDS = (2001, 2002)
BLOCKS = {
    "C0": [[0, 0, 7, 7, 0, 0, 0], [7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7], [1, 1, 1, 1, 8, 1, 1]],
    "C1": [[2] * 7 for _ in range(4)],
    "C2": [[0, 0, 0, 6, 8, 0, 0], [0, 8, 2, 2, 0, 0, 2], [0, 0, 0, 0, 0, 8, 6], [8, 0, 6, 0, 0, 0, 0]],
    "C3": [[0, 0, 0, 7, 7, 0, 0], [7, 7, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 7, 7], [1, 1, 8, 1, 1, 1, 1]],
}
ROUTES = {"C0": [1, 1, 0, 0, 3, 2, 2], "C1": [0] * 7, "C2": [3, 1, 3, 0, 0, 2, 2], "C3": [1, 1, 3, 0, 0, 2, 2]}


def pp(x, nd=2):
    return round(100 * x, nd)


class EvalFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ev = {(c, s): load(RESULTS / "candidates" / f"{c}-eval-s{s}.json") for c in CANDS for s in SEEDS}

    def test_cleaned_fields(self):
        for (c, s), d in self.ev.items():
            self.assertNotIn("checkpoint", d)
            self.assertEqual((d["eval_seed"], d["offset"], d["n"], d["generated"]), (s, 0, 400, 2800))
            self.assertEqual(d["routing"], ROUTES[c])
            for k in range(7):
                self.assertEqual(len(d["test_rows"][str(k)]), 400)
                ids = d["test_ids"][str(k)]
                self.assertEqual(len(ids), 400)
                self.assertEqual(len(set(ids)), 400)
                self.assertTrue(all(re.fullmatch(r"[0-9a-f]{16}", x) for x in ids))

    def test_same_items_for_every_candidate_and_seed(self):
        ref = self.ev[("C0", 2001)]["test_ids"]
        for d in self.ev.values():
            self.assertEqual(d["test_ids"], ref)

    def test_eval_seconds(self):
        secs = [d["seconds"] for d in self.ev.values()]
        self.assertTrue(all(32000 < x < 32500 for x in secs), secs)   # 论文："约 32,300 s"


class TrainingSummaryTests(unittest.TestCase):
    def test_blocks_routes_tokens(self):
        t = load(RESULTS / "candidates" / "training_summary.json")
        proxy = load(EXPECTED / "PROXY-posthoc.json")
        for c in CANDS:
            v = t["candidates"][c]
            self.assertEqual(v["blocks"], BLOCKS[c])
            self.assertEqual(proxy["blocks"][c], BLOCKS[c])
            self.assertEqual(v["routing"], ROUTES[c])
            self.assertEqual(proxy["block_routing"][c], ROUTES[c])
            self.assertTrue(all(sum(row) == 14 for row in v["blocks"]))
            self.assertTrue(all(sum(row[k] for row in v["blocks"]) == 8 for k in range(7)))
            self.assertEqual(round(v["block_tokens_realized"]["mean"]), 35144)
            self.assertEqual((v["processed_tokens"], v["completion_tokens"]), (2483424, 1968045))
            self.assertTrue(4500 <= v["train_wall_seconds"] <= 4600)
        self.assertEqual(t["warmup_W_round0"]["blocks"], [[2] * 7 for _ in range(4)])

    def test_only_listed_fields(self):
        t = load(RESULTS / "candidates" / "training_summary.json")
        self.assertEqual(set(t), {"note", "sources", "candidates", "warmup_W_round0"})
        keys = {"blocks", "allocation", "processed_tokens", "completion_tokens",
                "block_tokens_realized", "routing", "train_wall_seconds", "eval_wall_seconds",
                "eval_seconds_in_process", "eval_generations"}
        for v in t["candidates"].values():
            self.assertEqual(set(v), keys)
        self.assertEqual(set(t["warmup_W_round0"]),
                         {"blocks", "allocation", "processed_tokens", "completion_tokens",
                          "block_tokens_realized", "note"})


class AnalysisNumberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load(EXPECTED / "CANDIDATES-analysis.json")
        cls.l = load(EXPECTED / "LABELREF-analysis.json")
        cls.p = load(EXPECTED / "PROXY-posthoc.json")

    def test_primary_contrasts(self):
        want = {"C1-C0": (-2.54, [-4.04, -1.12]), "C2-C0": (-2.43, [-3.84, -1.14]),
                "C3-C0": (-1.54, [-2.89, -0.20])}
        for key, (d, ci) in want.items():
            v = self.c["contrasts"][key]
            self.assertEqual((pp(v["dU_all"]), [pp(x) for x in v["dU_all_ci95"]]), (d, ci))
        self.assertEqual({c: round(v["U_all"], 4) for c, v in self.c["point"].items()},
                         {"C0": 0.5036, "C1": 0.4782, "C2": 0.4793, "C3": 0.4882})
        flips = [x for c in ("C0", "C1", "C2") for x in self.c["meta"][c]["seed_flip_rate"]]
        self.assertEqual((min(flips), max(flips)), (0.205, 0.34))

    def test_posthoc_label_reference(self):
        v = self.l["pairs"]["C2-C1"]
        self.assertEqual((pp(v["dU_all"]), [pp(x) for x in v["dU_all_ci95"]]), (0.11, [-1.25, 1.54]))
        self.assertEqual(pp(self.l["flow_sensitivity"]["C1"]["range"][1], 3), 0.875)
        self.assertEqual(pp(self.l["flow_sensitivity"]["C2"]["range"][1], 3), 0.875)
        sim = self.l["simultaneous"]
        self.assertEqual(round(sim["critical"], 3), 3.025)
        self.assertEqual(sim["entirely_above_zero"], [])
        qs = [r["critical"] for r in self.l["stability"]]
        self.assertEqual((len(qs), round(min(qs), 3), round(max(qs), 3)), (22, 2.965, 3.094))

    def test_proxy_numbers(self):
        r = self.p["results"]
        self.assertEqual(pp(r["block"]["s17"]["delta_vs_C0"]["C2"]["U_all"]), 2.01)
        self.assertEqual(pp(r["block"]["s23*"]["delta_vs_C0"]["C2"]["U_all"]), 0.67)
        self.assertEqual(pp(r["zero_transfer"]["s17"]["delta_vs_C0"]["C2"]["U_all"]), 0.02)
        self.assertEqual(round(r["argmax"]["s17"]["U"]["C0"]["U_all"], 4), 0.4336)
        self.assertTrue(self.p["self_checks"]["C2_equals_s17_LP_optimum"])
        self.assertEqual(round(self.p["self_checks"]["s17_LP_optimum_U"], 4), 0.4426)
        self.assertIn(" 4   0.3946   0.4348   0.4426", (EXPECTED / "ALLOC-sweep.txt").read_text(encoding="utf-8"))

    def test_param_distance_consistency(self):
        d = load(EXPECTED / "PARAM-distance.json")
        self.assertEqual(d["matched_experts"], [1, 2])
        for row in d["experts"]:
            self.assertEqual(row["identical_blocks_in_C0_and_C3"], BLOCKS["C0"][row["expert"]] == BLOCKS["C3"][row["expert"]])
            self.assertAlmostEqual(row["ratio_C0_C3_over_C0_W"], row["dist_C0_C3"] / row["dist_C0_W"])
        self.assertEqual([round(x, 3) for x in d["matched_dist_C0_C3_range"]], [0.027, 0.028])
        self.assertEqual([round(x, 1) for x in d["matched_dist_C0_W_range"]], [1.4, 1.6])
        self.assertEqual([round(x, 2) for x in d["unmatched_dist_C0_C3_range"]], [0.88, 0.94])


class ProtocolSummaryTests(unittest.TestCase):
    def test_decoding_summary(self):
        d = load(RESULTS / "protocol" / "decoding_variance_summary.json")
        self.assertEqual(d["k_chosen"], 2)
        by_k = d["paired_se_extrapolation"]["by_k"]
        self.assertEqual([pp(by_k[k]["se"]) for k in ("1", "2", "4")], [0.98, 0.69, 0.48])
        self.assertEqual([pp(by_k[k]["mde"]) for k in ("1", "2", "4")], [2.74, 1.93, 1.36])
        rows = d["decoding_split"]["per_source"]
        rates = [r["two_decoding_disagreement_rate"] for r in rows]
        shares = [r["decoding_share"] for r in rows]
        self.assertEqual((round(min(rates), 2), round(max(rates), 2)), (0.21, 0.34))
        self.assertEqual((round(min(shares), 2), round(max(shares), 2)), (0.42, 0.67))

    def test_manifest_summary(self):
        m = load(RESULTS / "protocol" / "manifest_summary.json")
        self.assertTrue(m["eval_test_ids_equal_hashed_first_400_sorted"])
        self.assertEqual(m["min_test_count"], 421)
        self.assertEqual(m["first_400_test_sorted_by_id_by_dataset"]["algorithmic"],
                         {"evalplus/mbppplus": 376, "nvidia/OpenCodeInstruct": 24})
        self.assertEqual(m["test_by_dataset"]["algorithmic"],
                         {"nvidia/OpenCodeInstruct": 768, "evalplus/mbppplus": 376})


class TraceTests(unittest.TestCase):
    def test_results_and_analysis_have_no_local_paths(self):
        # 绝对路径与 GPU UUID 一类的本机信息不应出现在随包结果与分析脚本里
        banned = re.compile(r"(?<![\w.])/(home|root|Users|mnt|data|tmp)/|GPU-[0-9a-f]{8}-")
        for root in (RESULTS, PKG / "analysis"):
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in (".json", ".txt", ".py"):
                    hit = banned.search(path.read_text(encoding="utf-8"))
                    self.assertIsNone(hit, f"{path}: {hit and hit.group(0)}")

    def test_s23_metadata_cleaned(self):
        d = load(RESULTS / "probes" / "s23_transfer_partial.json")
        self.assertEqual(d["rows_missing"], ["generic"])
        self.assertEqual(set(d), {"complete", "rows_complete", "rows_missing", "stopped_reason",
                                  "source_log", "rows"})
        self.assertTrue(d["stopped_reason"].startswith("run stopped before the last row"))
        self.assertTrue(d["source_log"].endswith("(not included in this package)"))


if __name__ == "__main__":
    unittest.main()
