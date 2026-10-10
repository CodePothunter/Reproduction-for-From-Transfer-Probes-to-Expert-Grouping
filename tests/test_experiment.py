"""实验闭环测试：分配、区间原子性、检查点恢复与可逆探针。

  - 同来源数据流在所有方法中保持一致；整数执行预算同时满足来源和模型边际
  - train/feedback/route/audit/test 互不混用
  - 保存恢复后的继续训练与不中断轨迹一致；较早格式的检查点仍可分叉载入
  - 探针出错也能恢复参数、优化器、随机状态、scheduler 和数据游标
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy

from lwts.allocation import check_marginals
from lwts.backend import SyntheticBackend
from lwts.experiment import METHODS, Experiment
from lwts.utility import paired_return

MODELS, SOURCES = 2, 4
COUNTS = {"train": 40, "feedback": 12, "route": 6, "audit": 12, "test": 8}


def manifest():
    records = []
    for k in range(SOURCES):
        for split, count in COUNTS.items():
            for i in range(count):
                records.append({"id": f"{split}-{k}-{i}", "group_id": f"g{k}{i}",
                                "source": k, "split": split, "prompt": "p", "completion": "c",
                                "tokens": 300, "completion_tokens": 120,
                                "reference": {"kind": "x"}, "metadata": {"dataset": "d"}})
    return {"schema_version": 1, "records": records}


def config(method="uniform", **over):
    settings = {"models": MODELS, "sources": SOURCES, "method": method, "seed": 17,
                "tokens_per_interval": 960, "block_tokens": 240,
                "tau": 0.1, "eta": 1.0, "gamma": 0.1, "rho": 0.9,
                "fixed_blocks": [[1, 1, 0, 0], [0, 0, 1, 1]],
                "warmup_rounds": 1,
                # 这些机制测试刻意跑在退化区间里：4 块/区间少于 8 个 cell，
                # 便于用很小的数验证分发、原子性与恢复。真实实验里这是被
                # _block_plan 拦截的配置，必须显式承认才放行。
                "allow_degenerate_blocks": True}
    settings.update(over)
    return {"experiment": settings}


class Recorder(SyntheticBackend):
    """记录训练与评估分别看过哪些样本，用于检查 split 隔离。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trained, self.evaluated = [], []

    def train_block(self, model_id, records):
        self.trained.extend(r["id"] for r in records)
        return super().train_block(model_id, records)

    def evaluate(self, model_id, records):
        self.evaluated.extend(r["id"] for r in records)
        return super().evaluate(model_id, records)


def build(method="uniform", backend=None, **over):
    backend = backend or SyntheticBackend(MODELS, SOURCES, seed=3, response=5e-4)
    return Experiment(config(method, **over), manifest(), backend)


def comparable(value):
    """把状态里可能出现的 numpy 数组换成可直接比较的字节串。"""
    if isinstance(value, numpy.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [comparable(v) for v in value]
    return value


class ConfigTests(unittest.TestCase):
    def test_requires_explicit_parameters(self):
        """缺失的参数必须报错，不能被当作已验证默认值。"""
        for name in ("tokens_per_interval", "block_tokens", "tau", "eta", "gamma", "rho"):
            with self.assertRaises(ValueError, msg=name):
                build(**{name: None})

    def test_rejects_unknown_method(self):
        for method in ("nope", "adaptive", "", "Uniform"):
            with self.assertRaises(ValueError, msg=method):
                build(method)

    def test_fixed_needs_a_well_formed_block_matrix(self):
        with self.assertRaises(ValueError):
            build("fixed", fixed_blocks=None)
        for bad in ([[1, 1, 0, 0]], [[1, 1, 0], [0, 0, 1]], [[1, 1, 0, 0], [0, 0, 1, -1]],
                    [[1, 1, 0, 0], [0, 0, 1, 1.0]]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                build("fixed", fixed_blocks=bad)


class BlockPlanTests(unittest.TestCase):
    """block_tokens 是下界不是目标——名义值失效时必须报错而不是静默跑下去。"""

    def healthy(self, **over):
        """32 块/区间、每块恰好 2 条记录：两个致命项都不触发。"""
        settings = {"tokens_per_interval": 240 * 32, "allow_degenerate_blocks": False}
        return build(**{**settings, **over})

    def test_measures_real_block_size_not_nominal(self):
        plan = self.healthy().plan
        # 记录 120 token，阈值 240 → 每块恰好 2 条，实测等于名义
        self.assertEqual(plan["measured"]["mean"], 240)
        self.assertEqual(plan["measured"]["ratio_to_nominal"], 1.0)
        self.assertEqual(plan["blocks_per_interval"], 32)
        for k in range(SOURCES):
            entry = plan["measured"]["by_source"][k]
            self.assertEqual(entry["records_per_block"], 2)
            self.assertEqual(entry["blocks_per_full_pass"], 20)

    def test_worst_source_decides_the_tolerance(self):
        """来源之间记录长度可差一个数量级，平均会把两边的失配都掩盖掉。"""
        records = manifest()["records"]
        for row in records:
            if row["source"] == 1:                       # 只让来源 1 的记录变长
                row["completion_tokens"] = 1200
        settings = config(tokens_per_interval=240 * 32,
                          allow_degenerate_blocks=False)
        with self.assertRaises(ValueError) as caught:
            Experiment(settings, {"schema_version": 1, "records": records},
                       SyntheticBackend(MODELS, SOURCES))
        self.assertIn("来源 1", str(caught.exception))

    def test_recycled_source_stream_warns(self):
        """来源曝光按 token 均分，但各来源的 token 总量可以差一个数量级。"""
        records = manifest()["records"]
        for row in records:
            if row["source"] == 2 and row["split"] == "train":
                row["completion_tokens"] = 12          # 该来源整条流只有 480 token
        experiment = Experiment(config(tokens_per_interval=240 * 32,
                                       allow_degenerate_blocks=False),
                                {"schema_version": 1, "records": records},
                                SyntheticBackend(MODELS, SOURCES))
        self.assertTrue(any("回绕重训" in w for w in experiment.plan["warnings"]))
        self.assertGreater(
            experiment.plan["measured"]["by_source"][2]["passes_per_interval"], 1)

    def test_block_tokens_below_record_length_is_rejected(self):
        """block_tokens 小于记录长度时名义值失效：块至少是一条记录。"""
        with self.assertRaises(ValueError) as caught:
            self.healthy(block_tokens=10, tokens_per_interval=10 * 32)
        self.assertIn("倍", str(caught.exception))
        # 显式承认后仍可运行，但 plan 里留着证据
        acknowledged = build(block_tokens=10, tokens_per_interval=10 * 32)
        self.assertEqual(acknowledged.plan["measured"]["mean"], 120)
        self.assertEqual(acknowledged.plan["measured"]["ratio_to_nominal"], 12.0)

    def test_fewer_blocks_than_cells_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.healthy(tokens_per_interval=240 * 4)
        self.assertIn("cell", str(caught.exception))

    def test_coarse_granularity_warns_without_blocking(self):
        """粒度粗不阻断运行，但必须留下痕迹：一块占专家预算的比例太大。"""
        coarse = self.healthy(tokens_per_interval=240 * 8)      # 8 块 = 8 cell
        self.assertEqual(coarse.plan["granularity"], 0.25)
        self.assertTrue(any("粒度" in w for w in coarse.plan["warnings"]))
        self.assertFalse(self.healthy().plan["warnings"])

    def test_round_log_reports_realized_block_size(self):
        experiment = self.healthy()
        experiment.run(1)
        log = [e for e in experiment.logs if "block_tokens_realized" in e][-1]
        self.assertEqual(log["block_tokens_realized"]["mean"], 240)
        self.assertEqual(len(log["allocation"]), MODELS)


class AllocationTests(unittest.TestCase):
    def test_every_method_preserves_budgets(self):
        """两个方法都必须满足同一套模型预算与来源曝光。"""
        for method in METHODS:
            experiment = build(method)
            experiment.run(3)
            check_marginals(experiment.allocation, experiment.model_budget,
                            experiment.source_exposure, tolerance=1e-7)

    def test_integer_blocks_match_both_marginals(self):
        for method in METHODS:
            experiment = build(method)
            experiment.run(2)
            blocks = [entry["blocks"] for entry in experiment.logs if "blocks" in entry][-1]
            self.assertEqual([sum(row) for row in blocks], [2, 2])
            self.assertEqual([sum(row[k] for row in blocks) for k in range(SOURCES)],
                             [1, 1, 1, 1])

    def test_warmup_is_uniform(self):
        experiment = build("fixed")
        experiment.run(1)
        base = [[0.5 * 0.25] * SOURCES for _ in range(MODELS)]
        for row, expected in zip(experiment.allocation, base):
            for a, b in zip(row, expected):
                self.assertAlmostEqual(a, b, places=12)


class IsolationTests(unittest.TestCase):
    def test_methods_share_the_same_source_stream(self):
        """所有方法的来源流必须逐条相同——否则比较的是数据而不是分配。"""
        streams = []
        for method in METHODS:
            experiment = build(method)
            streams.append({k: [r["id"] for r in rows]
                            for k, rows in experiment.streams.items()})
        for other in streams[1:]:
            self.assertEqual(streams[0], other)

    def test_training_never_sees_non_train_records(self):
        backend = Recorder(MODELS, SOURCES, seed=3)
        build("uniform", backend=backend).run(3)
        self.assertTrue(backend.trained, "应当确实训练过")
        self.assertTrue(all(rid.startswith("train-") for rid in backend.trained))

    def test_route_and_test_are_never_touched(self):
        backend = Recorder(MODELS, SOURCES, seed=3)
        experiment = build("uniform", backend=backend)
        experiment.run(3)
        experiment.probe(0, 1)
        seen = set(backend.trained) | set(backend.evaluated)
        self.assertFalse([rid for rid in seen if rid.startswith(("route-", "test-"))])

    def test_online_loop_does_not_evaluate(self):
        """在线循环（构造与 run）不做任何评估；audit 只服务 probe。"""
        for method in METHODS:
            backend = Recorder(MODELS, SOURCES, seed=3)
            build(method, backend=backend).run(3)
            self.assertEqual(backend.evaluated, [], method)

    def test_missing_split_is_rejected(self):
        thin = manifest()
        thin["records"] = [r for r in thin["records"]
                           if not (r["split"] == "audit" and r["source"] == 2)]
        with self.assertRaises(ValueError):
            Experiment(config(), thin, SyntheticBackend(MODELS, SOURCES))


class RoundTests(unittest.TestCase):
    def test_round_advances_and_cursors_move(self):
        experiment = build("uniform")
        experiment.run(2)
        self.assertEqual(experiment.round, 2)
        self.assertTrue(all(v > 0 for v in experiment.cursors.values()))

    def test_failure_rolls_back_the_whole_round(self):
        """区间是原子的：中途失败不能留下半个区间。"""
        experiment = build("uniform")
        experiment.run(1)
        before = copy.deepcopy(experiment._state())

        calls = {"n": 0}
        original = experiment.backend.train_block

        def failing(model_id, records):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("模拟训练中断")
            return original(model_id, records)

        experiment.backend.train_block = failing
        with self.assertRaises(RuntimeError):
            experiment.run_round()
        experiment.backend.train_block = original
        self.assertEqual(comparable(before), comparable(experiment._state()))

    def test_run_rejects_backward_target(self):
        experiment = build("uniform")
        experiment.run(2)
        with self.assertRaises(ValueError):
            experiment.run(1)

    def test_one_log_entry_per_round(self):
        experiment = build("fixed")
        experiment.run(3)
        rounds = [e["round"] for e in experiment.logs if "blocks" in e]
        self.assertEqual(rounds, [0, 1, 2])


def _rewrite_index(path, arrays):
    """把检查点改写成较早的格式：npz 里放大向量，state 里带上额外的键。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    blob = Path(path).parent / payload["vectors"]["path"]
    numpy.savez(blob, **arrays)
    payload["vectors"]["keys"] = sorted(arrays)
    payload["vectors"]["sha256"] = hashlib.sha256(blob.read_bytes()).hexdigest()
    return payload, blob


class CheckpointTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_trajectory(self):
        straight = build("fixed")
        straight.run(4)

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ckpt.json"
            interrupted = build("fixed")
            interrupted.run(2)
            interrupted.checkpoint(path)
            interrupted.run(4)
            self.assertEqual(comparable(straight._state()), comparable(interrupted._state()))

            revived = build("fixed")
            self.assertEqual(revived.resume(path), 2)
            revived.run(4)
            self.assertEqual(comparable(straight._state()), comparable(revived._state()))

    def test_rollback_snapshots_are_released_but_checkpoints_are_not(self):
        """回滚快照用完即弃；检查点引用的那份绝不能被回收。

        真实后端每份快照是数百 MB 的张量文件，不回收就会一直堆积。
        """
        class Counting(SyntheticBackend):
            discarded = None

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.discarded = []

            def discard(self, state):
                self.discarded.append(state)

        backend = Counting(MODELS, SOURCES, seed=3, response=5e-4)
        experiment = build("uniform", backend=backend)
        experiment.run(3)
        self.assertEqual(len(backend.discarded), 3, "每个区间的回滚快照都该释放")
        experiment.probe(0, 1)
        self.assertEqual(len(backend.discarded), 4, "probe 的回滚快照同样要释放")
        with tempfile.TemporaryDirectory() as root:
            experiment.checkpoint(Path(root) / "ckpt.json")
        self.assertEqual(len(backend.discarded), 4, "检查点引用的快照不能被释放")

    def test_checkpoint_keeps_the_format_with_an_empty_vector_file(self):
        """格式 lwts-experiment-v2 不变：JSON 旁边带一个由索引核对的 npz，本包写的是空的。"""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ckpt.json"
            saved = build("uniform")
            saved.run(2)
            payload = saved.checkpoint(path)
            blob = path.with_name(path.name + Experiment.VECTOR_SUFFIX)
            self.assertTrue(blob.exists())
            self.assertEqual(payload["format"], "lwts-experiment-v2")
            self.assertEqual(payload["vectors"]["path"], blob.name)
            self.assertEqual(payload["vectors"]["keys"], [])
            self.assertEqual(payload["vectors"]["sha256"],
                             hashlib.sha256(blob.read_bytes()).hexdigest())
            self.assertEqual(set(payload["state"]),
                             {"backend", "cursors", "allocation", "round", "logs"})

    def test_tampered_vector_file_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ckpt.json"
            saved = build("uniform")
            saved.run(2)
            saved.checkpoint(path)
            blob = path.with_name(path.name + Experiment.VECTOR_SUFFIX)
            blob.write_bytes(blob.read_bytes() + b"\0")
            with self.assertRaises(ValueError):
                build("uniform").resume(path)
            blob.unlink()
            with self.assertRaises(FileNotFoundError):
                build("uniform").resume(path)

    def test_rejects_mismatched_config(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ckpt.json"
            saved = build("uniform")
            saved.run(1)
            saved.checkpoint(path)
            with self.assertRaises(ValueError):
                build("fixed").resume(path)
            with self.assertRaises(ValueError):
                build("uniform", eta=2.0).resume(path)

    def test_rejects_mismatched_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ckpt.json"
            saved = build("uniform")
            saved.run(1)
            saved.checkpoint(path)
            other = manifest()
            other["records"] = other["records"][:-1]
            for mode in Experiment.RESUME_MODES:
                with self.assertRaises(ValueError, msg=mode):
                    Experiment(config(), other, SyntheticBackend(MODELS, SOURCES)).resume(
                        path, mode=mode)

    def test_fork_from_warmup_and_diagnostic_reading(self):
        """候选从热身检查点分叉；热身之后的检查点拒绝分叉；diagnostic 只读不跑。"""
        with tempfile.TemporaryDirectory() as root:
            warm = Path(root) / "W.json"
            later = Path(root) / "later.json"
            source = build("uniform")
            source.run(1)
            source.checkpoint(warm)
            source.run(2)
            source.checkpoint(later)

            forked = build("fixed")
            self.assertEqual(forked.resume(warm, mode="fork"), 1)
            forked.run(2)
            executed = [e["blocks"] for e in forked.logs if e.get("round") == 1]
            self.assertEqual(executed, [config()["experiment"]["fixed_blocks"]])

            with self.assertRaises(ValueError):
                build("fixed").resume(later, mode="fork")

            reader = build("fixed")
            self.assertEqual(reader.resume(later, mode="diagnostic"), 2)
            with self.assertRaises(RuntimeError):
                reader.run(3)

    def test_older_checkpoint_with_vectors_and_extra_state_still_forks(self):
        """较早写出的检查点：npz 里有大向量，state 里有更多的键。

        分叉载入时仍按索引核对 npz 的存在性、sha256 与键集，但不载入其中的数组，
        也不读 state 里的额外键；训练侧状态必须与原检查点完全一致。
        """
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "W.json"
            source = build("uniform")
            source.run(1)
            source.checkpoint(path)
            expected = {"backend": source.backend.snapshot(), "cursors": dict(source.cursors),
                        "round": source.round, "allocation": source.allocation}

            arrays = {"0.params": numpy.ones(5, dtype=numpy.float32),
                      "0.vector": numpy.zeros(5, dtype=numpy.float32),
                      "1.params": numpy.ones(5, dtype=numpy.float32),
                      "1.vector": numpy.zeros(5, dtype=numpy.float32)}
            payload, blob = _rewrite_index(path, arrays)
            payload["state"].update({
                "extra_table": {"0,1": [{"round": 0, "tokens": 240, "gain": 0.1}]},
                "extra_map": {}, "extra_list": ["x"],
                "competence": [[0.5] * SOURCES for _ in range(MODELS)],
                "extra_counts": {"generated": 7},
                "extra_records": {"0": {"round": 0}, "1": {"round": 0}},
                "extra_tokens": 0, "extra_blocks": [0, 0], "extra_events": []})
            path.write_text(json.dumps(payload), encoding="utf-8")

            forked = build("fixed")
            self.assertEqual(forked.resume(path, mode="fork"), 1)
            self.assertEqual(forked.backend.snapshot(), expected["backend"])
            self.assertEqual(forked.cursors, expected["cursors"])
            self.assertEqual(forked.allocation, expected["allocation"])
            self.assertEqual(forked.competence, [[0.0] * SOURCES for _ in range(MODELS)],
                             "检查点里的能力表不被载入")
            forked.run(2)

            # 键集与索引不符：拒绝
            payload["vectors"]["keys"] = ["0.params"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                build("fixed").resume(path, mode="fork")


class ProbeTests(unittest.TestCase):
    def test_probe_is_fully_reversible(self):
        experiment = build("uniform")
        experiment.run(2)
        before = copy.deepcopy(experiment._state())
        experiment.probe(1, 2)
        self.assertEqual(comparable(before), comparable(experiment._state()))
        self.assertEqual(experiment.round, 2)

    def test_probe_reports_transfer_and_return(self):
        experiment = build("uniform")
        experiment.run(2)
        result = experiment.probe(0, 0)
        self.assertEqual(len(result["transfer"]), SOURCES)
        self.assertEqual(len(result["before"]), SOURCES)
        self.assertIn("portfolio_return", result)
        self.assertGreater(result["completion_tokens"], 0)

    def test_portfolio_return_counts_other_experts_as_zero(self):
        """构造期不做能力评估，所以其余专家的行恒为零：portfolio_return 是本专家
        的加权自身增益，不是组合口径。"""
        experiment = build("uniform")
        result = experiment.probe(1, 2, blocks=2)
        zeros = [[0.0] * SOURCES for _ in range(MODELS)]
        before, after = copy.deepcopy(zeros), copy.deepcopy(zeros)
        before[1], after[1] = result["before"], result["after"]
        self.assertEqual(result["portfolio_return"],
                         paired_return(before, after, result["completion_tokens"],
                                       experiment.weights, experiment.tau))

    def test_probe_rolls_back_on_exception(self):
        experiment = build("uniform")
        experiment.run(2)
        before = copy.deepcopy(experiment._state())
        experiment.backend.train_block = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("探针失败"))
        with self.assertRaises(RuntimeError):
            experiment.probe(0, 1)
        self.assertEqual(comparable(before), comparable(experiment._state()))

    def test_probe_does_not_write_logs(self):
        experiment = build("uniform")
        experiment.run(2)
        before = copy.deepcopy(experiment.logs)
        for i in range(MODELS):
            for k in range(SOURCES):
                experiment.probe(i, k)
        self.assertEqual(before, experiment.logs)

    def test_probe_rejects_bad_indices(self):
        experiment = build("uniform")
        for bad in ((-1, 0), (MODELS, 0), (0, SOURCES), (0.5, 0)):
            with self.assertRaises(ValueError):
                experiment.probe(*bad)
        for bad in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                experiment.probe(0, 1, blocks=bad)

    def test_more_blocks_trains_distinct_blocks_not_the_same_one(self):
        """blocks=N 必须往前推游标；否则训练量涨了信号却不涨。

        _take_block 只读 self.cursors、不写它，位置得由调用方带着走。忘了这一点
        的话 N 个块全是同一块，加 blocks 这个手段就完全失效——而且不会报错。
        """
        experiment = build("uniform")
        experiment.run(2)
        seen = []
        original = experiment.backend.train_block
        experiment.backend.train_block = lambda i, records: (
            seen.append(tuple(r["id"] for r in records)) or original(i, records))
        result = experiment.probe(0, 1, blocks=3)
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 3, "三个块应当互不相同")
        self.assertEqual(result["blocks"], 3)
        # 逐块训练日志必须留住：能力下降是真退化还是训练发散，只有它能分辨，
        # 而 lora.train_block 的裁剪阈值是 inf——梯度尖峰不会留下任何别的痕迹。
        self.assertEqual(len(result["training"]), 3)
        self.assertEqual([t["completion_tokens"] for t in result["training"]].count(0), 0)
        # 训练 token 是三块之和，迁移率按它归一
        self.assertGreater(result["completion_tokens"], 0)

    def test_probe_still_rolls_back_the_cursor_after_many_blocks(self):
        experiment = build("uniform")
        experiment.run(2)
        before = copy.deepcopy(experiment._state())
        experiment.probe(0, 1, blocks=4)
        self.assertEqual(comparable(before), comparable(experiment._state()))

    def test_shared_baseline_skips_the_before_row_and_resets_the_rng(self):
        """各行 probe 的 before 状态相同，测一次共用；after 前复位到 baseline 记下的随机流。

        各测一次不只是浪费一整行生成，还给每行各自注入一份独立噪声。
        """
        experiment = build("uniform")
        experiment.run(2)
        baseline = experiment.audit_baseline(0)
        self.assertEqual(len(baseline["row"]), SOURCES)
        self.assertEqual([len(d) for d in baseline["detail"]],
                         [len(experiment.banks["audit"][k]) for k in range(SOURCES)])

        rows = []
        original = experiment._audit_row
        experiment._audit_row = lambda i: (rows.append(i) or original(i))
        restored = []
        experiment.backend.set_evaluation_rng = lambda s: restored.append(s)
        result = experiment.probe(0, 2, baseline=baseline)
        # 只测了 after 那一行
        self.assertEqual(len(rows), 1)
        self.assertEqual(result["before"], baseline["row"])
        # after 之前复位到了 baseline 记下的随机流
        self.assertEqual(len(restored), 1)
        self.assertIs(restored[0], baseline["rng"])

    def test_baseline_of_another_expert_is_rejected(self):
        experiment = build("uniform")
        baseline = experiment.audit_baseline(0)
        with self.assertRaises(ValueError):
            experiment.probe(1, 0, baseline=baseline)

    def test_probe_reports_per_item_flips_for_a_paired_test(self):
        """逐条翻转数是配对检验的输入；取了均值就再也拿不回来。"""
        experiment = build("uniform")
        experiment.run(2)
        bank = len(experiment.banks["audit"][0])
        result = experiment.probe(0, 0, blocks=2)
        self.assertEqual(len(result["flips"]), SOURCES)
        for j, flip in enumerate(result["flips"]):
            self.assertLessEqual(flip["gained"] + flip["lost"],
                                 len(experiment.banks["audit"][j]))
            # 净翻转数必须与均值差一致：两者算的是同一件事
            net = (flip["gained"] - flip["lost"]) / len(experiment.banks["audit"][j])
            self.assertAlmostEqual(net, result["after"][j] - result["before"][j], places=9)
        self.assertGreater(bank, 0)


class FixedMethodTests(unittest.TestCase):
    """候选实际收益实验用的 fixed 方法：执行的块必须与给定的逐位相同。"""

    def test_executes_exactly_the_given_blocks(self):
        blocks = [[1, 0, 1, 0], [0, 1, 0, 1]]
        experiment = build("fixed", fixed_blocks=blocks)
        experiment.run(2)
        executed = [e["blocks"] for e in experiment.logs if "blocks" in e and e["round"] >= 1]
        self.assertEqual(executed, [blocks])

    def test_wrong_margins_are_refused(self):
        """边际与预算不符时，integer_counts 会悄悄改成另一份分配——必须报错。"""
        experiment = build("fixed", fixed_blocks=[[2, 0, 0, 0], [0, 0, 1, 1]])
        with self.assertRaises(ValueError):
            experiment.run(2)


if __name__ == "__main__":
    unittest.main()
