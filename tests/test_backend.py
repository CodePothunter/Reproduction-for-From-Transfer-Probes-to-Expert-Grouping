"""训练后端接口与 CPU 替身的测试。

重点：
  - 保存恢复后的继续训练与不中断轨迹一致
  - 探针出错也能恢复参数、优化器、随机状态和 scheduler
回滚必须覆盖 adapters / optimizer buffers / scheduler / RNG，所以这里逐项验证
「只恢复参数」是不够的。
"""
import tempfile
import unittest
from pathlib import Path

from lwts.backend import SyntheticBackend, TrainingBackend
from lwts.storage import load, save


def train_records(source, count=4, prefix="t"):
    return [{"id": f"{prefix}-{source}-{i}", "source": source, "split": "train",
             "tokens": 300, "completion_tokens": 120} for i in range(count)]


def eval_records(source, count=200, split="audit"):
    return [{"id": f"e-{source}-{i}", "source": source, "split": split,
             "tokens": 200, "completion_tokens": 0} for i in range(count)]


class InterfaceTests(unittest.TestCase):
    def test_abstract_base_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            TrainingBackend()

    def test_partial_implementation_is_rejected(self):
        class Partial(TrainingBackend):
            def train_block(self, model_id, records):
                return {}

        with self.assertRaises(TypeError):
            Partial()

    def test_synthetic_is_a_backend(self):
        self.assertIsInstance(SyntheticBackend(2, 4), TrainingBackend)


class TrainBlockTests(unittest.TestCase):
    def setUp(self):
        self.backend = SyntheticBackend(2, 4, seed=1)

    def test_reports_token_counts(self):
        log = self.backend.train_block(0, train_records(2, count=3))
        self.assertEqual(log["processed_tokens"], 900)
        self.assertEqual(log["completion_tokens"], 360)
        self.assertEqual(log["source"], 2)
        self.assertEqual(log["step"], 1)

    def test_rejects_non_train_split(self):
        rows = train_records(0)
        rows[1]["split"] = "audit"
        with self.assertRaises(ValueError):
            self.backend.train_block(0, rows)

    def test_requires_source_pure_block(self):
        """混合来源的块无法把收益归因到某个 cell。"""
        with self.assertRaises(ValueError):
            self.backend.train_block(0, train_records(0, 2) + train_records(1, 2))

    def test_rejects_bad_model_and_empty(self):
        for bad in (-1, 2, 1.0):
            with self.assertRaises(ValueError):
                self.backend.train_block(bad, train_records(0))
        with self.assertRaises(ValueError):
            self.backend.train_block(0, [])

    def test_identical_initial_adapters(self):
        """所有模型必须从同一初始 adapter 出发。"""
        backend = SyntheticBackend(3, 2)
        self.assertEqual(backend.params[0], backend.params[1])
        self.assertEqual(backend.params[1], backend.params[2])

    def test_learning_rate_decays(self):
        backend = SyntheticBackend(1, 1, total_steps=10, learning_rate=0.1)
        self.assertAlmostEqual(backend.scheduled_rate(step=0), 0.1)
        self.assertAlmostEqual(backend.scheduled_rate(step=5), 0.05)
        self.assertAlmostEqual(backend.scheduled_rate(step=10), 0.0)
        self.assertAlmostEqual(backend.scheduled_rate(step=99), 0.0)

    def test_each_model_has_its_own_scheduler(self):
        """共享 scheduler 会让先训练的模型拿到更高学习率，产生与来源无关的伪优势。"""
        backend = SyntheticBackend(2, 1, total_steps=100, learning_rate=0.1)
        backend.train_block(0, train_records(0))
        backend.train_block(0, train_records(0))
        self.assertEqual(backend.steps, [2, 0])
        self.assertAlmostEqual(backend.scheduled_rate(1), 0.1, msg="模型 1 不应被模型 0 的训练拖低")
        self.assertLess(backend.scheduled_rate(0), backend.scheduled_rate(1))

    def test_symmetric_plan_keeps_models_identical(self):
        """对称计划 + 无噪声下两个模型必须逐位相同：没有系统性的模型顺序优势。"""
        backend = SyntheticBackend(2, 3, noise=0.0)
        for _ in range(15):
            for model_id in range(2):
                for source in range(3):
                    backend.train_block(model_id, train_records(source))
        self.assertEqual(backend.params[0], backend.params[1])
        self.assertEqual(backend.velocity[0], backend.velocity[1])


class TransferTests(unittest.TestCase):
    """跨来源迁移是实验能区分各分配方法的前提。"""

    def backend(self):
        # 来源 0 与 1 正相关，与 2 反相关
        return SyntheticBackend(1, 3, noise=0.0, response=2e-3,
                                directions=[[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]])

    def test_training_transfers_to_correlated_source(self):
        backend = self.backend()
        before = [backend.true_competence(0, k) for k in range(3)]
        for _ in range(20):
            backend.train_block(0, train_records(0))
        after = [backend.true_competence(0, k) for k in range(3)]
        self.assertGreater(after[0], before[0], "自身来源应提升")
        self.assertGreater(after[1], before[1], "正相关来源应被带动")
        self.assertLess(after[2], before[2], "反相关来源应被拖累")

    def test_response_has_diminishing_returns(self):
        """能力越高，同样 token 带来的提升越小——响应有曲率。"""
        backend = self.backend()
        gains = []
        for _ in range(3):
            before = backend.true_competence(0, 0)
            for _ in range(15):
                backend.train_block(0, train_records(0))
            gains.append(backend.true_competence(0, 0) - before)
        self.assertLess(gains[-1], gains[0])


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.backend = SyntheticBackend(2, 3, seed=3)

    def test_scores_are_binary_and_average_to_competence(self):
        rows = eval_records(1, count=4000)
        scores = self.backend.evaluate(0, rows)
        self.assertEqual(len(scores), len(rows))
        self.assertTrue(set(scores) <= {0.0, 1.0})
        mean = sum(scores) / len(scores)
        self.assertAlmostEqual(mean, self.backend.true_competence(0, 1), delta=0.02)

    def test_does_not_mutate_any_state(self):
        """评估不能更新模型或优化器，也不能消耗训练随机数。"""
        self.backend.train_block(0, train_records(0))
        before = self.backend.snapshot()
        self.backend.evaluate(0, eval_records(0, count=500))
        self.backend.evaluate(1, eval_records(2, count=500))
        self.assertEqual(before, self.backend.snapshot())

    def test_is_repeatable(self):
        rows = eval_records(0, count=100)
        self.assertEqual(self.backend.evaluate(0, rows), self.backend.evaluate(0, rows))

    def test_paired_measurement_changes_only_through_competence(self):
        """同一批样本的判定阈值固定，配对差分不被重抽样噪声淹没。"""
        rows = eval_records(0, count=500)
        before = self.backend.evaluate(0, rows)
        for _ in range(30):
            self.backend.train_block(0, train_records(0))
        after = self.backend.evaluate(0, rows)
        self.assertGreater(sum(after), sum(before))
        # 阈值不变意味着分数只会从 0 翻到 1，不会有样本反向翻转
        self.assertFalse(any(a < b for a, b in zip(after, before)))

    def test_rejects_train_split(self):
        with self.assertRaises(ValueError):
            self.backend.evaluate(0, train_records(0))


class SnapshotRestoreTests(unittest.TestCase):
    def run_blocks(self, backend, plan):
        for model_id, source in plan:
            backend.train_block(model_id, train_records(source))

    def test_resume_matches_uninterrupted_trajectory(self):
        """保存恢复后的继续训练与不中断轨迹一致。"""
        plan_a = [(0, 0), (1, 1), (0, 2), (1, 3), (0, 1)]
        plan_b = [(1, 2), (0, 3), (1, 0), (0, 0), (1, 1)]

        straight = SyntheticBackend(2, 4, seed=7)
        self.run_blocks(straight, plan_a + plan_b)

        interrupted = SyntheticBackend(2, 4, seed=7)
        self.run_blocks(interrupted, plan_a)
        saved = interrupted.snapshot()
        self.run_blocks(interrupted, plan_b)

        self.assertEqual(straight.snapshot(), interrupted.snapshot())

        # 从快照重放后半段，必须回到同一个状态
        interrupted.restore(saved)
        self.run_blocks(interrupted, plan_b)
        self.assertEqual(straight.snapshot(), interrupted.snapshot())

    def test_probe_rollback_restores_everything(self):
        """探针：临时更新后回滚，全部状态必须与探针前逐位相同。"""
        backend = SyntheticBackend(2, 4, seed=11)
        self.run_blocks(backend, [(0, 0), (1, 1), (0, 2)])
        saved = backend.snapshot()
        for _ in range(5):
            backend.train_block(1, train_records(3))
        self.assertNotEqual(saved, backend.snapshot())
        backend.restore(saved)
        self.assertEqual(saved, backend.snapshot())

    def test_rollback_survives_an_exception(self):
        """探针中途出错也要能恢复。"""
        backend = SyntheticBackend(2, 4, seed=13)
        self.run_blocks(backend, [(0, 0), (1, 3)])
        saved = backend.snapshot()
        try:
            backend.train_block(0, train_records(1))
            raise RuntimeError("探针内部失败")
        except RuntimeError:
            backend.restore(saved)
        self.assertEqual(saved, backend.snapshot())

    def test_params_only_restore_is_not_enough(self):
        """证明上面的测试真的在检查完整状态：只还原参数会走出不同轨迹。"""
        plan = [(0, 0), (0, 1), (0, 2)]
        reference = SyntheticBackend(1, 4, seed=17)
        self.run_blocks(reference, plan)
        saved = reference.snapshot()
        self.run_blocks(reference, plan)
        expected = reference.snapshot()

        partial = SyntheticBackend(1, 4, seed=17)
        self.run_blocks(partial, plan)
        self.run_blocks(partial, plan)
        # 只把参数搬回去，动量/step/RNG 保持"跑过两轮"的状态
        partial.params = [row[:] for row in saved["params"]]
        self.run_blocks(partial, plan)
        self.assertNotEqual(expected, partial.snapshot(),
                            "只恢复参数却得到相同轨迹，说明测试没有区分能力")

    def test_snapshot_is_json_serializable(self):
        """快照要能直接交给 storage.save，CPU 上端到端验证中断恢复。"""
        backend = SyntheticBackend(2, 3, seed=19)
        self.run_blocks(backend, [(0, 0), (1, 2)])
        saved = backend.snapshot()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "backend.json"
            save(path, saved)
            reloaded = load(path)
        fresh = SyntheticBackend(2, 3, seed=99)
        fresh.restore(reloaded)
        self.assertEqual(fresh.snapshot(), saved)

    def test_rejects_foreign_or_malformed_state(self):
        backend = SyntheticBackend(2, 3)
        good = backend.snapshot()
        for bad in ({}, {"format": "other"},
                    {**good, "params": [[0.0]]},
                    {**good, "steps": [-1, 0]},
                    {**good, "steps": [0]},
                    {**good, "rng": [1, 2]}):
            with self.assertRaises(ValueError):
                backend.restore(bad)


if __name__ == "__main__":
    unittest.main()
