"""真实 LoRA 后端的 CPU 测试：用微型模型验证机制，不需要 GPU。

需要 train extra（torch + peft），没装则整体跳过；还需要 Qwen3.5-2B-Base 的
tokenizer（HuggingFace 缓存里有即可离线运行），取不到则整体跳过。

覆盖「写错了不会报错但结果无效」的地方：loss mask 与数据层一致、逐来源生成
预算、每专家独立 scheduler、评估不扰动训练 RNG、快照恢复。
"""
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from lwts.lora import LoRABackend
    HAVE_TRAIN = True
except ImportError:                                    # pragma: no cover - 环境相关
    HAVE_TRAIN = False

from lwts.datasets import prepare_records

MODEL = "Qwen/Qwen3.5-2B-Base"


def tiny(tokenizer, models=2, **kwargs):
    """两层微型 Llama + M 个 LoRA adapter；结构与真实路径相同，规模可在 CPU 上跑。"""
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    base = LlamaForCausalLM(config)
    base.requires_grad_(False)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lora, adapter_name="expert_0")
    for i in range(1, models):
        model.add_adapter(f"expert_{i}", lora)
    return LoRABackend(model, tokenizer, models=models, sources=2, **kwargs)


@unittest.skipUnless(HAVE_TRAIN, "需要 train extra：uv sync --extra train")
class LoRABackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
        except OSError as error:                       # pragma: no cover - 环境相关
            raise unittest.SkipTest(f"取不到 {MODEL} 的 tokenizer：{error}") from error

    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.backend = tiny(self.tokenizer, state_dir=self.tmp.name, max_new_tokens=4,
                            seed=0,
                            verifiers={"math": lambda answer, ref: float(ref["answer"] in answer)})

    def record(self, source=0, split="train", answer="2", completion="the answer is 2"):
        """用真实 tokenizer 走一遍数据层，拿到 manifest 口径的 completion_tokens。"""
        rows = prepare_records(
            [{"prompt": f"1+1=?{source}\n\nSolution:\n", "problem": f"q{source}",
              "source": source, "reference": {"kind": "math", "answer": answer},
              "completion": completion}],
            self.tokenizer, dataset="tiny")
        return dict(rows[0], split=split)

    # --- 坐标与 loss mask ---------------------------------------------------

    def test_adapter_coordinates_are_stable(self):
        first = self.backend.parameters(0)
        self.assertEqual(len(first), len(self.backend.parameters(1)))
        self.assertEqual(first, self.backend.parameters(0), "同一状态两次读取必须一致")

    def test_array_path_matches_the_list_path(self):
        """走 list[float] 每次要造 1480 万个 Python float 对象；数组路径必须给出
        完全相同的坐标。"""
        array = self.backend.parameters_array(0)
        self.assertEqual(array.tolist(), self.backend.parameters(0))
        self.assertEqual(array.ndim, 1)

    def test_supervised_count_must_match_manifest(self):
        """训练端与整备端的模板或 mask 规则不一致时必须报错，而不是静默训错。"""
        record = self.record()
        self.assertGreater(record["completion_tokens"], 0)
        self.backend.train_block(0, [record])          # 一致时通过
        bad = dict(record, completion_tokens=record["completion_tokens"] + 1)
        with self.assertRaises(ValueError):
            self.backend.train_block(0, [bad])

    def test_prompt_tokens_are_not_supervised(self):
        """把 completion 换长，监督数应只随 completion 变，prompt 部分不计入。"""
        short = self.record(completion="2")
        long = self.record(completion="the answer is definitely 2")
        self.assertLess(short["completion_tokens"], long["completion_tokens"])
        self.assertEqual(short["tokens"] - short["completion_tokens"],
                         long["tokens"] - long["completion_tokens"],
                         "prompt 部分的 token 数应当相同")

    # --- 训练 ---------------------------------------------------------------

    def test_train_block_moves_only_the_active_adapter(self):
        other_before = self.backend.parameters(1)
        before = self.backend.parameters(0)
        self.backend.train_block(0, [self.record()])
        self.assertNotEqual(before, self.backend.parameters(0))
        self.assertEqual(other_before, self.backend.parameters(1), "另一专家不应被改动")

    def test_each_expert_has_its_own_scheduler(self):
        self.backend.train_block(0, [self.record()])
        self.backend.train_block(0, [self.record()])
        self.assertEqual(self.backend.steps, [2, 0])
        rates = [s.get_last_lr()[0] for s in self.backend.schedulers]
        self.assertLess(rates[0], rates[1], "专家 1 不应被专家 0 的训练拖低学习率")

    def test_rejects_non_train_and_mixed_source(self):
        with self.assertRaises(ValueError):
            self.backend.train_block(0, [self.record(split="audit")])
        with self.assertRaises(ValueError):
            self.backend.train_block(0, [self.record(source=0), self.record(source=1)])
        with self.assertRaises(ValueError):
            self.backend.train_block(0, [])

    def test_loss_is_near_uniform_for_an_untrained_model(self):
        """随机初始化模型的 loss 应接近 ln(vocab)，偏离说明 mask 或对齐有问题。"""
        import math
        log = self.backend.train_block(0, [self.record()])
        self.assertAlmostEqual(log["loss"], math.log(len(self.tokenizer)), delta=0.5)

    # --- 评估 ---------------------------------------------------------------

    def test_evaluate_rejects_train_split_and_keeps_state(self):
        with self.assertRaises(ValueError):
            self.backend.evaluate(0, [self.record()])
        before = self.backend.parameters(0)
        scores = self.backend.evaluate(0, [self.record(split="audit")])
        self.assertEqual(len(scores), 1)
        self.assertTrue(0.0 <= scores[0] <= 1.0)
        self.assertEqual(before, self.backend.parameters(0), "评估不能改动参数")

    def test_generation_budget_is_per_source(self):
        """逐来源预算：数学侧需要长生成，代码侧不需要，同一个上限两头都错。"""
        from lwts.lora import _budgets
        self.assertEqual(_budgets(512, 4), {0: 512, 1: 512, 2: 512, 3: 512})
        self.assertEqual(_budgets([8192, 8192, 1024, 1024], 4),
                         {0: 8192, 1: 8192, 2: 1024, 3: 1024})
        self.assertEqual(_budgets({0: 7, 1: 8}, 2), {0: 7, 1: 8})
        with self.assertRaises(ValueError):
            _budgets([1, 2], 4)                 # 缺来源 2、3
        with self.assertRaises(ValueError):
            _budgets([1, 0, 1, 1], 4)           # 非正

        backend = tiny(self.tokenizer, state_dir=self.tmp.name,
                       max_new_tokens=[3, 3, 9, 9], seed=0,
                       verifiers={"math": lambda answer, ref: 0.0})
        self.assertEqual(backend.max_new_tokens, {0: 3, 1: 3, 2: 9, 3: 9})

    def test_evaluate_rejects_mixed_sources(self):
        """预算逐来源取值，混源就无法确定该用哪一个。

        批量解码是同步推进的，混源还会让短来源按长来源的上限陪跑。调用方本来
        就是按来源取 bank 的，这里把那个隐含前提变成显式检查。
        """
        rows = [self.record(source=0, split="audit"),
                self.record(source=1, split="audit")]
        with self.assertRaises(ValueError) as caught:
            self.backend.evaluate(0, rows)
        self.assertIn("同源", str(caught.exception))
        # 同源仍然正常
        self.assertEqual(len(self.backend.evaluate(0, rows[:1])), 1)

    def test_generation_does_not_disturb_training_rng(self):
        """评估生成若消耗全局（CPU）RNG，恢复检查点后的训练轨迹将不可复现，
        而且这一点不会以任何形式报错。"""
        before = torch.get_rng_state()
        self.backend.evaluate(0, [self.record(split="audit")])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_greedy_is_a_separate_deterministic_protocol(self):
        stochastic = self.backend.evaluate(0, [self.record(split="audit")])
        greedy = self.backend.evaluate(0, [self.record(split="audit")],
                                       decoding="greedy")
        self.assertEqual(len(stochastic), len(greedy))
        again = self.backend.evaluate(0, [self.record(split="audit")],
                                      decoding="greedy")
        self.assertEqual(greedy, again, "greedy 必须是确定性的")
        with self.assertRaises(ValueError):
            self.backend.evaluate(0, [self.record(split="audit")], decoding="nucleus")

    def test_missing_verifier_is_rejected(self):
        self.backend.verifiers = {}
        with self.assertRaises(ValueError):
            self.backend.evaluate(0, [self.record(split="audit")])

    # --- 状态 ---------------------------------------------------------------

    def test_snapshot_restore_round_trip(self):
        self.backend.train_block(0, [self.record()])
        saved = self.backend.snapshot()
        self.assertEqual(set(saved) >= {"format", "path", "sha256", "steps"}, True)
        expected = self.backend.parameters(0)

        self.backend.train_block(0, [self.record()])
        self.assertNotEqual(expected, self.backend.parameters(0))
        self.backend.restore(saved)
        self.assertEqual(expected, self.backend.parameters(0))
        self.assertEqual(self.backend.steps, saved["steps"])

    def test_restore_rejects_tampered_file(self):
        saved = self.backend.snapshot()
        Path(saved["path"]).write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            self.backend.restore(saved)
        with self.assertRaises(ValueError):
            self.backend.restore({"format": "other"})

    def test_snapshot_metadata_is_json_safe(self):
        import json
        json.dumps(self.backend.snapshot())         # 不抛异常即可交给 storage.save

    def test_discard_frees_the_file(self):
        saved = self.backend.snapshot()
        self.assertTrue(Path(saved["path"]).exists())
        self.backend.discard(saved)
        self.assertFalse(Path(saved["path"]).exists())
        self.backend.discard(saved)                 # 重复调用不应报错
        self.backend.discard({"format": "other"})   # 不是本后端的快照，不碰

    def test_discarded_name_is_never_reused(self):
        """序号必须单调：否则删掉中间一份之后，新快照会覆盖仍被检查点引用的名字。"""
        first, second, third = (self.backend.snapshot() for _ in range(3))
        self.backend.discard(second)
        fourth = self.backend.snapshot()
        paths = {s["path"] for s in (first, third)}
        self.assertNotIn(fourth["path"], paths)
        self.assertTrue(all(Path(p).exists() for p in paths), "存活的快照不能被覆盖")


if __name__ == "__main__":
    unittest.main()
