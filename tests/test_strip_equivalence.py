"""删减后的 lwts 在证据路径上复现删减前代码的轨迹。

tests/fixtures/strip_equivalence_{synthetic,lora}.json 是用删减前的代码在
tests/_strip_scenarios.py 的同一组场景上生成的金标准（只含记录 id 序列与数值）。
本测试用本包的 lwts 重跑这些场景并逐项比对：

  S1  热身 W 的训练块序列、游标、分配、日志、后端完整训练状态、检查点内容
  S2  从 W 分叉的 C0–C3：同上，另加 fork 起点一致性
  S3  W 的检查点改写成较早的布局后再分叉：C0–C3 必须与 S2 的金标准完全相同
  S4  单源迁移探针：基线、每格的训练日志、前后行、迁移、翻转、portfolio_return，
      以及探针结束后的状态（必须回滚）
  S5  diagnostic 恢复候选检查点后按路由评 test 的逐题分数

比对规则：
  · 记录 id 序列、整数（游标、块数、步数、RNG 状态）、字符串、布尔严格相等；
  · 浮点值允许 1e-12 的容差（合成后端是纯 Python 浮点，容差只为跨平台的 libm 差异）；
  · 微型 LoRA 的张量按摘要比对：字节 sha256 相同即逐位一致。浮点张量的字节不同时
    退到和与平方和的 1e-5 相对容差并发出警告——float32 的 CPU 核在不同 CPU / torch
    构建上不保证逐位一致，那不是删减引入的差异；整数张量（RNG 状态）必须逐位一致。
    LoRA 只跑不含生成的 S1–S3。
"""
import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path

import _strip_scenarios as S

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TENSOR_KEYS = {"dtype", "shape", "sha256", "sum", "sumsq"}


def normalized(value):
    """与金标准同样经过一次 JSON：元组变列表、整数键变字符串。"""
    return json.loads(json.dumps(value, sort_keys=True))


class Mismatch(AssertionError):
    pass


def match(expected, actual, tolerance, path="", notes=None):
    if isinstance(expected, dict) and set(expected) == TENSOR_KEYS:
        if not isinstance(actual, dict) or set(actual) != TENSOR_KEYS:
            raise Mismatch(f"{path}: 期望张量摘要")
        if (expected["dtype"], expected["shape"]) != (actual["dtype"], actual["shape"]):
            raise Mismatch(f"{path}: dtype/形状不同")
        if expected["sha256"] == actual["sha256"]:
            return
        rel = tolerance.get("tensor_rel_fallback")
        # 整数张量（RNG 状态等）没有浮点舍入可言，必须逐位一致
        if rel is None or not expected["dtype"].startswith(("torch.float", "torch.bfloat")):
            raise Mismatch(f"{path}: 字节不同")
        # 和可以接近零，按 Cauchy–Schwarz 上界 √(n·Σx²) 定尺度；平方和本身是正的
        size = math.prod(expected["shape"])
        if abs(expected["sumsq"] - actual["sumsq"]) > rel * expected["sumsq"] + 1e-12 or \
                abs(expected["sum"] - actual["sum"]) > rel * math.sqrt(size * expected["sumsq"]) + 1e-12:
            raise Mismatch(f"{path}: 和 {expected['sum']!r} vs {actual['sum']!r}，"
                           f"平方和 {expected['sumsq']!r} vs {actual['sumsq']!r}")
        if notes is not None:
            notes.append(path)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or sorted(expected) != sorted(actual):
            raise Mismatch(f"{path}: 键不同 {sorted(set(expected) ^ set(actual or {}))[:5]}")
        for key in expected:
            match(expected[key], actual[key], tolerance, f"{path}.{key}", notes)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise Mismatch(f"{path}: 长度不同")
        for i, (a, b) in enumerate(zip(expected, actual)):
            match(a, b, tolerance, f"{path}[{i}]", notes)
        return
    if isinstance(expected, float):
        if type(actual) is not float or not math.isclose(
                expected, actual, rel_tol=tolerance["float_rel"], abs_tol=tolerance["float_abs"]):
            raise Mismatch(f"{path}: {expected!r} vs {actual!r}")
        return
    if type(expected) is not type(actual) or expected != actual:
        raise Mismatch(f"{path}: {expected!r} vs {actual!r}")


class MatcherTests(unittest.TestCase):
    """比对器本身要有分辨力，否则「全部通过」不说明任何事。"""

    TOL = {"float_abs": 1e-12, "float_rel": 1e-12, "tensor_rel_fallback": 1e-5}

    def tensor(self, sha, total, squares, dtype="torch.float32"):
        return {"dtype": dtype, "shape": [4], "sha256": sha, "sum": total, "sumsq": squares}

    def test_ids_ints_and_floats(self):
        match({"t": [[0, 1, ["a", "b"]]], "x": 0.5}, {"t": [[0, 1, ["a", "b"]]], "x": 0.5}, self.TOL)
        for actual in ({"t": [[0, 1, ["b", "a"]]], "x": 0.5}, {"t": [[1, 1, ["a", "b"]]], "x": 0.5},
                       {"t": [[0, 1, ["a", "b"]]], "x": 0.5 + 1e-9}, {"t": [[0, 1, ["a"]]], "x": 0.5}):
            with self.assertRaises(Mismatch, msg=repr(actual)):
                match({"t": [[0, 1, ["a", "b"]]], "x": 0.5}, actual, self.TOL)
        match({"x": 0.5}, {"x": 0.5 + 1e-13}, self.TOL)

    def test_tensor_summaries(self):
        expected = self.tensor("aa", 0.01, 2.0)
        notes = []
        match(expected, self.tensor("aa", 0.01, 2.0), self.TOL, notes=notes)
        self.assertEqual(notes, [])
        match(expected, self.tensor("bb", 0.01 + 1e-6, 2.0 + 1e-6), self.TOL, notes=notes)
        self.assertEqual(len(notes), 1, "字节不同但在容差内：要留下记录")
        with self.assertRaises(Mismatch):
            match(expected, self.tensor("bb", 0.5, 2.0), self.TOL)
        with self.assertRaises(Mismatch):
            match(self.tensor("aa", 10, 30, "torch.uint8"), self.tensor("bb", 10, 30, "torch.uint8"),
                  self.TOL)


class Base:
    """子类设置 KIND 与 setUpClass 里的 make/manifest/block_tokens。"""

    KIND = None

    @classmethod
    def load_fixture(cls):
        cls.fixture = json.loads((FIXTURES / f"strip_equivalence_{cls.KIND}.json")
                                 .read_text(encoding="utf-8"))
        cls.tolerance = cls.fixture["tolerance"]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, expected, actual, path):
        notes = []
        try:
            match(expected, normalized(actual), self.tolerance, path, notes)
        except Mismatch as error:
            self.fail(f"与删减前代码的金标准不一致：{error}")
        if notes:
            warnings.warn(f"{len(notes)} 个张量不是逐位一致（在容差内），首个：{notes[0]}")

    @classmethod
    def warmup(cls):
        if not hasattr(cls, "_warmup"):
            cls._warmup = S.warmup(cls.make, cls.manifest, cls.root / "strip", cls.block_tokens)
        return cls._warmup

    def test_s1_warmup(self):
        view, _ = self.warmup()
        self.assertEqual(len(view["trace"]), S.BLOCKS_PER_INTERVAL)
        self.check(self.fixture["S1"], view, "S1")

    def test_s2_candidates_forked_from_warmup(self):
        _, w_path = self.warmup()
        out = S.candidates(self.make, self.manifest, w_path, self.root / "strip-forks",
                           self.block_tokens)
        for name, view in out.items():
            self.assertTrue(view["fork_consistent"], name)
            self.assertEqual(view["resumed_round"], 1, name)
        self.check(self.fixture["S2"], out, "S2")

    def test_s3_fork_from_older_checkpoint_layout(self):
        _, w_path = self.warmup()
        older = S.older_layout(w_path, self.root / "older" / "checkpoint.json")
        out = S.candidates(self.make, self.manifest, older, self.root / "older-forks",
                           self.block_tokens)
        # 检查点视图只含与训练有关的字段：额外的键与日志条目不出现在比对里
        self.check(self.fixture["S2"], out, "S3")


class SyntheticEquivalence(Base, unittest.TestCase):
    KIND = "synthetic"

    @classmethod
    def setUpClass(cls):
        cls.load_fixture()
        cls.make = staticmethod(S.synthetic_factory())
        cls.manifest = S.synthetic_manifest()
        cls.block_tokens = S.SYNTHETIC_BLOCK_TOKENS

    def test_s4_probes(self):
        out = S.probes(self.make, self.manifest, self.root, self.block_tokens)
        self.assertEqual(out["state_before"], out["state_after"], "探针结束后状态必须完全回滚")
        self.check(self.fixture["S4"], out, "S4")

    def test_s5_test_evaluation_after_diagnostic_resume(self):
        _, w_path = self.warmup()
        S.candidates(self.make, self.manifest, w_path, self.root / "eval-forks", self.block_tokens)
        out = S.evaluation(self.make, self.manifest, self.root / "eval-forks", self.root,
                           self.block_tokens)
        self.check(self.fixture["S5"], out, "S5")

    def test_fixture_is_not_trivial(self):
        """金标准本身要有分辨力：探针有翻转；各候选的块序列、参数与 test 分数各不相同。

        （各候选的游标相同是对的：每个来源都训 8 块，取的是同一段数据，只是分给
        不同的专家。）
        """
        flips = [f for cell in self.fixture["S4"]["probes"] for f in cell["flips"]]
        self.assertGreater(sum(f["gained"] + f["lost"] for f in flips), 0)
        for field in (lambda v: v["trace"], lambda v: v["state"]["backend"]["params"]):
            distinct = {json.dumps(field(v)) for v in self.fixture["S2"].values()}
            self.assertEqual(len(distinct), len(S.CANDIDATES))
        rows = {json.dumps(v["test_rows"]) for v in self.fixture["S5"].values()}
        self.assertEqual(len(rows), len(S.CANDIDATES))


try:
    import peft  # noqa: F401
    import tokenizers  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TRAIN = True
except ImportError:                                    # pragma: no cover - 环境相关
    HAVE_TRAIN = False


@unittest.skipUnless(HAVE_TRAIN, "需要 train extra：uv sync --extra train")
class LoRAEquivalence(Base, unittest.TestCase):
    KIND = "lora"

    @classmethod
    def setUpClass(cls):
        cls.load_fixture()
        tokenizer = S.byte_tokenizer()
        cls.make = staticmethod(S.lora_factory(tokenizer))
        cls.manifest = S.lora_manifest(tokenizer)
        cls.block_tokens = S.LORA_BLOCK_TOKENS

    def check(self, expected, actual, path):
        # eval_rng 不在金标准里：删减前的 W 在区间内的只读测量会推进它（采样用的
        # CPU 随机流），它不属于训练状态。
        super().check(expected, _without_eval_rng(normalized(actual)), path)


def _without_eval_rng(view):
    if isinstance(view, dict):
        return {k: _without_eval_rng(v) for k, v in view.items() if k != "eval_rng"}
    if isinstance(view, list):
        return [_without_eval_rng(v) for v in view]
    return view


if __name__ == "__main__":
    unittest.main()
