"""两项实现细节的回归测试。

A SFT 训练损失的默认温度是 1：logits 不缩放，即标准的完成 token 负对数似然；
B 整数化每格只取 floor/ceil，精确满足边际，且平方误差最小。
"""
import itertools
import random
import types

import pytest

from lwts.allocation import integer_counts, mirror


# ---------------------------------------------------------------- A
def test_sft_loss_default_temperature_unchanged():
    torch = pytest.importorskip("torch")
    from lwts.lora import LoRABackend

    logits = torch.tensor([[[2.0, 0.5], [2.0, 0.5]]])

    class Tiny:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(logits=logits)

    backend = LoRABackend.__new__(LoRABackend)
    backend.torch, backend.model = torch, Tiny()
    backend._stack = lambda: (None, None)
    loss, _ = backend._supervised_loss(torch.tensor([[0, 0]]), torch.tensor([[-100, 0]]))
    ref = torch.nn.functional.cross_entropy(logits[0, :1], torch.tensor([0]))
    assert torch.allclose(loss, ref)


# ---------------------------------------------------------------- B
def _err(n, t):
    return sum((n[i][k] - t[i][k]) ** 2 for i in range(len(t)) for k in range(len(t[0])))


def test_rounding_is_floor_ceil_and_optimal_small_bruteforce():
    rng = random.Random(7)
    for _ in range(200):
        x = [[rng.random() ** 3 for _ in range(3)] for _ in range(2)]
        q = mirror([[0.0] * 3 for _ in range(2)], x, [0.5, 0.5], [1 / 3] * 3, eta=0.0)
        n = integer_counts(q, [6, 6], [4, 4, 4])
        t = [[12 * v for v in row] for row in q]
        assert [sum(r) for r in n] == [6, 6] and [sum(c) for c in zip(*n)] == [4, 4, 4]
        assert all(int(t[i][k] + 1e-9) <= n[i][k] <= int(t[i][k] + 1e-9) + 1
                   for i in range(2) for k in range(3))
        best = min(_err([[a, b, c], [4 - a, 4 - b, 4 - c]], t)
                   for a, b, c in itertools.product(range(5), repeat=3) if a + b + c == 6)
        assert _err(n, t) <= best + 1e-9


def test_near_uniform_request_rounds_to_uniform():
    """每格目标都在 2 附近、最近整数全为 2 且恰好满足边际时，结果必须全 2。

    只检查行列总量发现不了「同一格被反复补块」这种失真，所以逐格核对。
    """
    rng = random.Random(3)
    x = [[1.0 + 0.15 * (rng.random() - 0.5) for _ in range(7)] for _ in range(4)]
    q = mirror([[0.0] * 7 for _ in range(4)], x, [0.25] * 4, [1 / 7] * 7, eta=0.0)
    n = integer_counts(q, [14] * 4, [8] * 7)
    assert all(v == 2 for row in n for v in row)


def test_exact_integer_targets_are_preserved():
    """连续目标本身就是整数（含 7−ε 的浮点误差）时，结果必须逐格相等。"""
    t = [[0, 0, 7, 7, 0, 0, 0], [7, 7, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 7, 7], [1, 1, 1, 1, 8, 1, 1]]
    q = [[(v - 1e-12 if v else 0.0) / 56 for v in row] for row in t]
    assert integer_counts(q, [14] * 4, [8] * 7) == t
