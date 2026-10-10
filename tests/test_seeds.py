"""种子派生必须跨进程稳定，且各路互不干扰。

背景：训练流洗牌种子不控制 LoRA 初始化；初始化若不另行播种，两次 nominally
identical 的热身会拿到完全不同的 `lora_A`——而且不报错。
"""
import subprocess
import sys
from pathlib import Path

import pytest

from lwts.seeds import STREAMS, all_seeds, derive


def test_streams_are_distinct():
    """同一个 base_seed 派生出的各路种子必须互不相同，否则初始化与训练同源。"""
    seeds = all_seeds(101)
    assert len(set(seeds.values())) == len(STREAMS), seeds


@pytest.mark.parametrize("stream", STREAMS)
def test_different_base_seeds_differ(stream):
    assert derive(101, stream) != derive(103, stream)


def test_unknown_stream_raises():
    with pytest.raises(ValueError, match="未知的随机流"):
        derive(101, "nonexistent")


def test_stable_across_processes():
    """**关键**：跨进程必须相同。内置 hash() 带进程级随机盐，用它等于没播种。

    这里真的另起一个带不同 PYTHONHASHSEED 的解释器来验，而不是在本进程里
    调两次——本进程调两次对 hash() 也会一致，测不出那个坑。
    """
    import lwts
    root = str(Path(lwts.__file__).resolve().parent.parent)
    # 子进程必须导入同一份 lwts（本包目录里的），否则测的是另一份代码
    code = ("import lwts; from lwts.seeds import all_seeds; "
            "print(lwts.__file__); print(sorted(all_seeds(101).items()))")
    outputs = set()
    for salt in ("0", "1", "random"):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, check=True,
                                env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin",
                                     "PYTHONPATH": root})
        where, seeds = result.stdout.strip().splitlines()
        assert Path(where).resolve().is_relative_to(Path(root)), where
        outputs.add(seeds)
    assert len(outputs) == 1, f"跨进程不稳定：{outputs}"


def test_isolated_rng_restores_state():
    """播种初始化不得推进后续训练/评估的 RNG，否则等于用一个混淆换另一个。"""
    torch = pytest.importorskip("torch")
    from lwts.seeds import isolated_torch_rng

    before = torch.get_rng_state()
    reference = torch.randn(4)
    torch.set_rng_state(before)
    with isolated_torch_rng(derive(101, "init")):
        torch.randn(128)                      # 在隔离上下文里大量抽样
    after = torch.randn(4)
    assert torch.equal(reference, after), "隔离上下文没有还原 RNG 状态"


def test_isolated_rng_is_deterministic():
    torch = pytest.importorskip("torch")
    from lwts.seeds import isolated_torch_rng

    draws = []
    for _ in range(2):
        with isolated_torch_rng(derive(101, "init")):
            draws.append(torch.randn(8))
    assert torch.equal(draws[0], draws[1])
