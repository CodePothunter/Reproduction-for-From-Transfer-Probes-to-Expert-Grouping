"""fork 起点一致性断言的正负对照：起点相同通过，任一共同字段不同必须失败。"""
import copy

import pytest

torch = pytest.importorskip("torch")

from lwts.guards import check_fork_consistency  # noqa: E402

ARMS = ("C0", "C1", "C2")


def common_state():
    return {"adapters": [{"lora_A": torch.arange(6.0)}],
            "optimizer": [{"step": 7}], "scheduler": [{"last_epoch": 7}],
            "training_rng": torch.ByteTensor([4, 5]), "cursors": {0: 23},
            "steps": [7, 7, 7, 7]}


def test_identical_forks_pass():
    common = common_state()
    arms = {n: copy.deepcopy(common) for n in ARMS}
    assert check_fork_consistency(common, arms)


def test_private_state_may_differ():
    """检查点里与训练无关的字段（日志、成本账等）不同是允许的，不得据此判失败。"""
    common = common_state()
    arms = {}
    for i, n in enumerate(ARMS):
        state = copy.deepcopy(common)
        state["logs"] = {f"cell{i}": [i]}           # 私有字段，不参与比较
        state["cost"] = {"generated": i}
        arms[n] = state
    assert check_fork_consistency(common, arms)


@pytest.mark.parametrize("field", ["adapters", "optimizer", "scheduler",
                                   "training_rng", "cursors", "steps"])
def test_diverged_fork_fails(field):
    common = common_state()
    arms = {n: copy.deepcopy(common) for n in ARMS}
    arms["C0"][field] = ([{"lora_A": torch.zeros(6)}] if field == "adapters"
                         else [{"step": 99}] if field in ("optimizer", "scheduler")
                         else torch.ByteTensor([9, 9]) if field == "training_rng"
                         else {0: 99} if field == "cursors" else [9, 9, 9, 9])
    with pytest.raises(RuntimeError, match="起点与共同 checkpoint 不一致"):
        check_fork_consistency(common, arms)
