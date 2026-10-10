"""fork 起点一致性的运行时断言。

是**加载/比对**，不重训、不产生新实验。失败必须报出字段并停止，不是 warning。

为什么需要它：各候选的比较前提是它们从**同一个**热身状态 W 出发。独立训练重跑
做不到逐位相同，所以起点只能靠**从同一 checkpoint 精确分叉**来保证，并在加载后
立即验证。

⚠ 比较的是**共同训练状态**（adapters、optimizer、scheduler、训练 RNG、游标、
  步数）。检查点里其它与训练无关的字段不要求相等。
"""

import json
from pathlib import Path


def tensors_equal(x, y):
    """递归逐张量 torch.equal。不用 allclose：这里要的是「没被改过」，不是「差不多」。"""
    import torch
    if type(x) is not type(y):
        return False
    if torch.is_tensor(x):
        return x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, dict):
        return sorted(x) == sorted(y) and all(tensors_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(tensors_equal(a, b) for a, b in zip(x, y))
    return x == y


def check_fork_consistency(common, arms):
    """各分支加载完成、执行任何新计算**之前**，与共同 checkpoint 比对。

    common / arms[name] 都是 {字段: 值}，字段见下。只比共同训练状态；
    各分支私有的其它字段不参与比较。
    """
    fields = ("adapters", "optimizer", "scheduler", "training_rng", "cursors", "steps")
    failures = {}
    for name, state in arms.items():
        bad = [f for f in fields if not tensors_equal(common.get(f), state.get(f))]
        if bad:
            failures[name] = bad
    if failures:
        raise RuntimeError(
            f"以下分支的起点与共同 checkpoint 不一致：{failures}。"
            f"比较的前提是各分支起点相同；独立训练重跑达不到这一点，"
            f"只能靠从同一 checkpoint 精确分叉，所以这里不一致就必须停。")
    return True


def common_training_state(checkpoint_path):
    """从检查点读出 fork 比对要用的**共同训练状态**。

    后端快照按检查点里记录的路径读取（LoRABackend.snapshot 写入的是绝对路径）。
    """
    import torch
    state = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))["state"]
    blob = torch.load(state["backend"]["path"], map_location="cpu", weights_only=False)
    return {"adapters": blob["adapters"], "optimizer": blob["optimizers"],
            "scheduler": blob["schedulers"], "training_rng": blob["torch_rng"],
            "cursors": state["cursors"], "steps": blob["steps"]}
