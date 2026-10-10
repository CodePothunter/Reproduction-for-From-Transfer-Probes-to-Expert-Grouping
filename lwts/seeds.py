"""从一个 base_seed 派生出互相独立的各路随机种子。

为什么要分家：训练流洗牌用 `random.Random(f"{seed}:{k}")`，它**不控制 LoRA
初始化**。若初始化不另行播种，两次 nominally identical 的热身会拿到不同的
Kaiming `lora_A`，从第 0 步就是不同的模型——而且不报错。由同一个 base_seed
派生出互相独立的各路种子，初始化才可复现，又不与其它随机流相互牵连。

    base_seed
    ├── init_seed             LoRA 初始化，必须跨进程稳定
    ├── train_stream_seed     来源流洗牌
    ├── training_rng_seed     训练侧 torch RNG
    └── evaluation_rng_seed   生成/判分侧 RNG

⚠ **不得使用 Python 内置 `hash()`**：默认带进程级随机盐（PYTHONHASHSEED），
跨进程不可复现，用它等于没播种。这里用 blake2b。
"""
import hashlib
from contextlib import contextmanager

STREAMS = ("init", "train_stream", "training_rng", "evaluation_rng")


def derive(base_seed, stream):
    """base_seed × 流名 → 确定性 32 位种子。跨进程、跨机器稳定。"""
    if stream not in STREAMS:
        raise ValueError(f"未知的随机流 {stream!r}，可选 {STREAMS}")
    digest = hashlib.blake2b(f"{int(base_seed)}:{stream}".encode("utf-8"),
                             digest_size=8).digest()
    return int.from_bytes(digest[:4], "big")


def all_seeds(base_seed):
    return {stream: derive(base_seed, stream) for stream in STREAMS}


@contextmanager
def isolated_torch_rng(seed):
    """在隔离的 RNG 状态里做一件事，结束后**完整还原**。

    初始化不得推进后续训练/评估的 RNG：否则「播种初始化」这个动作本身就会
    改变训练轨迹，等于用一个混淆换掉另一个。CPU 与 CUDA 两侧都要存取。
    """
    import torch
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
