"""统一的 LoRA 后端构造，供 test 评价脚本使用，避免各写一份而参数漂移。

sources 必须显式传入：来源数传小时，喂 7 来源 manifest 不会报错，多出来的
来源只会被静默忽略或在评估时用错生成预算——不崩溃，只让结论无效。所以这里
把 sources 变成必须显式传入的参数，并在构造后立即自检。
"""


def build_backend(model_id, *, models, sources, state_dir, max_new_tokens,
                  batch_size, seq_len=16384, rank=32,
                  dtype="bfloat16", base_seed=None):
    """建基座 + M 个 LoRA adapter + LoRABackend，并核对来源数。

    base_seed 不为 None 时，LoRA 初始化在**隔离的 RNG 上下文**里用
    derive(base_seed, "init") 播种，结束后完整还原 CPU 与 CUDA 的 RNG。
    同一 base_seed 得到逐 tensor 相同的初始 adapter；不同 base_seed 不同。
    不播种（None）时 LoRA 初始化取决于进程当时的全局 RNG。test 评价随后用
    resume 整个覆盖 adapter，所以对它无影响；但从零开始训练时**不得**走这条路——
    两次 nominally identical 的运行会拿到完全不同的 lora_A。
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .lora import LoRABackend
    from .seeds import derive, isolated_torch_rng
    from .verifiers import default_verifiers

    if not isinstance(sources, int) or sources < 1:
        raise ValueError("sources 必须显式传入且为正整数——写死它正是本模块要防的事")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
    base.requires_grad_(False)
    config = LoraConfig(r=rank, lora_alpha=2 * rank,
                        target_modules=list(LoRABackend.DEFAULT_TARGETS),
                        task_type="CAUSAL_LM")
    # 全部 M 个 adapter 的构造都在同一个隔离上下文里完成：expert_0 由
    # get_peft_model 建、其余由 add_adapter 建，两者都抽 RNG。放在一个上下文里
    # 才能保证「同 base_seed → 同一组初始专家」，而不是只有第一个专家可复现。
    if base_seed is not None:
        with isolated_torch_rng(derive(base_seed, "init")):
            model = get_peft_model(base, config, adapter_name="expert_0")
            for i in range(1, models):
                model.add_adapter(f"expert_{i}", config)
    else:
        model = get_peft_model(base, config, adapter_name="expert_0")
        for i in range(1, models):
            model.add_adapter(f"expert_{i}", config)
    model.cuda()

    backend = LoRABackend(
        model, tokenizer, models=models, sources=sources,
        state_dir=state_dir, max_length=seq_len,
        max_new_tokens=(max_new_tokens[0] if isinstance(max_new_tokens, (list, tuple))
                        and len(max_new_tokens) == 1 else max_new_tokens),
        batch_size=batch_size,
        # 缺了它，_verify 会在第一次判分时抛「没有为 reference.kind 注册判分器」。
        # 那是会报错的失效，不是静默的——但每个脚本都得记得传，所以放在这里。
        verifiers=default_verifiers())

    if backend.sources != sources:
        raise ValueError(f"backend.sources={backend.sources} 与要求的 {sources} 不符")
    missing = [k for k in range(sources) if k not in backend.max_new_tokens]
    if missing:
        raise ValueError(f"max_new_tokens 缺少来源 {missing}；逐来源的生成预算不全，"
                         "那些来源的评估会直接 KeyError 或用错预算")
    return backend, tokenizer
