"""数据集接入层：公共流程在 common.py，每个数据集一个模块。

    from lwts.datasets import prepare_records, split_records, validate_manifest
    from lwts.datasets import DEEPMATH, OPENCODEINSTRUCT, MBPPPLUS

新增数据集只需要写一个模块（parse + SPEC + load），然后在这里注册。
"""
from . import deepmath, mbppplus, opencodeinstruct
from .common import (SPLITS, Excluded, Parsed, Record, Spec, Target, problem_key,
                     register, split_records, validate_manifest, write_report)
from .common import download_hf as _download_hf
from .common import prepare_records as _prepare_records
from .deepmath import DEEPMATH
from .mbppplus import MBPPPLUS
from .opencodeinstruct import OPENCODEINSTRUCT

for _module in (deepmath, opencodeinstruct, mbppplus):
    register(_module.SPEC)

MODULES = {"deepmath": deepmath, "opencodeinstruct": opencodeinstruct,
           "mbppplus": mbppplus}
DATASETS = tuple(MODULES)


def prepare_records(raw_records, tokenizer, *, dataset=DEEPMATH, **kwargs):
    """见 common.prepare_records；dataset 默认 DeepMath，统一输入可传自己的标签。"""
    return _prepare_records(raw_records, tokenizer, dataset=dataset, **kwargs)


def load_data(dataset=DEEPMATH, *, revision="main", **kwargs):
    """按数据集标识调用对应模块的 load()；返回 HF DatasetDict。"""
    for module in MODULES.values():
        if module.SPEC.dataset == dataset:
            return module.load(revision=revision, **kwargs)
    from .common import load_hf
    return load_hf(dataset, revision=revision, **kwargs)


def download(name, root="data/raw", *, revision="main"):
    """统一下载入口：name 取 DATASETS 之一，走 HuggingFace。"""
    if name not in MODULES:
        raise ValueError(f"未知数据集 {name}；可选 {DATASETS}")
    module = MODULES[name]
    return _download_hf(module.SPEC.label, module.load, str(root) + "/hf", revision=revision)


__all__ = ["DATASETS", "DEEPMATH", "MBPPPLUS", "MODULES", "OPENCODEINSTRUCT",
           "SPLITS", "Excluded", "Parsed", "Record", "Spec", "Target", "download",
           "load_data", "prepare_records", "problem_key", "register", "split_records",
           "validate_manifest", "write_report"]
