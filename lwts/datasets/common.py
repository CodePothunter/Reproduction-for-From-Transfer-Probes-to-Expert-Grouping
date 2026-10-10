"""数据契约与公共流程：所有数据集共用这一份。

各数据集只负责把原始字段映射成 Parsed（见同目录 deepmath.py /
opencodeinstruct.py / mbppplus.py），之后的题目去重、目标选择、token 计数、
切分和清单审计都走这里，只写一遍。

SPECS 由包的 __init__.py 注册，请从 lwts.datasets 导入公开函数。
"""
import ast
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple, TypedDict

from ..storage import canonical

_SURROGATE = re.compile(r"[\ud800-\udfff]")

# 五路切分的名字写在 manifest 的每条记录里，不能改名。本包的流水线只读取
# train（训练流）、audit（探针）与 test（候选评价）；其余两路只参与切分与校验。
SPLITS = ("train", "feedback", "route", "audit", "test")
ORIGINAL_SPLITS = ("train", "test", "dev", "validation")
DEFAULT_SOURCES = {"Algebra": 0, "Number Theory": 1}


class Record(TypedDict):
    id: str
    group_id: str
    source: int
    prompt: str
    completion: str
    split: str
    tokens: int
    completion_tokens: int
    reference: dict
    metadata: dict


class Excluded(Exception):
    """按规则排除这一条；计入报告并继续处理其余记录。

    与 ValueError 的分工：ValueError 表示调用方配置或输入契约有误，整批中止。
    """

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Context(NamedTuple):
    """adapter 需要的调用级参数，与单条记录无关。"""
    topic_sources: dict
    evaluation: bool
    schemas: dict | None
    database_paths: dict | None


class Parsed(NamedTuple):
    """adapter 的统一产出；后续流程只认这六个字段。"""
    source: int
    identity: str            # 用于题目分组的文本，去重前先做规范化
    prompt: str
    reference: dict
    targets: list            # [(目标名, 目标文本)]，按优先级排列
    topic: object            # 原样写入 metadata，各数据集含义不同


class Target(NamedTuple):
    name: str
    text: str
    tokens: int
    completion_tokens: int


class Spec(NamedTuple):
    """一个数据集的规格；把散落在流程里的数据集条件集中到这里声明。"""
    dataset: str                    # 数据集标识（HF id）
    label: str                      # 报错信息里的人类名字
    parse: object = None            # adapter 函数；None 表示走统一输入
    default_sources: dict = DEFAULT_SOURCES
    evaluation_only: bool = False   # 只能作为外部 test（MBPP+）
    external: bool = False          # 写进 metadata.external
    official_test_only: bool = False    # 最终 test 必须来自官方 test（本包的数据集都不设）
    unified_when_prompt: bool = False   # 记录已带 prompt 字段时按统一输入处理


SPECS = {}


def register(spec):
    SPECS[spec.dataset] = spec
    return spec


def spec_for(dataset):
    """未注册的 dataset 按统一输入处理，dataset/revision 只作版本标签。"""
    return SPECS.get(dataset) or Spec(dataset=dataset, label=dataset)


def parse_for(spec, raw):
    if spec.parse is None or (spec.unified_when_prompt and "prompt" in raw):
        return parse_unified
    return spec.parse


# --- 提示词模板 --------------------------------------------------------------
#
# 同一侧的训练数据与外部测试数据**必须**共用同一个模板来源（代码侧：
# OpenCodeInstruct 与 MBPP+ 共用 code_prompt）。两处各写一份字面量时它们靠
# 巧合保持一致：一旦分叉，模型就在 A 格式下训练、在 B 格式下被最终评估，而
# **没有任何检查会发现**——两边各自都是自洽的。
#
# 模板内容在实验前冻结。

# DeepMath 官方评测提示里的那句要求。少了它，模型不会主动把答案放进 \boxed{}，
# 判分只能靠 Math-Verify 的抽取容错去猜，实测代价是 2.4-2.6 倍的非零奖励率。
DEEPMATH_INSTRUCTION = ("Please reason step by step, and put your final answer "
                        "within \\boxed{}.")


def math_prompt(question):
    """数学来源的输入模板（DeepMath 的全部切分共用）。

    加官方指令句是实验前定下的：预实验里两个数学来源的非零奖励率都因此明显
    提高（见上面 DEEPMATH_INSTRUCTION 的说明）。注意**只加这一句**，不要换成
    完整官方模板：预实验里完整模板反而更差。
    """
    return question.rstrip() + "\n\n" + DEEPMATH_INSTRUCTION + "\n\nSolution:\n"


def code_prompt(task):
    """代码来源的输入模板。OpenCodeInstruct 与 MBPP+ 必须共用这个包装。

    参数是**已经拼好的完整任务陈述**，因为两个数据集的题面结构不同：
    OpenCodeInstruct 的 input 本身就是自足的题目，MBPP 的 prompt 只有一句话
    描述、必须附上断言才能让模型知道函数签名。包装一致、任务文本各自组装，
    这样共享的部分仍由单一来源保证，不共享的部分是数据本身的差异。

    预实验里这类题目上 `Python solution:` 与 ```` ```python ```` 前缀的非零率
    在噪声内，取前者，少一个与参考解格式耦合的环节。
    """
    return task.rstrip() + "\n\nPython solution:\n"


# --- 小工具 ------------------------------------------------------------------

def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _filled(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
    return value


def problem_key(text):
    """直接用规范化题目文本识别重复，不计算文件或内容摘要。"""
    return " ".join(unicodedata.normalize("NFC", _text(text, "problem")).split())


def repair_surrogates(text):
    """把 UTF-16 代理还原成真字符；落单的代理无法还原，抛 ValueError。

    literal_eval 与 json.loads 都是**逐个**解码 `\\udXXX` 转义的，不会把代理对
    合并回它编码的那个星平面字符。于是一个 emoji 会变成两个落单代理，而落单
    代理不存在合法的 UTF-8 编码——json.dumps(...).encode() 会直接报
    "surrogates not allowed"。

    这不只是能否落盘的问题，更是判分正确性的问题。实测 OpenCodeInstruct 有
    590 行（30 万行中）中招，全部在 unit_tests，全部成对，例如

        assert encode_message("\\ud83d\\ude0a") == "8J+Yig=="

    该代理对正是 U+1F60A 😊 的 UTF-16 编码，其 UTF-8 字节 base64 后恰好是
    "8J+Yig=="。合并回去这条单测是对的；不合并则 "…".encode() 直接抛异常，
    模型写什么都过不了，等于给代码来源注入一批与模型无关的零奖励。

    utf-16 往返是无损且无歧义的：成对代理只有一种还原结果，落单的则解码失败。
    """
    if not _SURROGATE.search(text):
        return text
    try:
        return text.encode("utf-16", "surrogatepass").decode("utf-16")
    except UnicodeDecodeError as error:
        raise ValueError("落单的 UTF-16 代理字符，无法还原") from error


def _repair_deep(value):
    if isinstance(value, str):
        return repair_surrogates(value)
    if isinstance(value, list):
        return [_repair_deep(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_repair_deep(item) for item in value)
    if isinstance(value, dict):
        return {_repair_deep(k): _repair_deep(v) for k, v in value.items()}
    return value


def _has_surrogate(value):
    """深度扫描落单代理：任何一个都让整条记录无法 UTF-8 序列化。"""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if _SURROGATE.search(item):
                return True
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def parse_field(value, expected, literal=False):
    """解析可能被序列化成字符串的字段；literal=True 时只认 Python 字面量。

    解码后统一走 repair_surrogates，把转义拆散的代理对合并回真字符；落单代理
    无法还原，按「无法解析」中止这一条。
    """
    if isinstance(value, str):
        try:
            value = _repair_deep(ast.literal_eval(value) if literal else json.loads(value))
        except (ValueError, SyntaxError, TypeError) as error:
            raise ValueError("无法解析序列化字段") from error
    if not isinstance(value, expected):
        raise ValueError(f"字段需要 {expected.__name__}")
    return value


def single_source(names, topic_sources):
    """恰好命中一个候选主题才保留；命中多个或零个都排除。"""
    matches = {source for name, source in topic_sources.items() if name in names}
    if len(matches) != 1:
        raise Excluded("ambiguous_topic" if matches else "out_of_scope")
    return matches.pop()


def report(message):
    print(message, file=sys.stderr, flush=True)


def human(size):
    for unit in ("KB", "MB", "GB"):
        size /= 1024
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"


def load_hf(dataset, *, revision="main", **kwargs):
    """调用时才下载；返回 HF DatasetDict。revision=None 表示不传该参数。

    导入本模块不联网；HuggingFace 依赖也只在这里按需导入。
    """
    from datasets import load_dataset
    if revision is None:
        return load_dataset(dataset, **kwargs)
    return load_dataset(dataset, revision=revision, **kwargs)


def download_hf(label, loader, root, *, revision="main"):
    """跑一次 loader 触发下载，返回缓存目录与各 split 行数。

    load_dataset 的 cache_dir 只管处理后的 arrow 缓存；原始文件下载走
    HF_HUB_CACHE，不设的话会落到 ~/.cache/huggingface，同一份数据占两份磁盘。
    这里在首次导入 datasets 之前设好，已有的环境变量不覆盖。
    """
    cache = Path(root).resolve()
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    report(f"[{label}]")
    data = loader(revision=revision, cache_dir=str(cache))
    counts = {split: data[split].num_rows for split in data}
    report(f"  {counts}")
    return {"revision": revision, "cache_dir": str(cache),
            "hub_cache": os.environ["HF_HUB_CACHE"], "counts": counts}


def write_report(result, path):
    """把下载结果并入 JSON 报告，便于记录本次实验用的本地路径。

    按数据集合并而不是覆写：只下一个数据集时不该抹掉其他数据集的记录。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                merged = existing
        except json.JSONDecodeError:
            report(f"  旧报告无法解析，重新写入：{path}")
    merged.update(result)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return str(path)


# --- 统一输入 ----------------------------------------------------------------

def parse_unified(raw, ctx):
    """已经是统一格式的记录：prompt/source/reference/completion 由调用方给全。"""
    prompt = _text(raw.get("prompt"), "prompt")
    identity = _text(raw.get("problem", prompt), "problem")
    source = _integer(raw.get("source"), "source")
    reference = deepcopy(raw.get("reference"))
    if not isinstance(reference, dict) or not reference:
        raise ValueError("统一输入需要独立非空 reference")
    return Parsed(source=source, identity=identity, prompt=prompt, reference=reference,
                  targets=[("completion", _text(raw.get("completion"), "completion"))],
                  topic=raw.get("topic"))


# --- 公共流程 ----------------------------------------------------------------

def token_counts(tokenizer, prompt, completion):
    # 整条序列只 tokenize 一次，通过 offsets 确定监督边界。
    # 跨边界 token 不监督，避免把 prompt 的一部分计入 target。
    encoded = tokenizer(prompt + completion, add_special_tokens=False,
                        return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) != len(offsets) or not ids:
        raise ValueError("tokenizer 返回无效 ids/offsets")
    boundary = len(prompt)
    supervised = sum(end > start and start >= boundary for start, end in offsets)
    if completion and supervised == 0:
        raise ValueError("completion 没有可监督 token")
    return len(ids), supervised


def _select_target(parsed, tokenizer, max_tokens, target_validator):
    """按 targets 顺序选第一个合格目标；全部失败时按合并原因排除。"""
    failures = []
    for name, target in parsed.targets:
        if not isinstance(target, str) or not target.strip():
            failures.append("empty_target")
            continue
        tokens, completion_tokens = token_counts(tokenizer, parsed.prompt, target)
        if max_tokens is not None and tokens > max_tokens:
            failures.append("too_long")
            continue
        if target_validator is not None:
            valid = target_validator(target, deepcopy(parsed.reference))
            if type(valid) is not bool:
                raise ValueError("target_validator 必须返回 bool")
            if not valid:
                failures.append("invalid_target")
                continue
        return Target(name, target, tokens, completion_tokens)
    if not failures:                    # targets 本身就是空的：原始记录没给任何参考解
        raise Excluded("no_target")
    raise Excluded("no_eligible_target:" + ",".join(sorted(set(failures))))


def _dedup_key(record):
    return canonical([record["prompt"], record["completion"], record["reference"]])


def prepare_records(raw_records, tokenizer, *, dataset=None, revision="main",
                    original_split=None, topic_sources=None, max_tokens=None,
                    schemas=None, database_paths=None,
                    target_validator=None, report=None) -> list[Record]:
    """把任一数据集的原始记录转成统一 Record；目标验证可传 target_validator。

    source 默认编号取各数据集 SPEC 的 default_sources；7 来源 manifest 通过
    topic_sources 显式传入来源映射（见 pipeline/build_manifest_7src.py）。
    evaluation_only 的数据集（MBPP+）永远为外部 test；官方 test 不参与训练。
    tokenizer 需支持 offsets；不自动加特殊 token，训练端复用相同模板。
    报告版本和排除原因，不执行文件完整性验证。
    """
    _text(dataset, "dataset")
    _text(revision, "revision")
    spec = spec_for(dataset)
    original_split = original_split or ("test" if spec.evaluation_only else "train")
    if original_split not in ORIGINAL_SPLITS:
        raise ValueError("不支持的原始 split")
    if spec.evaluation_only and original_split != "test":
        raise ValueError(f"{spec.label} 只用于外部 test")
    if max_tokens is not None:
        _integer(max_tokens, "max_tokens", 1)
    if isinstance(raw_records, Mapping):
        raise ValueError('请传 ds["train"] 或记录列表，而非 DatasetDict/单个 dict')
    if topic_sources is None:
        topic_sources = spec.default_sources
    for topic, source in topic_sources.items():
        _text(topic, "topic")
        _integer(source, "source")

    evaluation = original_split != "train"
    ctx = Context(topic_sources=topic_sources, evaluation=evaluation,
                  schemas=schemas, database_paths=database_paths)
    stats = {"dataset": dataset, "revision": revision, "original_split": original_split,
             "input": 0, "kept": 0, "excluded": [], "reasons": {}}
    candidates, identities = {}, {}

    def exclude(locator, reason):
        stats["excluded"].append({"raw_id": locator, "reason": reason})

    for index, raw in enumerate(raw_records):
        stats["input"] += 1
        if not isinstance(raw, Mapping):
            raise ValueError("每条原始记录必须是 mapping")
        raw = dict(raw)
        locator = str(raw.get("unique_id", raw.get("question_id", raw.get("url") or index)))
        try:
            parsed = parse_for(spec, raw)(raw, ctx)
            # 兜底：落单的 UTF-16 代理没有合法 UTF-8 编码，这条记录写不进 manifest。
            # 正常路径已由 parse_field → repair_surrogates 合并了代理对，走到这里
            # 说明是真损坏（或某个字段绕过了 parse_field）。当场按条排除，否则
            # json.dumps 会在全部解析与 bank 采样完成后的落盘阶段整批失败，且报错
            # 指不出是哪个数据集哪一条。
            if _has_surrogate((parsed.prompt, parsed.identity, parsed.targets,
                               parsed.reference)):
                raise Excluded("lone_surrogate")
            group = problem_key(parsed.identity)
            signature = canonical(parsed.reference)
            known = identities.get(group)
            # 同题跨来源会破坏来源纯净性，属于必须中止的数据问题；但同题带不同
            # reference 是真实语料的常态（同一道题可能配两套测试集），
            # 按重复处理并在报告里单列原因，既不中止也不静默合并。
            conflicting = False
            if known is None:
                identities[group] = (parsed.source, signature)
            elif known[0] != parsed.source:
                raise ValueError(f"同题被分配到不同来源：{group}")
            else:
                conflicting = known[1] != signature
            if evaluation:
                # 评估记录没有训练目标，只统计 prompt 长度。
                target = Target("evaluation_only", "", token_counts(tokenizer, parsed.prompt, "")[0], 0)
            else:
                target = _select_target(parsed, tokenizer, max_tokens, target_validator)
        except Excluded as error:
            exclude(locator, error.reason)
            continue
        record = Record(
            id=f"{dataset}:{group}", group_id=group, source=parsed.source, prompt=parsed.prompt,
            completion=target.text, split="test" if evaluation else "",
            tokens=target.tokens, completion_tokens=target.completion_tokens,
            reference=parsed.reference,
            metadata={"dataset": dataset, "revision": revision,
                      "original_split": original_split, "raw_id": locator,
                      "problem_key": group, "external": spec.external,
                      "topic": parsed.topic, "target_choice": target.name,
                      "target_status": "evaluation_only" if evaluation else ("verified" if target_validator else "unverified"),
                      "template": f"{dataset}-v1",
                      "token_policy": "offset-start-v1-no-special-tokens"})
        reason = "duplicate_conflicting_reference" if conflicting else "duplicate"
        previous = candidates.get(group)
        if previous is None or _dedup_key(record) < _dedup_key(previous):
            if previous is not None:
                exclude(previous["metadata"]["raw_id"], reason)
            candidates[group] = record
            identities[group] = (parsed.source, signature)
        else:
            exclude(locator, reason)

    result = sorted(candidates.values(), key=lambda row: row["id"])
    stats["kept"] = len(result)
    stats["excluded"].sort(key=lambda item: (item["raw_id"], item["reason"]))
    stats["reasons"] = dict(Counter(item["reason"] for item in stats["excluded"]))
    if report is not None:
        report.clear()
        report.update(stats)
    return result


def split_records(records: list[dict], ratios: dict[str, float], salt: str) -> list[dict]:
    """按排序后的题目组和固定随机种子切分；输入顺序不影响归属。

    五个 SPLITS 键必须齐全，零比例合法。拒绝重切已分配记录和官方 test。
    比例不保证有限样本每个 split 的精确题数；新增题目后需重新切分。
    """
    _text(salt, "salt")
    if set(ratios) != set(SPLITS):
        raise ValueError(f"ratios 的键必须为 {SPLITS}")
    values = [ratios[name] for name in SPLITS]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("比例必须是有限非负数")
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("比例和必须为 1")
    rng = random.Random(salt)
    groups = sorted({_text(row.get("group_id"), "group_id") for row in records})
    points = {group: rng.random() for group in groups}
    fallback = next(name for name in reversed(SPLITS) if ratios[name] > 0)

    def assign(point):
        cumulative = 0.0
        for name, value in zip(SPLITS, values):
            cumulative += value
            if value > 0 and point < cumulative:
                return name
        return fallback

    seen_sources = {}
    result = []
    for row in records:
        group = _text(row.get("group_id"), "group_id")
        if row.get("split") or row.get("metadata", {}).get("original_split") in ("test", "dev", "validation"):
            raise ValueError("不能重切已分配记录或官方 test")
        source = _integer(row.get("source"), "source")
        if seen_sources.setdefault(group, source) != source:
            raise ValueError("同一 group 的 source 冲突")
        item = deepcopy(row)
        item["split"] = assign(points[group])
        result.append(item)
    return result


def _manifest_header(manifest):
    """校验 manifest 的头部字段，返回 (dataset_revisions, sources, required_splits)。"""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version 必须为 1")
    revisions = manifest.get("dataset_revisions")
    if not isinstance(revisions, dict) or not revisions:
        raise ValueError("需要 dataset_revisions")
    for name, value in revisions.items():
        _text(name, "dataset")
        _text(value, "revision")
    tok = manifest.get("tokenizer")
    if not isinstance(tok, dict):
        raise ValueError("需要 tokenizer 元数据")
    _text(tok.get("id"), "tokenizer.id")
    _text(tok.get("revision"), "revision")
    _text(manifest.get("split_salt"), "split_salt")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources 必须是非空整数列表")
    for source in sources:
        _integer(source, "source")
    if len(set(sources)) != len(sources):
        raise ValueError("sources 不能重复")
    required = manifest.get("required_splits", list(SPLITS))
    if not isinstance(required, list) or not required or any(s not in SPLITS for s in required):
        raise ValueError("required_splits 无效")
    return revisions, sources, required


def _manifest_record(row, revisions, sources, seen_ids):
    """校验单条记录，返回 (group, problem_key, split, source, tokens)。"""
    if not isinstance(row, dict):
        raise ValueError("record 必须是 dict")
    ident = _text(row.get("id"), "id")
    group = _text(row.get("group_id"), "group_id")
    if ident in seen_ids:
        raise ValueError("重复 id")
    seen_ids.add(ident)
    split, source = row.get("split"), row.get("source")
    _integer(source, "source")
    if split not in SPLITS or source not in sources:
        raise ValueError("未知 split/source")
    _text(row.get("prompt"), "prompt")
    if not isinstance(row.get("completion"), str):
        raise ValueError("completion 必须是字符串")
    if split == "train":
        _text(row["completion"], "train completion")
    tokens = _integer(row.get("tokens"), "tokens", 1)
    if _integer(row.get("completion_tokens"), "completion_tokens", 0) > tokens:
        raise ValueError("completion_tokens 不能超过 tokens")
    if not isinstance(row.get("reference"), dict) or not row["reference"]:
        raise ValueError("缺少 reference")
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        raise ValueError("缺少 metadata")
    dataset = meta.get("dataset")
    if dataset not in revisions or meta.get("revision") != revisions[dataset]:
        raise ValueError("记录 dataset/revision 不匹配")
    if meta.get("original_split") not in ORIGINAL_SPLITS:
        raise ValueError("缺少 original_split")
    if meta["original_split"] in ("test", "dev", "validation") and split != "test":
        raise ValueError("官方 test 泄漏")
    spec = SPECS.get(dataset)
    if spec is not None and spec.official_test_only and split == "test" and meta["original_split"] != "test":
        raise ValueError(f"{spec.label} 最终评估必须来自官方 test")
    for key in ("raw_id", "template", "token_policy"):
        _text(meta.get(key), key)
    return group, _text(meta.get("problem_key"), "problem_key"), split, source, tokens


def validate_manifest(manifest: dict) -> dict:
    """审计 schema_version=1 的内存清单，失败抛 ValueError。

    必填：dataset_revisions、tokenizer {id, revision}、sources（整数列表）、
    split_salt、records。可选 required_splits（默认五路）。
    只核验结构/一致性，不重新 tokenize、不验证数学等价或近重复。
    """
    revisions, sources, required = _manifest_header(manifest)
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("records 必须是非空列表")
    seen_ids, groups, problems = set(), {}, {}
    counts = {s: {str(k): {"records": 0, "tokens": 0} for k in sources} for s in SPLITS}
    for row in rows:
        group, problem, split, source, tokens = _manifest_record(row, revisions, sources, seen_ids)
        identity = (split, source, canonical(row["reference"]))
        if groups.setdefault(group, identity) != identity:
            raise ValueError("同题跨 split/source 或 reference 冲突")
        if problems.setdefault(problem, group) != group:
            raise ValueError("同一规范化题目被赋予不同 group")
        counts[split][str(source)]["records"] += 1
        counts[split][str(source)]["tokens"] += tokens
    for split in required:
        if any(counts[split][str(k)]["records"] == 0 for k in sources):
            raise ValueError(f"{split} 存在空来源")
    return {"records": len(rows), "groups": len(groups), "counts": counts,
            "checks": "structural_only", "status": manifest.get("status", "draft")}
