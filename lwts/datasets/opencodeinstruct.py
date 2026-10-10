"""OpenCodeInstruct：代码来源的训练、校准与审计数据。

选它作代码语料的理由：基座模型在它上面的非零奖励率足够高，逐题的对错
才有可分辨的变化。

按 `domain` 分源：algorithmic 与 generic 各一个来源（7 来源 manifest 里是 5 与 6，
由 topic_sources 指定；SOURCES 只是缺省映射）。两者共享语料、只差域标签，
域内与跨域的迁移才能分别观察。

单测是**断言式**的（`assert f(x) == y`），走 lwts.verifiers.assertion_verifier。

unit_tests 存的是 Python repr，其中 590/300000 行把 emoji 写成了 UTF-16 代理
转义。literal_eval 逐个解码转义、不会把代理对合并回去，于是单测里出现落单
代理——既写不进 UTF-8 的 manifest，判分时也必然失败。修复在
common.repair_surrogates（parse_field 内自动调用），那里有完整推导。
"""
from .common import (Excluded, Parsed, Spec, _filled, code_prompt, parse_field,
                     problem_key, single_source)

OPENCODEINSTRUCT = "nvidia/OpenCodeInstruct"
SOURCES = {"algorithmic": 2, "generic": 3}

# 数据按 domain 分块存放，不是随机打散的：实测分片 0-44 全是 generic，
# algorithmic 只出现在 45 / 48 / 49（合计 44180 条，其中参考解合格 13480 条）。
# 因此**不能**只读头部若干分片，那样一条 algorithmic 都取不到。
# 这个分布随 revision 而定，换版本必须用 scan_domains() 重新核对。
SHARD_COUNT = 50
DEFAULT_SHARDS = (45, 48, 49)


def parse(raw, ctx):
    domain = raw.get("domain")
    source = single_source({domain} if isinstance(domain, str) else set(),
                           ctx.topic_sources)
    question = raw.get("input")
    if not _filled(question):
        raise Excluded("missing_question")

    # 参考解必须通过它自己记录的全部单测。实测只有 25-46% 的条目满足，
    # 不过滤就会把「题目本身有问题」记成「模型答错」，让代码来源的
    # competence 带上一份与模型无关的负偏差。
    try:
        score = float(raw.get("average_test_score"))
    except (TypeError, ValueError):
        raise Excluded("missing_test_score") from None
    if score != 1.0:
        raise Excluded("reference_fails_own_tests")

    try:
        tests = parse_field(raw.get("unit_tests"), list, literal=True)
    except ValueError as error:
        raise Excluded("unparsable_unit_tests") from error
    tests = [t.strip() for t in tests if isinstance(t, str) and t.strip()]
    if not tests:
        raise Excluded("no_unit_tests")

    return Parsed(source=source, identity=question,
                  prompt=code_prompt(question),
                  reference={"kind": "assertions", "tests": tests},
                  targets=[("output", raw.get("output"))],
                  topic=domain)


def data_files(shards=DEFAULT_SHARDS, revision="main"):
    """指定分片的 parquet 路径。整份数据 6.4 GB，按需只取需要的分片。"""
    bad = [s for s in shards if not isinstance(s, int) or not 0 <= s < SHARD_COUNT]
    if bad:
        raise ValueError(f"分片编号必须落在 [0, {SHARD_COUNT})，收到 {bad}")
    root = f"hf://datasets/{OPENCODEINSTRUCT}@{revision}/data"
    return [f"{root}/train-{s:05d}-of-{SHARD_COUNT:05d}.parquet" for s in shards]


def drop_domain_conflicts(data):
    """丢掉同一题面被标成两个 domain 的题目。

    实测（分片 45/48/49，参考解合格的 108674 条）：去重后 107332 题里有 **246 题
    （0.23%）**同时挂着 algorithmic 与 generic。比例很低，是合成语料的标注噪声，
    不足以否定这个分源轴；但必须清掉——同一道题落进两个来源会直接破坏来源纯净性，
    数据层为此设了硬中止（common.prepare_records 的「同题被分配到不同来源」）。

    处理方式与 single_source 的既有规则一致：命中多个候选即排除，而不是任选
    一个。任选会给两个来源都注入标签噪声，且无法事后分辨。

    这道防线留在数据集模块而不是通用流程里：对一般语料而言同题跨来源
    确实是需要中止的 bug，只有这份合成语料把它当常态。
    """
    labels = {}
    for question, domain in zip(data["input"], data["domain"]):
        if not _filled(question):
            continue
        labels.setdefault(problem_key(question), set()).add(domain)
    conflicting = {key for key, seen in labels.items() if len(seen) > 1}
    if not conflicting:
        return data
    return data.filter(lambda row: not _filled(row["input"])
                       or problem_key(row["input"]) not in conflicting)


def load(*, revision="main", shards=DEFAULT_SHARDS, drop_conflicts=True, **kwargs):
    """只加载指定分片，返回 {"train": Dataset}。

    不走整库 load_dataset：6.4 GB 里绝大部分是 generic，而两个来源都只需要
    万条量级。分片选择必须与 scan_domains() 的结果一致，见 DEFAULT_SHARDS。
    """
    from .common import load_hf
    loaded = load_hf("parquet", revision=None,
                     data_files={"train": data_files(shards, revision)}, **kwargs)
    if drop_conflicts:
        loaded["train"] = drop_domain_conflicts(loaded["train"])
    return loaded


def scan_domains(shards=range(SHARD_COUNT), *, revision="main", cache_dir=None):
    """逐分片统计 domain 与合格条数，用来核对 DEFAULT_SHARDS。

    换 revision 或怀疑分布变化时跑它；返回 {shard: {domain: (总数, 合格数)}}。
    只读 domain 与 average_test_score 两列，但仍要下载整个分片文件。
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    report = {}
    for shard in shards:
        path = hf_hub_download(
            OPENCODEINSTRUCT, f"data/train-{shard:05d}-of-{SHARD_COUNT:05d}.parquet",
            repo_type="dataset", revision=revision, cache_dir=cache_dir)
        table = pq.read_table(path, columns=["domain", "average_test_score"])
        columns = table.to_pydict()
        counts = {}
        for domain, score in zip(columns["domain"], columns["average_test_score"]):
            total, good = counts.get(domain, (0, 0))
            counts[domain] = (total + 1, good + (float(score) == 1.0))
        report[shard] = counts
    return report


SPEC = Spec(dataset=OPENCODEINSTRUCT, label="OpenCodeInstruct", parse=parse,
            default_sources=SOURCES)
