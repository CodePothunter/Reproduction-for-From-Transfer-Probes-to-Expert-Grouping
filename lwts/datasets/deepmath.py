"""DeepMath-103K：数学来源的训练、校准、审计和内部测试。

topic 是 `A -> B -> C` 形式的层级串，按节点匹配来源。
final_answer 只进 reference，绝不拼进 prompt。
"""
from .common import (Excluded, Parsed, Spec, _filled, load_hf, math_prompt,
                     single_source)

DEEPMATH = "zwhe99/DeepMath-103K"
SOURCES = {"Algebra": 0, "Number Theory": 1}


def parse(raw, ctx):
    topic = raw.get("topic", "")
    nodes = {part.strip() for part in topic.split("->")} if isinstance(topic, str) else set()
    source = single_source(nodes, ctx.topic_sources)
    question, answer = raw.get("question"), raw.get("final_answer")
    if not _filled(question) or not _filled(answer):
        raise Excluded("missing_question_or_answer")
    return Parsed(source=source, identity=question,
                  prompt=math_prompt(question),
                  reference={"kind": "math", "answer": answer},
                  targets=[(f"r1_solution_{i}", raw.get(f"r1_solution_{i}")) for i in range(1, 4)],
                  topic=topic)


def drop_topic_conflicts(data, topic_sources):
    """丢掉同一题面被挂到两个候选主题上的题目。

    topic 是层级串，候选主题必须在层级上互斥（不能同时取祖先与后代），但**同一道
    题可能以两行分别挂在两个分支下**。只取两个粗主题时几乎不发生，取五个细分
    主题后就会碰上。

    实测（候选为 Integral / Differential Calculus、Abstract / Linear
    Algebra、Statistics）：命中恰好一个候选的 51569 行、去重后 51039 题，其中
    **16 题（0.03%）**跨来源，且 16 题全部是 Integral ↔ Differential Calculus。
    比例比代码侧的 0.23% 还低一个数量级，是标注噪声而不是主题纠缠——这两个主题
    在数据里是干净可分的，而它们恰好又是标签最可能归成一组的那一对。

    处理方式与 single_source 的既有规则一致：命中多个即排除，不任选其一。任选会
    给两个来源都注入标签噪声，且事后无法分辨。

    不清掉的话 prepare_records 会按数据问题硬中止（「同题被分配到不同来源」）。
    那道中止是对的：同题跨来源会直接破坏来源纯净性，而块必须来源纯净，否则
    块的训练效果不知道该记在哪个格子上。
    """
    from .common import problem_key
    labels = {}
    for question, topic in zip(data["question"], data["topic"]):
        if not _filled(question) or not isinstance(topic, str):
            continue
        nodes = {part.strip() for part in topic.split("->")}
        matched = {topic_sources[n] for n in nodes if n in topic_sources}
        if matched:
            labels.setdefault(problem_key(question), set()).update(matched)
    conflicting = {key for key, seen in labels.items() if len(seen) > 1}
    if not conflicting:
        return data
    return data.filter(lambda row: not _filled(row["question"])
                       or problem_key(row["question"]) not in conflicting)


def load(*, revision="main", **kwargs):
    return load_hf(DEEPMATH, revision=revision, **kwargs)


SPEC = Spec(dataset=DEEPMATH, label="DeepMath", parse=parse, default_sources=SOURCES,
            unified_when_prompt=True)
