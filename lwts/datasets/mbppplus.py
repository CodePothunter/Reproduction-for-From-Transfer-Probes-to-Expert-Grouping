"""MBPP+：代码侧的外部评估，永远是 test，不参与训练或路由选择。

训练语料（OpenCodeInstruct）之外的知名基准，只做推理与判分。

**没有域标签**，不区分 algorithmic / generic，只能整批指到一个代码来源
（7 来源 manifest 里指到 algorithmic，见 pipeline/build_manifest_7src.py）。
因此它评的是代码侧的整体能力，而不是逐来源能力——逐来源评估由
OpenCodeInstruct 自身的 held-out 切分承担。

判分用 EvalPlus 增强后的 `test` 字段而非基础的 `test_list`：前者是 MBPP+ 的
全部意义所在（基础断言太弱，公认会放过错误解）。该字段是一段完整脚本，末尾
直接按函数名调用，因此**先 exec 模型代码、再在同一命名空间 exec 它**即可，
正是 assertion_verifier 的协议。

题面只有一句话描述、不含函数名，必须附上基础断言模型才知道签名——这是 MBPP
的标准用法，不是本项目的发明。
"""
from .common import Excluded, Parsed, Spec, Target, _filled, _text, code_prompt, load_hf

MBPPPLUS = "evalplus/mbppplus"

# 与 OpenCodeInstruct 的两个来源同属代码域。MBPP+ 不带域标签，两个来源共用
# 同一批题目；分配到哪个来源由调用方通过 topic_sources 指定。
SOURCES = {"code": 2}


def task_statement(prompt, test_list):
    """题面 + 基础断言。断言是签名的唯一来源，去掉模型无从知道函数叫什么。"""
    body = prompt.rstrip()
    shown = [t.strip() for t in test_list if isinstance(t, str) and t.strip()]
    if not shown:
        raise Excluded("no_signature_tests")
    return body + "\n\nYour code should pass these tests:\n\n" + "\n".join(shown)


def parse(raw, ctx):
    prompt = _text(raw.get("prompt"), "prompt")
    test_list = raw.get("test_list")
    if not isinstance(test_list, list):
        raise Excluded("missing_test_list")
    plus = raw.get("test")
    if not _filled(plus):
        raise Excluded("missing_plus_tests")

    # test_imports 只有 2.6% 的题非空；有就放在增强测试之前，作为独立的一条。
    imports = [t for t in (raw.get("test_imports") or [])
               if isinstance(t, str) and t.strip()]
    tests = imports + [plus]

    source = ctx.topic_sources.get("code")
    if source is None:
        raise Excluded("out_of_scope")
    statement = task_statement(prompt, test_list)
    return Parsed(source=source, identity=prompt,
                  prompt=code_prompt(statement),
                  reference={"kind": "assertions", "tests": tests},
                  targets=[], topic="code")


def load(*, revision="main", **kwargs):
    return load_hf(MBPPPLUS, revision=revision, **kwargs)


SPEC = Spec(dataset=MBPPPLUS, label="MBPP+", parse=parse, default_sources=SOURCES,
            evaluation_only=True, external=True)
