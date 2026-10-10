"""两个判分器：数学答案、断言式代码单测。全部在 CPU 上运行。

统一签名 `(answer: str, reference: dict) -> float`，取值落在 [0, 1]，供
LoRABackend 与合成后端共用。判分规则在所有专家、所有候选之间保持一致。

  math        HuggingFace Math-Verify（DeepMath 官方环境即 math-verify==0.7.0），
              抽取→latex2sympy 解析→符号判等，verify(gold, pred) 顺序不可颠倒
  assertions  OpenCodeInstruct / MBPP+ 形式的 `assert f(x) == y`：先执行模型代码，
              再在同一命名空间里逐条执行断言，全部通过才记 1.0

核心理由是判分错误**不报错**：抽不到答案一律记 0，competence 整体塌成 0，
所有比较退化为噪声而全程无异常。

无效输出（抽不到答案、代码不可编译）一律记 0.0。可传 `outcome=` 收一个标签，
把「无效」与「答错」分开计数——两者混在 0.0 里就数不出来了。

⚠️ assertion_verifier 执行的是模型生成的代码。它用独立子进程、超时和资源上限做
**健壮性隔离**，不是安全沙箱：只应在你自己的、可丢弃的实验机上运行，
不要在共享机器或含敏感数据的环境里跑。

⚠️ **math_verifier 必须在主线程调用。** Math-Verify 的解析超时用
signal.alarm 实现（math_verify/utils.py），而 signal 只能在主线程注册处理器；
放进 ThreadPoolExecutor 里每一次 parse 都抛 "signal only works in main thread"，
连 `$48$` 这种参考答案都解析失败。_parse_gold 在解析失败时抛异常而不是记 0，
正是为了让这类错误立刻暴露：若改成静默记 0，整批数学分数会一起塌成零而毫无提示。
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# extract_code 用它剥代码块。语言标签要捕获而不是丢弃——Base 模型常在一次
# 生成里连着给 Python / Java / C# 三份实现，不看标签就会把 Java 送去 Python 执行。
FENCE = re.compile(r"```([A-Za-z0-9+#-]*)\s*\n(.*?)```", re.DOTALL)


def _label(outcome, tag):
    """把一次判分的结局报给可选的回调，返回值透传方便写成 `return _label(...)`。"""
    if outcome is not None:
        outcome(tag)
    return tag


# --- 数学 -------------------------------------------------------------------

def _parse_gold(text):
    """解析参考答案。**必须包在 $…$ 里**，裸 LaTeX 会解析失败甚至解析错。

    Math-Verify 的抽取默认要定界标记（$…$、\\boxed{}、"answer is"）。DeepMath 的
    final_answer 存的是裸 LaTeX（`-\\dfrac{2}{3}`、`\\infty`、`m^2 + 1 = 0`），
    直接 parse 会：

        parse(r"-\\dfrac{2}{3}")   → []            全部题目判 0，且无异常
        parse(r"m^2 + 1 = 0")     → [0]           抓到末尾的数，静默判错

    第二种尤其危险：它不是抽取失败而是抽取到错的东西，抽取成功率显示 100%。
    """
    from math_verify import parse
    if not isinstance(text, str) or not text.strip():
        raise ValueError("参考答案缺失")
    gold = parse(f"${text.strip()}$")
    if not gold:
        raise ValueError(f"参考答案无法解析：{text!r}")
    return gold


def math_verifier(answer, reference, *, outcome=None):
    """按 Math-Verify 判等。**不预处理模型输出**——它的抽取本就是格式无关的。

    参考答案解析不了按配置错误抛出，不记 0 分：那是数据问题，
    静默记 0 会把整个来源的 competence 抹平而看不出原因。
    """
    from math_verify import verify
    gold = _parse_gold(reference.get("answer"))
    if not isinstance(answer, str) or not answer.strip():
        _label(outcome, "no_answer")
        return 0.0
    from math_verify import parse
    predicted = parse(answer)
    if not predicted:
        _label(outcome, "no_answer")            # 输出里没有可解析的答案
        return 0.0
    try:
        correct = verify(gold, predicted)       # 顺序不可颠倒：gold 在前
    except Exception:                           # noqa: BLE001 - 解析/超时失败按未通过计
        _label(outcome, "no_answer")
        return 0.0
    _label(outcome, "correct" if correct else "wrong")
    return float(correct)


# --- 代码（断言式单测）-------------------------------------------------------

PYTHON_TAGS = ("python", "python3", "py", "")


def extract_code(text):
    """取第一个 Python 代码块；没有带标签的就退回第一个块，再退回整段。

    取**第一个**而不是最后一个：Base 模型经常在 Python 解答之后接着给
    Javascript / Java / C# 版本，取最后一个会把非 Python 代码送去执行，
    判成「模型的代码不能编译」。
    """
    if not isinstance(text, str):
        return None
    blocks = FENCE.findall(text)
    typed = [body for tag, body in blocks if tag.lower() in PYTHON_TAGS]
    for candidate in (typed, [body for _, body in blocks], [text]):
        if candidate and candidate[0].strip():
            return candidate[0].strip()
    return None


# 判分子进程是同目录下的真实模块，不是源码字符串：模板里的语法错误只会炸在
# 子进程里，被记成「模型的代码失败」，与真正的判分结果无法区分。
ASSERT_RUNNER = Path(__file__).with_name("_assertrunner.py")


def assertion_verifier(answer, reference, *, timeout=30.0, memory_mb=2048,
                       outcome=None):
    """执行模型代码后逐条跑断言，**全部通过**才记 1.0。

    通过判据是 all-test-pass，不用 pass-rate 部分分：competence 由判分器定义，
    改成通过比例就把服务效用从「全部通过的概率」换成了「平均通过比例」，是另一个量。

    结局标签：no_answer / compile_error / runtime_error / timeout / wrong / correct。

    ⚠️ 执行的是模型生成的不可信代码。子进程 + 超时 + 资源上限只是健壮性隔离，
    **不是安全沙箱**。只在可丢弃的实验机上运行。
    """
    tests = reference.get("tests")
    if not isinstance(tests, list) or not tests or \
            any(not isinstance(t, str) or not t.strip() for t in tests):
        raise ValueError("reference.tests 必须是非空的断言字符串列表")
    code = extract_code(answer)
    if not code:
        _label(outcome, "no_answer")
        return 0.0

    with tempfile.TemporaryDirectory() as root:
        spec = Path(root) / "spec.json"
        spec.write_text(json.dumps({"code": code, "tests": tests}), encoding="utf-8")
        try:
            done = subprocess.run([sys.executable, "-I", "-W", "ignore",
                                   str(ASSERT_RUNNER), str(spec),
                                   str(memory_mb * 1024 * 1024), str(int(timeout) + 1)],
                                  capture_output=True, text=True, timeout=timeout,
                                  cwd=root)
        except subprocess.TimeoutExpired:
            _label(outcome, "timeout")
            return 0.0
        if not done.stdout.strip():
            _label(outcome, "runtime_error")     # 子进程被资源上限杀掉等
            return 0.0
        try:
            payload = json.loads(done.stdout.splitlines()[-1])
        except json.JSONDecodeError:
            _label(outcome, "runtime_error")
            return 0.0

    if payload.get("outcome") != "ran":
        _label(outcome, payload.get("outcome") or "runtime_error")
        return 0.0
    results = payload.get("results") or []
    if len(results) != len(tests):
        raise ValueError("判分子进程返回的断言条数与 reference.tests 不符")
    if all(results):
        _label(outcome, "correct")
        return 1.0
    _label(outcome, "wrong")                     # 代码跑通了，只是断言没全过
    return 0.0


def default_verifiers(**options):
    """按 reference.kind 分发，供 LoRABackend(verifiers=...) 直接使用。

    本包的数据层只产出两种 kind：DeepMath 的 math，OpenCodeInstruct 与 MBPP+
    的 assertions。
    """
    return {"math": lambda a, r: math_verifier(a, r, **options.get("math", {})),
            "assertions": lambda a, r: assertion_verifier(
                a, r, **options.get("assertions", {}))}
