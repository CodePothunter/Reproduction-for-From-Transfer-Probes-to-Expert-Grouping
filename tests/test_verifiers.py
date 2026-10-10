"""两个判分器的测试。全部 CPU，不需要模型也不需要网络。

数学判分来自 Math-Verify 的官方实现，所以这里测的是**接入是否正确**，
不是「比较规则设计得好不好」。断言式代码判分测的是通过判据、结局标签与
子进程隔离。
"""
import ast
import unittest

from lwts.verifiers import assertion_verifier, default_verifiers, extract_code, math_verifier


class MathTests(unittest.TestCase):
    def check(self, answer, expected="2", outcome=None):
        return math_verifier(answer, {"kind": "math", "answer": expected}, outcome=outcome)

    def test_bare_latex_gold_parses(self):
        """DeepMath 的 final_answer 是裸 LaTeX，不带 \\boxed。

        Math-Verify 的抽取默认要锚点，裸 LaTeX 必须先包进 $…$，否则
        parse(r"-\\dfrac{2}{3}") 返回 []，全部题目静默判 0。
        """
        for gold in (r"-\dfrac{2}{3}", r"\dfrac{\pi}{2}", r"\infty", r"\phi^4", "|A|"):
            self.assertEqual(self.check(f"the answer is ${gold}$", gold), 1.0, gold)

    def test_gold_that_would_silently_misparse(self):
        """裸 parse 会把 `m^2 + 1 = 0` 抓成 `0`——抽到错的东西，比抽不到更难发现。"""
        self.assertEqual(self.check(r"we get $m^2 + 1 = 0$", "m^2 + 1 = 0"), 1.0)
        self.assertEqual(self.check(r"we get $0$", "m^2 + 1 = 0"), 0.0)

    def test_base_model_output_without_boxed(self):
        """Base 模型不遵循格式指令，实测生成里 \\boxed 出现 0 次。

        典型陷阱：朴素抽取会取到**模数 645**，而不是模型主张的 2。
        抽取正确性与答案正确性是两件事——判分器的职责是忠实取出模型的主张，
        再判它对不对。这里模型主张 2、真值是 8，所以正确的结果是「抽到 2、判错」。
        """
        text = (r"The order of 2 modulo 645 is 10, and 731 mod 10 = 1, "
                r"so \(2^{731} \equiv 2 \pmod{645}\).")
        self.assertEqual(self.check(text, "2"), 1.0, "应抽到模型主张的 2，而不是模数 645")
        self.assertEqual(self.check(text, "8"), 0.0, "真值是 8，模型答错")
        self.assertEqual(self.check(text, "645"), 0.0, "抽到模数才会误判为对")

    def test_symbolic_equivalence(self):
        """符号判等，不是字符串匹配——这正是 Math-Verify 取代 is_equiv 的理由。"""
        self.assertEqual(self.check(r"\boxed{\sqrt{4}}", "2"), 1.0)
        self.assertEqual(self.check(r"\boxed{1/2}", "0.5"), 1.0)
        self.assertEqual(self.check(r"\boxed{\frac{2}{4}}", "0.5"), 1.0)

    def test_wrong_answer(self):
        self.assertEqual(self.check(r"\boxed{3}", "2"), 0.0)

    def test_outcome_separates_invalid_from_wrong(self):
        """无效输出与答错必须分开计数，混在 0.0 里就数不出无效率。"""
        for text, tag in ((r"\boxed{3}", "wrong"), (r"\boxed{2}", "correct"),
                          ("I cannot solve this", "no_answer"), ("", "no_answer")):
            tags = []
            self.check(text, "2", outcome=tags.append)
            self.assertEqual(tags, [tag], text)

    def test_missing_gold_raises(self):
        """参考答案缺失是数据问题，必须报错而不是记 0 分。

        静默记 0 会把整个来源的 competence 抹平，而且看不出原因。
        """
        for bad in ("", None, "   "):
            with self.assertRaises(ValueError):
                self.check(r"\boxed{2}", bad)

    def test_parse_falls_back_to_raw_string(self):
        """parse 的 fallback_mode='first_match' 会把解析不了的 gold 原样退回。

        所以「gold 解析失败」这道防线几乎拦不住东西：真正的风险是 gold 被
        解析成**错的符号值**（不加 $ 时 `m^2 + 1 = 0` → `0`），而那个由
        _parse_gold 的 $…$ 包裹解决，不是由这里的空值检查解决。
        """
        from math_verify import parse
        self.assertEqual(parse(r"$\this{is not$"), [r"\this{is not"])
        self.assertEqual(parse("$   $"), [])

    def test_must_run_on_the_main_thread(self):
        """Math-Verify 的解析超时用 signal.alarm，signal 只能在主线程注册。

        放进线程池后每一次 parse 都抛 "signal only works in main thread"，
        连 `$48$` 这种参考答案都解析失败——若判分器写成「解析失败记 0」，
        整批数学分数会一起塌成零而毫无提示。要并发只能并发代码判分（它本就起子进程）。
        """
        from concurrent.futures import ThreadPoolExecutor
        reference = {"kind": "math", "answer": "48"}
        answer = "the answer is $48$"
        self.assertEqual(math_verifier(answer, reference), 1.0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaises(ValueError):
                pool.submit(math_verifier, answer, reference).result()


class ExtractCodeTests(unittest.TestCase):
    def test_extract_code_prefers_fence(self):
        self.assertEqual(extract_code("blah\n```python\nprint(1)\n```\ntail"), "print(1)")
        self.assertEqual(extract_code("print(2)"), "print(2)")
        self.assertIsNone(extract_code("   "))

    def test_extract_code_picks_the_python_block(self):
        """Base 模型常一次给出多语言实现，取最后一块会把 Java 送去 Python 执行。"""
        text = ("```python\nprint(1)\n```\n\nJavascript solution:\n"
                "```javascript\nconsole.log(1);\n```\n\nJava solution:\n"
                "```java\nSystem.out.println(1);\n```")
        self.assertEqual(extract_code(text), "print(1)")
        # 没有 python 标签时退回第一个块，而不是最后一个
        self.assertEqual(extract_code("```\nprint(1)\n```\n```java\nx;\n```"), "print(1)")


class AssertionTests(unittest.TestCase):
    """断言式单测（OpenCodeInstruct / MBPP+ 这一类）。"""

    TESTS = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]

    def score(self, answer, tests=None):
        tags = []
        value = assertion_verifier(answer, {"kind": "assertions",
                                            "tests": tests or self.TESTS},
                                   timeout=15.0, outcome=tags.append)
        return value, (tags[0] if tags else None)

    def test_all_pass_is_the_criterion(self):
        """全部断言通过才记 1.0，不给部分分。"""
        self.assertEqual(self.score("```python\ndef add(a, b):\n    return a + b\n```"),
                         (1.0, "correct"))
        # 三条里过两条仍是 0：a-b 在 (0,0) 与 (-1,1) 上碰巧对，(1,2) 上错
        self.assertEqual(self.score("```python\ndef add(a, b):\n    return a - b\n```"),
                         (0.0, "wrong"))

    def test_failure_tags(self):
        """无效输出按失败类型分开计数。"""
        self.assertEqual(self.score("```python\ndef add(a, b)\n    return a\n```")[1],
                         "compile_error")
        self.assertEqual(self.score("```python\nraise ValueError('boom')\n```")[1],
                         "runtime_error")
        self.assertEqual(self.score("```python\nwhile True:\n    pass\n```")[1],
                         "timeout")
        self.assertEqual(self.score("")[1], "no_answer")

    def test_rejects_malformed_reference(self):
        """题目本身无效属于配置错误，必须抛异常而不是静默记 0。"""
        for tests in ([], ["  "], "assert True", [1, 2]):
            with self.assertRaises(ValueError, msg=repr(tests)):
                assertion_verifier("```python\npass\n```",
                                   {"kind": "assertions", "tests": tests})

    def test_runner_is_a_real_module(self):
        """字符串模板里的语法错误会被误记成模型的失败，所以判分子进程是真实模块。"""
        from lwts.verifiers import ASSERT_RUNNER
        self.assertTrue(ASSERT_RUNNER.exists())
        ast.parse(ASSERT_RUNNER.read_text(encoding="utf-8"))

    def test_subprocess_isolation_survives_hostile_code(self):
        """模型代码可能调 sys.exit 或改全局状态，不能影响判分进程。"""
        self.assertEqual(self.score("```python\nimport sys\nsys.exit(1)\n```")[0], 0.0)
        self.assertEqual(self.score("```python\ndef add(a, b):\n    return a + b\n```"),
                         (1.0, "correct"), "上一条不应污染后续判分")


class RegistryTests(unittest.TestCase):
    def test_default_verifiers_cover_every_reference_kind(self):
        """数据层产出的 reference.kind 必须都能被分发到。"""
        kinds = {"math", "assertions"}
        self.assertEqual(set(default_verifiers()), kinds)
        rows = {
            "deepmath": dict(question="Compute 1+1", topic="Algebra", final_answer="2",
                             r1_solution_1="2"),
            "opencodeinstruct": dict(input="Implement add.", domain="generic",
                                     average_test_score=1.0,
                                     unit_tests="['assert add(1, 2) == 3']",
                                     output="def add(a, b): return a + b"),
            "mbppplus": dict(prompt="Add two numbers.", test_list=["assert add(1, 2) == 3"],
                             test="assert add(2, 2) == 4", test_imports=[]),
        }
        from lwts.datasets import MODULES, prepare_records

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": list(range(len(text))),
                        "offset_mapping": [(i, i + 1) for i in range(len(text))]}

        for name, module in MODULES.items():
            split = "test" if module.SPEC.evaluation_only else "train"
            extra = {"topic_sources": {"code": 5}} if name == "mbppplus" else {}
            out = prepare_records([rows[name]], Tokenizer(), dataset=module.SPEC.dataset,
                                  original_split=split, **extra)
            self.assertIn(out[0]["reference"]["kind"], kinds, name)

    def test_options_reach_every_verifier(self):
        """default_verifiers 的每个 kind 都要能收 options，否则超时设不进去。"""
        verifiers = default_verifiers(math={"outcome": None}, assertions={"timeout": 15.0})
        self.assertEqual(verifiers["math"](r"\boxed{2}", {"answer": "2"}), 1.0)
        self.assertEqual(verifiers["assertions"](
            "```python\ndef add(a, b):\n    return a + b\n```",
            {"tests": ["assert add(1, 2) == 3"]}), 1.0)


if __name__ == "__main__":
    unittest.main()
