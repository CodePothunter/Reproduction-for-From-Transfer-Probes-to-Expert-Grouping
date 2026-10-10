"""在独立子进程里执行模型代码，再逐条运行断言式单测，结果打到 stdout。

走**断言式**单测（`assert f(x) == y`），是 OpenCodeInstruct、MBPP 这一类
数据集的形式：要在模型代码建立的命名空间里直接执行。

写成真实模块而不是字符串模板：模板里的语法错误只会在运行时炸在子进程里，
被上层记成「模型的代码失败」，与真正的判分结果无法区分。

必须隔在子进程里：模型代码可能死循环、吃光内存、改写全局状态或调 sys.exit，
留在主进程里会污染后续所有判分。

⚠️ 执行不可信代码。子进程 + 超时 + 地址空间与 CPU 上限只是健壮性隔离，
不是安全沙箱。只在可丢弃的实验机上运行。

用法：python -I _assertrunner.py spec.json memory_bytes cpu_seconds
spec.json 含 {"code": str, "tests": [str, ...]}。
"""
import json
import sys


def judge(code, tests):
    """先加载模型代码，再在同一命名空间里逐条执行断言。

    返回 (outcome, results)：
      compile_error  代码本身语法错误，一条断言都跑不了
      runtime_error  代码语法正确但加载时就抛异常（如顶层调用出错）
      ran            加载成功，results 逐条给出断言是否通过

    区分前两者是为了让失败构成可以分开计数。
    """
    try:
        compiled = compile(code, "<generated>", "exec")
    except SyntaxError:
        return "compile_error", [False] * len(tests)
    namespace = {"__name__": "__lwts_generated__"}
    try:
        exec(compiled, namespace)                      # noqa: S102 - 判分即执行
    except BaseException:                              # noqa: BLE001
        return "runtime_error", [False] * len(tests)
    results = []
    for test in tests:
        try:
            exec(test, namespace)                      # noqa: S102
            results.append(True)
        except BaseException:                          # noqa: BLE001 - 断言失败即未通过
            results.append(False)
    return "ran", results


def main(argv):
    spec_path, memory, cpu = argv[1], int(argv[2]), int(argv[3])
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except (ImportError, ValueError, OSError):         # 非 POSIX 或不允许设限
        pass
    spec = json.loads(open(spec_path, encoding="utf-8").read())
    try:
        outcome, results = judge(spec["code"], spec["tests"])
    except BaseException as error:                     # noqa: BLE001 - 兜底
        print(json.dumps({"outcome": "runtime_error", "results": [],
                          "error": f"{type(error).__name__}: {error}"}))
        return 0
    print(json.dumps({"outcome": outcome, "results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
