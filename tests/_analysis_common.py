"""分析测试的共用工具：在包根目录下以子进程重跑 analysis/ 脚本，并确认用的是包内的 lwts。

子进程只设 PYTHONPATH=包根目录、CUDA_VISIBLE_DEVICES=""、HF_HUB_OFFLINE=1；输出写到临时目录，
再与 results/analysis/ 中随包的期望文件比对。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
RESULTS = PKG / "results"
EXPECTED = RESULTS / "analysis"


def env():
    e = dict(os.environ)
    e["PYTHONPATH"] = str(PKG)
    e["CUDA_VISIBLE_DEVICES"] = ""
    e["HF_HUB_OFFLINE"] = "1"
    return e


def lwts_location():
    """子进程里 import lwts 实际落到的文件。"""
    out = subprocess.run([sys.executable, "-c", "import lwts, os; print(os.path.realpath(lwts.__file__))"],
                         cwd=PKG, env=env(), check=True, capture_output=True, text=True)
    return Path(out.stdout.strip())


def assert_package_lwts():
    loc = lwts_location()
    if PKG.resolve() not in loc.parents:
        raise AssertionError(f"lwts 不是包内版本：{loc}")


def run(script, *args):
    """在包根目录运行 analysis/<script>，返回 stdout。"""
    proc = subprocess.run([sys.executable, str(PKG / "analysis" / script), *map(str, args)],
                          cwd=PKG, env=env(), capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"{script} 失败：\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tmpdir(test):
    d = Path(tempfile.mkdtemp(prefix="analysis-test-"))
    test.addCleanup(shutil.rmtree, d, True)
    return d


def without_last_line(text, marker="写入 "):
    """去掉末尾记录输出路径的那一行（路径随 --out 而变）。"""
    lines = text.rstrip("\n").split("\n")
    assert lines[-1].startswith(marker), lines[-1]
    return "\n".join(lines[:-1])


def import_analysis(name):
    """在本进程导入 analysis/<name>.py（包根目录与 analysis/ 置于 sys.path 最前）。"""
    import importlib
    for p in (str(PKG / "analysis"), str(PKG)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    module = importlib.import_module(name)
    lwts = sys.modules.get("lwts")
    if lwts is not None and PKG.resolve() not in Path(os.path.realpath(lwts.__file__)).parents:
        raise AssertionError(f"本进程的 lwts 不是包内版本：{lwts.__file__}")
    return module
