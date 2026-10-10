"""测试前置检查：被测的 lwts 必须是本包目录里的那一份。

如果环境里另装了一份 lwts（例如以可编辑方式装在虚拟环境里的开发版本），而本包
目录又不在 sys.path 的前面，测试会静默地测到另一份代码。这里在收集任何测试之前
就核对导入位置，不对就直接停。

运行方式（在包根目录）：

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest tests
"""
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import lwts  # noqa: E402

_imported = Path(lwts.__file__).resolve()
if PACKAGE_ROOT not in _imported.parents:
    raise RuntimeError(
        f"lwts 导入自 {_imported}，不在本包目录 {PACKAGE_ROOT} 内。"
        f"请在包根目录运行，并设置 PYTHONPATH=.")
