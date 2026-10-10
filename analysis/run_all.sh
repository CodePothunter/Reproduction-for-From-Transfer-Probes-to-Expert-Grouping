#!/usr/bin/env bash
# 在 CPU 上从 results/ 的随包输入重算 results/analysis/ 下的全部分析输出（为本报告新写）。
# 不包括 param_distance.py（输入是不随包的模型快照）与 prepare_results.py（输入是不随包的原始运行目录）。
# 用法：在包根目录运行 bash analysis/run_all.sh；PYTHON 可指定解释器（默认 python）。约 3 分钟。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
"$PY" -c "import lwts, os, sys; p = os.path.realpath(lwts.__file__); sys.exit(0 if p.startswith(os.path.realpath('.') + os.sep) else 'lwts 不是包内版本: ' + p)"
OUT=results/analysis
"$PY" analysis/analyze_candidates.py > "$OUT/CANDIDATES-analysis.txt"
"$PY" analysis/analyze_label_reference.py > /dev/null          # 自写 LABELREF-analysis.{json,txt}
"$PY" analysis/proxy_posthoc.py > /dev/null                    # 自写 PROXY-posthoc.{json,txt}
"$PY" analysis/allocation_sweep.py --experts 2 3 4 5 > "$OUT/ALLOC-sweep.txt"
"$PY" analysis/probe_stats.py > /dev/null                      # 自写 PROBE-stats.{json,txt}
"$PY" analysis/c0_construction.py > /dev/null                  # 自写 C0-construction.{json,txt}
"$PY" analysis/control_agreement.py > /dev/null                # 自写 CONTROL-agreement.json
echo "完成：$OUT"
