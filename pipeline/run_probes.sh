#!/usr/bin/env bash
# 单源迁移探针：从基座 + 新初始化的 LoRA 出发（状态 S0），对每个来源 k 训练 16 个
# 连续块，在每来源 80 道 audit 题上测全部 7 个来源的前后变化。两个训练流种子 17 / 23。
#
# ⚠ 推断的命令：这两次运行的命令行没有留档。下面的参数由运行日志的表头与输出反推
#   （7 来源 manifest、每来源 audit bank 80、每格 16 块、数学 8192 / 代码 1024 的生成
#   上限、种子 17 / 23），其余取 probe_transfer.py 的缺省值。
# ⚠ 种子 23 那次运行在最后一行（generic）完成前被停止；随包的
#   results/probes/s23_transfer_partial.json 是用 salvage_partial_tensor.py 从运行日志
#   恢复的前六行。本脚本会把七行都跑完。
#
# 用法（在包根目录或任意目录）：
#   GPU=0 MANIFEST=runs/manifest-7src/manifest.jsonl OUT_ROOT=runs pipeline/run_probes.sh [17 23]
# 每个种子约 16 GPU 小时（单卡、24 GB）。
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
GPU=${GPU:-0}
MANIFEST=${MANIFEST:-runs/manifest-7src/manifest.jsonl}
OUT_ROOT=${OUT_ROOT:-runs}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEEDS=("$@"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(17 23)
[ -f "$MANIFEST" ] || { echo "缺 manifest：$MANIFEST（先跑 pipeline/build_manifest_7src.py --code-pool 0）"; exit 1; }

for seed in "${SEEDS[@]}"; do
  out="$OUT_ROOT/probe-s$seed"
  mkdir -p "$out"
  echo "[$(date '+%F %T')] 探针 种子 $seed → $out"
  "$PY" -u pipeline/probe_transfer.py --manifest "$MANIFEST" \
    --bank 80 --blocks 16 --seed "$seed" \
    --max-new-tokens 8192 8192 8192 8192 8192 1024 1024 \
    --out "$out" > "$out/stdout.txt" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] 种子 $seed 退出码 $rc"
  [ $rc -eq 0 ] || exit $rc
done
