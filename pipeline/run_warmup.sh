#!/usr/bin/env bash
# 热身 W：从基座 + 新初始化的 LoRA（按 --seed 101 派生的种子初始化）出发，跑一个
# 均匀区间：4 个专家 × 7 个来源，每格 2 块，共 56 块（每专家 14 块）。
#
# 与实际产生 W 的那次运行相比，这里少了两个只做测量、不改训练的选项（那次运行在
# 均匀区间里额外做了生成与判分形式的测量，本包不含这部分代码）。训练侧的数据顺序、
# 块分配与优化器轨迹由同一套参数决定；GPU 上的数值不保证逐位复现。
# --set anchor_interval_tokens=... 对 uniform 不起作用，保留它只为与原命令的设置一致。
#
# 用法：GPU=0 MANIFEST=runs/manifest-7src/manifest.jsonl OUT=runs/W pipeline/run_warmup.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
GPU=${GPU:-0}
MANIFEST=${MANIFEST:-runs/manifest-7src/manifest.jsonl}
OUT=${OUT:-runs/W}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$MANIFEST" ] || { echo "缺 manifest：$MANIFEST（先跑 pipeline/build_manifest_7src.py --code-pool 0）"; exit 1; }
mkdir -p "$OUT"

"$PY" -u pipeline/run_pilot.py --manifest "$MANIFEST" --models 4 --bank 80 --seed 101 \
  --batch-size 32 --max-new-tokens 8192 8192 8192 8192 8192 1024 1024 \
  --set block_tokens=32768 --set tokens_per_interval=1835008 \
  --set anchor_interval_tokens=1835008 --set gamma=0.5 --checkpoint-each-round \
  --method uniform --rounds 1 --warmup 1 \
  --out "$OUT" > "$OUT/stdout.txt" 2>&1
rc=$?
echo "[$(date '+%F %T')] W 退出码 $rc"
[ $rc -eq 0 ] && [ -f "$OUT/checkpoint.json" ] || { echo "W 没有产出 checkpoint.json"; exit 1; }
