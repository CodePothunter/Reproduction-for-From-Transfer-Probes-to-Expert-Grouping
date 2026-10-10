#!/usr/bin/env bash
# 候选训练：C0–C3 各自从 W 分叉，按给定的整数块矩阵训练一个区间（56 块，每专家
# 14 块、每来源 8 块）。fixed 方法只执行给定的块，不做任何额外测量。
#
#   C0  标签参照：Algebra / Calculus / code 各一个专家（每来源 7 块），Statistics 专家
#       拿 8 块 Statistics 加其余每个来源 1 块（推导见 tests/test_c0_derivation.py）
#   C1  均匀默认（每格 2 块）
#   C2  探针导出：s17 探针表上加性代理模型的最优分配（运输 LP 顶点，不用标签树）
#   C3  在 C0 上把 Abstract Algebra 与 Statistics 在两个专家之间互换
#
# 用法：GPU=0 MANIFEST=runs/manifest-7src/manifest.jsonl W=runs/W/checkpoint.json \
#       OUT=runs/candidates pipeline/run_candidates.sh [C0 C1 C2 C3]
# 训练之后跑 pipeline/run_eval.sh 做 test 评价。
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
GPU=${GPU:-0}
MANIFEST=${MANIFEST:-runs/manifest-7src/manifest.jsonl}
W=${W:-runs/W/checkpoint.json}
OUT=${OUT:-runs/candidates}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$MANIFEST" ] || { echo "缺 manifest：$MANIFEST"; exit 1; }
[ -f "$W" ] || { echo "缺热身检查点：$W（先跑 pipeline/run_warmup.sh）"; exit 1; }
mkdir -p "$OUT"
L=$OUT/ledger.txt
note() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
stop() { note "⛔ 停止：$*"; exit 1; }

# --set anchor_interval_tokens=... 对 fixed 不起作用，保留它只为与原命令一致。
COMMON=(--manifest "$MANIFEST" --models 4 --bank 80 --seed 101 --batch-size 32
        --max-new-tokens 8192 8192 8192 8192 8192 1024 1024
        --set block_tokens=32768 --set tokens_per_interval=1835008
        --set anchor_interval_tokens=1835008 --set gamma=0.5
        --rounds 2 --warmup 1 --checkpoint-each-round --fork-from "$W")

C0='[[0,0,7,7,0,0,0],[7,7,0,0,0,0,0],[0,0,0,0,0,7,7],[1,1,1,1,8,1,1]]'
C1='[[2,2,2,2,2,2,2],[2,2,2,2,2,2,2],[2,2,2,2,2,2,2],[2,2,2,2,2,2,2]]'
C2='[[0,0,0,6,8,0,0],[0,8,2,2,0,0,2],[0,0,0,0,0,8,6],[8,0,6,0,0,0,0]]'
C3='[[0,0,0,7,7,0,0],[7,7,0,0,0,0,0],[0,0,0,0,0,7,7],[1,1,8,1,1,1,1]]'

NAMES=("$@"); [ ${#NAMES[@]} -gt 0 ] || NAMES=(C0 C1 C2 C3)
note "=== 候选训练启动：${NAMES[*]} ==="
for name in "${NAMES[@]}"; do
  blocks=${!name}
  note "=== $name：从 W 训练一个区间（fixed 分配）==="
  t0=$SECONDS
  "$PY" -u pipeline/run_pilot.py "${COMMON[@]}" --method fixed --fixed-blocks "$blocks" \
    --out "$OUT/$name" > "$OUT/$name.stdout.txt" 2>&1
  rc=$?; note "$name 训练 退出码 $rc 用时 $(( SECONDS - t0 ))s"
  [ $rc -eq 0 ] || stop "$name 训练失败"
done
note "=== 候选训练全部完成 ==="
