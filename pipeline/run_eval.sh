#!/usr/bin/env bash
# 候选 test 评价：每个候选用训练完的 checkpoint-r2.json，按规则固定路由，在 test
# 每来源排序后的前 400 题上随机解码，每题采样 K 次（K=2，评价种子 2001..2000+K）。
# 评价种子在各候选间相同，便于按题配对比较。
#
# 路由规则：每个来源由获得它最多块的专家服务，并列取编号小者。
#
# 用法：GPU=0 MANIFEST=runs/manifest-7src/manifest.jsonl OUT=runs/candidates K=2 \
#       pipeline/run_eval.sh [C0 C1 C2 C3]
# 每份评价约 9 GPU 小时（单卡、24 GB）。
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
GPU=${GPU:-0}
MANIFEST=${MANIFEST:-runs/manifest-7src/manifest.jsonl}
OUT=${OUT:-runs/candidates}
K=${K:-2}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f "$MANIFEST" ] || { echo "缺 manifest：$MANIFEST"; exit 1; }
L=$OUT/ledger.txt
note() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
stop() { note "⛔ 停止：$*"; exit 1; }

C0='[[0,0,7,7,0,0,0],[7,7,0,0,0,0,0],[0,0,0,0,0,7,7],[1,1,1,1,8,1,1]]'
C1='[[2,2,2,2,2,2,2],[2,2,2,2,2,2,2],[2,2,2,2,2,2,2],[2,2,2,2,2,2,2]]'
C2='[[0,0,0,6,8,0,0],[0,8,2,2,0,0,2],[0,0,0,0,0,8,6],[8,0,6,0,0,0,0]]'
C3='[[0,0,0,7,7,0,0],[7,7,0,0,0,0,0],[0,0,0,0,0,7,7],[1,1,8,1,1,1,1]]'

routing () {  # 每个来源由获得它最多块的专家服务，并列取编号小者
  "$PY" -c "import json,sys;b=json.loads(sys.argv[1]);print(','.join(str(max(range(4),key=lambda i:(b[i][k],-i))) for k in range(7)))" "$1"
}

NAMES=("$@"); [ ${#NAMES[@]} -gt 0 ] || NAMES=(C0 C1 C2 C3)
for name in "${NAMES[@]}"; do
  blocks=${!name}
  ckpt="$OUT/$name/checkpoint-r2.json"
  [ -f "$ckpt" ] || stop "缺 $ckpt（先跑 pipeline/run_candidates.sh）"
  route=$(routing "$blocks"); note "$name 路由（按规则）$route"
  for s in $(seq 1 "$K"); do
    seed=$((2000 + s)); t0=$SECONDS
    "$PY" -u pipeline/score_test_extension.py --checkpoint "$ckpt" \
      --routing "$route" --manifest "$MANIFEST" --method fixed --fixed-blocks "$blocks" \
      --offset 0 --n 400 --eval-seed $seed \
      --out "$OUT/$name-eval-s$seed" > "$OUT/$name-eval-s$seed.stdout.txt" 2>&1
    rc=$?; note "$name 评价 种子 $seed 退出码 $rc 用时 $(( SECONDS - t0 ))s"
    [ $rc -eq 0 ] || stop "$name 评价失败"
  done
done
note "=== 评价全部完成 ==="
