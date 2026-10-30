#!/usr/bin/env bash
# usage: 04_train.sh {stage1|stage2} [extra train_mtp.py args...]
# The two training runs behind the published head, with the same arguments:
#   stage1: K=3 TTT self-distillation on data/gen.jsonl (2,884 records), 2 epochs  -> ckpt/stage1
#   stage2: Korean continued training on gen.jsonl + gen_ko.jsonl, 1 epoch          -> ckpt/stage2
. "$(dirname "$0")/common.sh"
stage=${1:-}; shift || true
case $stage in
  stage1)
    "$ROOT/scripts/train.sh" stage1 --data /data/gen.jsonl --K 3 --epochs 2 --lr 5e-5 --warmup 10 \
      --eval-every 50 --eval-max 263 "$@" ;;
  stage2)
    [ -f "$CKPT_DIR/stage1/mtp.safetensors" ] || die "ckpt/stage1/mtp.safetensors missing; run stage1 first"
    cat "$DATA_DIR/gen.jsonl" "$DATA_DIR/gen_ko.jsonl" > "$DATA_DIR/gen_all.jsonl"
    "$ROOT/scripts/train.sh" stage2 --data /data/gen_all.jsonl --K 3 --epochs 1 --lr 2e-5 --warmup 10 \
      --eval-every 90 --eval-max 343 --eval-by-src --init /ckpt/stage1/mtp.safetensors "$@" ;;
  *) sed -n '2,5p' "$0" >&2; exit 2 ;;
esac
