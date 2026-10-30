#!/usr/bin/env bash
# usage: analyze.sh regime [MTP]   paired test: same answer tokens after real reasoning vs after an empty think block
#        analyze.sh eval [MTP]      offline acceptance per source and mode on the stage 2 eval split (343 records)
# MTP is a checkpoint name under ckpt/ (uses ckpt/NAME/mtp.safetensors); omit it for the original head in MODEL_DIR.
# Both run on the GPU and need the generated data (03_generate.sh). Results: ckpt/analysis-*/log.jsonl.
. "$(dirname "$0")/common.sh"
what=${1:-}; mtp=${2:-}
init=()
[ -n "$mtp" ] && { [ -f "$CKPT_DIR/$mtp/mtp.safetensors" ] || die "ckpt/$mtp/mtp.safetensors not found"; init=(--init "/ckpt/$mtp/mtp.safetensors"); }
tag=${mtp:-orig}
case $what in
  regime)
    # The published numbers (58.0% vs 2.5%) come from the 384 pilot records.
    head -n 384 "$DATA_DIR/gen.jsonl" > "$DATA_DIR/gen_pilot.jsonl"
    "$ROOT/scripts/train.sh" "analysis-regime-$tag" --eval-only --regime 512 --data /data/gen_pilot.jsonl "${init[@]}"
    grep -h regime_mtp_same_tokens "$CKPT_DIR/analysis-regime-$tag/log.jsonl" | tail -1 ;;
  eval)
    [ -f "$DATA_DIR/gen_all.jsonl" ] || cat "$DATA_DIR/gen.jsonl" "$DATA_DIR/gen_ko.jsonl" > "$DATA_DIR/gen_all.jsonl"
    "$ROOT/scripts/train.sh" "analysis-eval-$tag" --eval-only --eval-by-src --eval-max 343 --data /data/gen_all.jsonl "${init[@]}" ;;
  *) sed -n '2,5p' "$0" >&2; exit 2 ;;
esac
