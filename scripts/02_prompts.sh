#!/usr/bin/env bash
# Build the prompt files from $WORK_DIR/raw and check them against data/SHA256SUMS.prompts.
. "$(dirname "$0")/common.sh"
[ -f "$RAW_DIR/aihub_emodial/emodial.xlsx" ] || die "raw/aihub_emodial/emodial.xlsx missing; run 01_download.sh"
koalpaca="$(cd "$RAW_DIR" && ls koalpaca/data/train-00000-of-00001-*.parquet)"

log "build_prompts -> data/prompts.jsonl"
run_tools python -I /src/build_prompts.py --raw /raw --tokenizer /model/tokenizer.json --out /data/prompts.jsonl
log "emodial_to_jsonl -> data/emodial.jsonl"
run_tools python -I /src/emodial_to_jsonl.py --xlsx /raw/aihub_emodial/emodial.xlsx --out /data/emodial.jsonl
log "add_ko_prompts -> data/prompts_ko.jsonl"
run_tools python -I /src/add_ko_prompts.py --emodial /data/emodial.jsonl --koalpaca "/raw/$koalpaca" \
  --prompts /data/prompts.jsonl --out /data/prompts_ko.jsonl

(cd "$DATA_DIR" && sha256sum --quiet -c "$ROOT/data/SHA256SUMS.prompts") \
  && log "prompt files match data/SHA256SUMS.prompts" \
  || die "prompt files differ from the published run; check tool versions and raw checksums"
