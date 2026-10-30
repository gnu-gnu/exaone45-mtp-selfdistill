#!/usr/bin/env bash
# usage: 03_generate.sh {pilot|main|ko|all|smoke [N]}
# Collects EXAONE's own responses (token ids) for the prompt files, in the same three rounds as the published run:
#   pilot: first 384 prompts (263 eval + 121 train), no system prompt
#   main:  next 2,500 train prompts with the system prompt mix      -> data/gen.jsonl  (stage 1 data)
#   ko:    all Korean prompts with the system prompt mix            -> data/gen_ko.jsonl
# Each round resumes where it stopped. smoke writes N records to data/gen_smoke.jsonl.
. "$(dirname "$0")/common.sh"
round=${1:-}; [ -n "$round" ] || { sed -n '2,8p' "$0" >&2; exit 2; }
[ -f "$DATA_DIR/prompts.jsonl" ] || die "data/prompts.jsonl missing; run 02_prompts.sh"
url="$VLLM_BASE_URL/v1/chat/completions"
gen() { run_tools python -I /src/gen_client.py --url "$url" --model "$SERVED_MODEL_NAME" \
          --concurrency "${GEN_CONCURRENCY:-96}" "$@"; }

started=0
if [ "$SERVER_MODE" = local ] && ! api -f -o /dev/null "$VLLM_BASE_URL/v1/models" 2>/dev/null; then
  "$ROOT/scripts/serve.sh" up gen
  started=1
fi
cleanup() { [ "$started" = 1 ] && "$ROOT/scripts/serve.sh" down; }
trap cleanup EXIT

case $round in
  pilot) gen --prompts /data/prompts.jsonl --out /data/gen.jsonl --limit 384 ;;
  main)  gen --prompts /data/prompts.jsonl --out /data/gen.jsonl --limit 2500 --sys-mix ;;
  ko)    gen --prompts /data/prompts_ko.jsonl --out /data/gen_ko.jsonl --sys-mix ;;
  all)
    n=$( [ -f "$DATA_DIR/gen.jsonl" ] && wc -l < "$DATA_DIR/gen.jsonl" || echo 0)
    [ "$n" -lt 384 ] && gen --prompts /data/prompts.jsonl --out /data/gen.jsonl --limit $((384 - n))
    n=$(wc -l < "$DATA_DIR/gen.jsonl")
    [ "$n" -lt 2884 ] && gen --prompts /data/prompts.jsonl --out /data/gen.jsonl --limit $((2884 - n)) --sys-mix
    gen --prompts /data/prompts_ko.jsonl --out /data/gen_ko.jsonl --sys-mix ;;
  smoke) rm -f "$DATA_DIR/gen_smoke.jsonl"
         gen --prompts /data/prompts.jsonl --out /data/gen_smoke.jsonl --limit "${2:-8}" --concurrency "${2:-8}" ;;
  *) die "unknown round $round" ;;
esac
log "gen.jsonl: $(wc -l < "$DATA_DIR/gen.jsonl" 2>/dev/null || echo 0) records, gen_ko.jsonl: $(wc -l < "$DATA_DIR/gen_ko.jsonl" 2>/dev/null || echo 0)"
