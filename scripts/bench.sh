#!/usr/bin/env bash
# usage: bench.sh LABEL {off|on|greedy|conc|all} [RUNS]
# Measures the server at VLLM_BASE_URL with the five natural prompts (max_tokens 400, default sampling):
#   off / on: thinking off / on, RUNS passes (default 2); tok/s and draft acceptance from /metrics deltas
#   greedy:   thinking off, temperature 0, per-position acceptance; token ids saved to greedy.jsonl
#   conc:     thinking off at concurrency 1/4/8 (needs MAX_NUM_SEQS >= 8)
# Results go to $WORK_DIR/bench/LABEL/. Run one benchmark at a time against an otherwise idle server.
. "$(dirname "$0")/common.sh"
. "$ROOT/bench/lib.sh"
label=${1:-}; what=${2:-}; runs=${3:-2}
[ -n "$label" ] && [ -n "$what" ] || { sed -n '2,8p' "$0" >&2; exit 2; }
out="$BENCH_DIR/$label"; mkdir -p "$out"
api -f -o /dev/null "$VLLM_BASE_URL/v1/models" || die "no server at $VLLM_BASE_URL"
A=vllm:spec_decode_num_accepted_tokens_total
D=vllm:spec_decode_num_draft_tokens_total

natural() { # thinking: false|true
  local think=$1 r name body resp t0 t1 a0 d0 a1 d1 c p
  for r in $(seq 1 "$runs"); do
    echo "--- natural thinking=$think run $r"
    for name in "${PROMPT_NAMES[@]}"; do
      body=$(jq -cn --arg m "$SERVED_MODEL_NAME" --argjson msgs "$(prompt_messages "$name")" --argjson th "$think" \
        '{model:$m,messages:$msgs,max_tokens:400,chat_template_kwargs:{enable_thinking:$th}}')
      a0=$(metric $A); d0=$(metric $D)
      t0=$(date +%s.%N)
      resp=$(api "$VLLM_BASE_URL/v1/chat/completions" -H "content-type: application/json" -d "$body")
      t1=$(date +%s.%N)
      a1=$(metric $A); d1=$(metric $D)
      c=$(echo "$resp" | jq -r '.usage.completion_tokens'); p=$(echo "$resp" | jq -r '.usage.prompt_tokens')
      [ "$c" != null ] || die "request failed: $(echo "$resp" | head -c 300)"
      printf "  %-16s prompt=%-5s compl=%-4s %6.1fs  %5.1f tok/s  draft=%-5d acc=%-4d rate=%5.1f%%\n" "$name" "$p" "$c" \
        "$(echo "$t1 - $t0" | bc)" "$(echo "$c / ($t1 - $t0)" | bc -l)" $((d1 - d0)) $((a1 - a0)) \
        "$(awk -v a=$((a1 - a0)) -v d=$((d1 - d0)) 'BEGIN{print (d > 0 ? 100 * a / d : 0)}')"
    done
  done
}

summarize() { # mean tok/s and pooled acceptance over all lines of a natural log
  awk '/tok\/s/ {for (i = 1; i <= NF; i++) {if ($(i+1) == "tok/s") {s += $i; n++} if ($i ~ /^draft=/) {split($i, x, "="); d += x[2]} if ($i ~ /^acc=/) {split($i, y, "="); a += y[2]}}}
       END {if (n) printf "mean %.2f tok/s over %d requests, acceptance %.1f%% (%d/%d)\n", s / n, n, (d ? 100 * a / d : 0), a, d}' "$1"
}

greedy() {
  local name resp b0 b1
  : > "$out/greedy.jsonl"
  pp() { api "$VLLM_BASE_URL/metrics" | awk '/^vllm:spec_decode_num_drafts_total\{/ {d+=$2} /^vllm:spec_decode_num_accepted_tokens_per_pos_total\{/ {match($0,/position="[0-9]+"/); k=substr($0,RSTART+10,RLENGTH-11); a[k]+=$2} END{printf "%d %d %d %d", d, a[0], a[1], a[2]}'; }
  for name in "${PROMPT_NAMES[@]}"; do
    read -ra b0 <<< "$(pp)"
    resp=$(api "$VLLM_BASE_URL/v1/chat/completions" -H "content-type: application/json" \
      -d "$(jq -cn --arg m "$SERVED_MODEL_NAME" --argjson msgs "$(prompt_messages "$name")" \
        '{model:$m,messages:$msgs,max_tokens:400,temperature:0,return_token_ids:true,chat_template_kwargs:{enable_thinking:false}}')")
    read -ra b1 <<< "$(pp)"
    echo "$resp" | jq -c --arg n "$name" --argjson d $((b1[0] - b0[0])) --argjson a0 $((b1[1] - b0[1])) \
      --argjson a1 $((b1[2] - b0[2])) --argjson a2 $((b1[3] - b0[3])) \
      '{name:$n, prompt_ids:.prompt_token_ids, out_ids:.choices[0].token_ids, vllm:{drafts:$d, acc_per_pos:[$a0,$a1,$a2]}}' >> "$out/greedy.jsonl"
    echo "  $name steps=$((b1[0] - b0[0])) acc_per_pos=[$((b1[1] - b0[1])),$((b1[2] - b0[2])),$((b1[3] - b0[3]))]"
  done
}

conc() {
  local w="$out/conc" C N t0 t1 a0 d0 a1 d1 i name
  mkdir -p "$w"
  i=0
  for name in "${PROMPT_NAMES[@]}"; do
    jq -cn --arg m "$SERVED_MODEL_NAME" --argjson msgs "$(prompt_messages "$name")" \
      '{model:$m,messages:$msgs,max_tokens:400,chat_template_kwargs:{enable_thinking:false}}' > "$w/p$i.json"
    i=$((i + 1))
  done
  one() { local t0 t1 r; t0=$(date +%s.%N)
    r=$(api "$VLLM_BASE_URL/v1/chat/completions" -H "content-type: application/json" -d @"$W/p$(( $1 % 5 )).json"); t1=$(date +%s.%N)
    echo "$1 $(echo "$r" | jq -r .usage.completion_tokens) $(echo "$t1 - $t0" | bc)"; }
  export -f one api; export W="$w" VLLM_BASE_URL VLLM_API_KEY
  for C in 1 4 8; do
    N=$(( C == 1 ? 5 : (2 * C < 10 ? 10 : 2 * C) ))
    a0=$(metric $A); d0=$(metric $D); t0=$(date +%s.%N)
    seq 0 $((N - 1)) | xargs -P "$C" -I{} bash -c 'one {}' > "$w/c$C.txt"
    t1=$(date +%s.%N); a1=$(metric $A); d1=$(metric $D)
    awk -v C="$C" -v N="$N" -v T="$(echo "$t1 - $t0" | bc)" -v a=$((a1 - a0)) -v d=$((d1 - d0)) -v L="$label" \
      '{tok += $2; r += $2 / $3} END {printf "  %-10s conc=%d reqs=%-3d wall=%6.1fs  total=%6.1f tok/s  per-req=%5.1f tok/s  draft-acc=%5.1f%%\n", L, C, N, T, tok / T, r / NR, (d > 0 ? 100 * a / d : 0)}' "$w/c$C.txt"
  done
}

case $what in
  off) natural false | tee "$out/natural_off.log"; summarize "$out/natural_off.log" | tee -a "$out/natural_off.log" ;;
  on) natural true | tee "$out/natural_on.log"; summarize "$out/natural_on.log" | tee -a "$out/natural_on.log" ;;
  greedy) greedy | tee "$out/greedy.log" ;;
  conc) conc | tee "$out/conc.log" ;;
  all) "$0" "$label" off "$runs"; "$0" "$label" on "$runs"; "$0" "$label" greedy; "$0" "$label" conc ;;
  *) die "unknown benchmark $what" ;;
esac
