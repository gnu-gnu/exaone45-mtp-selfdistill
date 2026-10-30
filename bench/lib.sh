#!/usr/bin/env bash
# shellcheck disable=SC2034
# The five natural prompts used in every published measurement, plus /metrics helpers.
# Sourced after scripts/common.sh. BENCH_SYSTEM adds a system prompt (empty in the published runs).
metric() { api "$VLLM_BASE_URL/metrics" | awk -v k="$1" '$1 ~ "^"k"\\{" {s+=$2} END{printf "%d", s}'; }
user_msg() {
  jq -cn --arg c "$1" --arg s "${BENCH_SYSTEM:-}" \
    '(if $s == "" then [] else [{role:"system",content:$s}] end) + [{role:"user",content:$c}]'
}
PROMPT_NAMES=(casual-ko casual-en code-review doc-en-sum code-gen)
prompt_messages() {
  local code doc
  case $1 in
    casual-ko) user_msg '이번 주말에 서울 근교로 당일치기 여행을 가려는데, 대중교통으로 갈 수 있는 곳 중에 추천해 줄 만한 데 있어?' ;;
    casual-en) user_msg 'Can you explain the difference between TCP and UDP to a junior developer, with a couple of real-world examples?' ;;
    code-review)
      code=$(cat "$ROOT/bench/inputs/code.py")
      user_msg "다음 파이썬 코드를 리뷰하고 버그나 개선점이 있으면 알려줘.

\`\`\`python
$code
\`\`\`" ;;
    doc-en-sum)
      doc=$(cat "$ROOT/bench/inputs/doc_en.md")
      user_msg "다음 vLLM 문서를 읽고 핵심 내용을 한국어로 요약해 줘.

$doc" ;;
    code-gen) user_msg 'Write a Python class implementing an LRU cache with get/put and type hints, plus a short usage example.' ;;
  esac
}
