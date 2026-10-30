#!/usr/bin/env bash
# usage: serve.sh up {gen|nospec|mtp} [MODEL_PATH] | serve.sh down | serve.sh status
# Starts one vLLM container on this host. With SERVER_MODE=external it only checks VLLM_BASE_URL.
. "$(dirname "$0")/common.sh"

wait_ready() {
  local t0
  t0=$(date +%s)
  for _ in $(seq 1 "${SERVER_START_TIMEOUT_TRIES:-120}"); do
    api -f -o /dev/null "$VLLM_BASE_URL/v1/models" 2>/dev/null && { log "server ready in $(( $(date +%s) - t0 ))s"; return 0; }
    if [ "$SERVER_MODE" = local ] && ! running "$SERVER_CONTAINER"; then
      docker logs "$SERVER_CONTAINER" 2>&1 | tail -20
      die "$SERVER_CONTAINER exited"
    fi
    sleep 10
  done
  die "server not ready at $VLLM_BASE_URL"
}

up() {
  local profile=$1 model=${2:-$MODEL_DIR}
  if [ "$SERVER_MODE" = external ]; then
    log "SERVER_MODE=external: using $VLLM_BASE_URL as is (profile $profile is not applied)"
    wait_ready
    return
  fi
  model="$(abspath "$model")"
  [ -f "$model/config.json" ] || die "no checkpoint at $model"
  require_gpu_free "$SERVER_CONTAINER"
  docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
  local args=(--served-model-name "$SERVED_MODEL_NAME" --trust-remote-code)
  case $profile in
    gen)
      args+=(--gpu-memory-utilization "${GEN_GPU_MEM_UTIL:-0.80}" --max-model-len "${GEN_MAX_MODEL_LEN:-8192}"
             --max-num-seqs "${GEN_MAX_NUM_SEQS:-128}" --max-num-batched-tokens 8192 --enable-prefix-caching
             --limit-mm-per-prompt '{"image":0,"video":0}') ;;
    nospec|mtp)
      [ "$profile" = mtp ] && args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS:-3}}")
      args+=(--gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}" --max-model-len "${MAX_MODEL_LEN:-131072}"
             --max-num-seqs "${MAX_NUM_SEQS:-8}" --max-cudagraph-capture-size "${CUDAGRAPH_MAX_SIZE:-32}"
             --reasoning-parser qwen3 --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192) ;;
    *) die "unknown profile $profile" ;;
  esac
  # shellcheck disable=SC2206
  local extra=(${VLLM_EXTRA_ARGS:-})
  log "starting $SERVER_CONTAINER ($profile) from $model on port $VLLM_PORT"
  docker run -d --name "$SERVER_CONTAINER" --gpus "$(gpus_arg)" --ipc host --shm-size 8g \
    -p "$VLLM_PORT:8000" -v "$model:/model:ro" -v "$CACHE_DIR/vllm:/root/.cache" \
    -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e VLLM_API_KEY \
    "$VLLM_IMAGE" /model "${args[@]}" "${extra[@]}" >/dev/null
  wait_ready
}

down() {
  [ "$SERVER_MODE" = external ] && { log "SERVER_MODE=external: nothing to stop"; return; }
  docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 && log "stopped $SERVER_CONTAINER" || true
}

case "${1:-}" in
  up) shift; up "$@" ;;
  down) down ;;
  status) api -f "$VLLM_BASE_URL/v1/models" | jq -r '.data[].id' ;;
  *) echo "usage: $0 up {gen|nospec|mtp} [MODEL_PATH] | down | status" >&2; exit 2 ;;
esac
