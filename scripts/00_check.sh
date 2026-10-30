#!/usr/bin/env bash
# Checks .env, host tools, the model checkpoint, GPU access and the AI Hub key source. Prints no secrets.
. "$(dirname "$0")/common.sh"
ok() { echo "  ok    $*"; }
warn() { echo "  warn  $*"; }
fail() { echo "  FAIL  $*"; bad=1; }
bad=0
echo "configuration ($ENV_FILE)"
echo "  MODEL_DIR=$MODEL_DIR"
echo "  WORK_DIR=$WORK_DIR"
echo "  VLLM_IMAGE=$VLLM_IMAGE  GPU_DEVICES=$GPU_DEVICES  SERVER_MODE=$SERVER_MODE  VLLM_BASE_URL=$VLLM_BASE_URL"
echo "  VLLM_API_KEY: $([ -n "$VLLM_API_KEY" ] && echo set || echo 'not set')  HF_TOKEN: $([ -n "${HF_TOKEN:-}" ] && echo set || echo 'not set')  AI Hub key: $(aihub_key_source)"
echo "checks"
for t in docker curl jq bc sha256sum awk; do command -v "$t" >/dev/null && ok "$t" || fail "$t not found"; done
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  n=$(jq '[.weight_map | keys[] | select(startswith("mtp."))] | length' "$MODEL_DIR/model.safetensors.index.json")
  [ "$n" -gt 0 ] && ok "checkpoint has $n mtp.* tensors" || fail "checkpoint has no mtp.* tensors"
  rev=$(head -1 "$MODEL_DIR/.cache/huggingface/download/config.json.metadata" 2>/dev/null || true)
  case $rev in
    69570ad5d9a5159816ef267f2f4d37e36447f1be) ok "EXAONE-4.5-33B-FP8 revision ${rev:0:8} (as published)" ;;
    "") warn "checkpoint revision unknown (not downloaded with hf download --local-dir)" ;;
    *) warn "checkpoint revision ${rev:0:8} differs from 69570ad5 used for the published results" ;;
  esac
  echo "$(sha256sum "$MODEL_DIR/tokenizer.json" | cut -c1-64)" | grep -q '^0bd798efa30739e209d51f36cfc2f0a636711e37ad9d69b77e4e5c8ca5f09fab$' \
    && ok "tokenizer.json matches" || warn "tokenizer.json differs; prompt checksums will not match"
else
  fail "no checkpoint at MODEL_DIR=$MODEL_DIR"
fi
if [ "$SERVER_MODE" = local ] || [ "${CHECK_GPU:-1}" = 1 ]; then
  if docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    gpus=$(docker run --rm --gpus "$(gpus_arg)" --entrypoint nvidia-smi "$VLLM_IMAGE" -L 2>&1) \
      && ok "GPU visible: $(echo "$gpus" | head -3 | paste -sd';')" || fail "docker --gpus $GPU_DEVICES failed: $(echo "$gpus" | tail -1)"
  else
    warn "$VLLM_IMAGE not pulled yet (docker pull $VLLM_IMAGE)"
  fi
fi
if [ "$SERVER_MODE" = external ]; then
  api -f -o /dev/null "$VLLM_BASE_URL/v1/models" && ok "server reachable at $VLLM_BASE_URL" || fail "no server at $VLLM_BASE_URL"
fi
for c in "$SERVER_CONTAINER" "$TRAIN_CONTAINER"; do running "$c" && warn "$c is running"; done
if [ "$(aihub_key_source)" = "not set" ]; then
  warn "AI Hub key not set: 01_download.sh needs it unless SKIP_AIHUB=1"
else
  k="$(aihub_key 2>/dev/null)" && [ -n "$k" ] && ok "AI Hub key readable from $(aihub_key_source)" || fail "AI Hub key source $(aihub_key_source) returned nothing"
  unset k
fi
[ "$bad" = 0 ] && echo "all required checks passed" || { echo "some checks failed"; exit 1; }
