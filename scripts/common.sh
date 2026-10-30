#!/usr/bin/env bash
# Sourced by every script. All host-specific settings come from .env; see docs/configuration.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [ ! -f "$ENV_FILE" ]; then
  echo "error: $ENV_FILE not found; copy .env.example to .env and fill it in" >&2
  exit 1
fi
# Variables already exported by the caller win over .env, so a key can be passed for one run only.
caller_env="$(export -p)"
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a
eval "$caller_env"
unset caller_env

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date +%T)] $*"; }
abspath() { case "$1" in /*) printf "%s" "$1" ;; *) printf "%s/%s" "$ROOT" "${1#./}" ;; esac; }

[ -n "${MODEL_DIR:-}" ] || die "MODEL_DIR is not set in $ENV_FILE"
MODEL_DIR="$(abspath "$MODEL_DIR")"
WORK_DIR="$(abspath "${WORK_DIR:-work}")"
RAW_DIR="$WORK_DIR/raw"
DATA_DIR="$WORK_DIR/data"
CKPT_DIR="$WORK_DIR/ckpt"
SERVE_DIR="$WORK_DIR/serve"
LOG_DIR="$WORK_DIR/logs"
BENCH_DIR="$WORK_DIR/bench"
CACHE_DIR="$(abspath "${CACHE_DIR:-$WORK_DIR/cache}")"
mkdir -p "$RAW_DIR" "$DATA_DIR" "$CKPT_DIR" "$SERVE_DIR" "$LOG_DIR" "$BENCH_DIR" "$CACHE_DIR"

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
TOOLS_IMAGE="${TOOLS_IMAGE:-exaone45-mtp-tools:0.1}"
GPU_DEVICES="${GPU_DEVICES:-all}"
VLLM_PORT="${VLLM_PORT:-18001}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:$VLLM_PORT}"
VLLM_BASE_URL="${VLLM_BASE_URL%/}"
VLLM_BASE_URL="${VLLM_BASE_URL%/v1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-exaone45-33b-fp8}"
SERVER_MODE="${SERVER_MODE:-local}"
SERVER_CONTAINER="${SERVER_CONTAINER:-exaone45-mtp-serve}"
TRAIN_CONTAINER="${TRAIN_CONTAINER:-exaone45-mtp-train}"
EXCLUSIVE_GPU="${EXCLUSIVE_GPU:-1}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
export VLLM_API_KEY="${VLLM_API_KEY:-}"

# docker --gpus needs the inner quotes when the value lists several devices.
gpus_arg() { case "$GPU_DEVICES" in *,*) printf '"%s"' "$GPU_DEVICES" ;; *) printf '%s' "$GPU_DEVICES" ;; esac; }

running() { [ -n "$(docker ps -q -f "name=^$1\$")" ]; }

require_gpu_free() { # $1: the container about to start
  [ "$EXCLUSIVE_GPU" = 1 ] || return 0
  local other
  for other in "$SERVER_CONTAINER" "$TRAIN_CONTAINER"; do
    [ "$other" = "$1" ] && continue
    if running "$other"; then die "$other is running; EXCLUSIVE_GPU=1 allows one GPU job at a time"; fi
  done
}

ensure_tools_image() {
  docker image inspect "$TOOLS_IMAGE" >/dev/null 2>&1 && return 0
  log "building $TOOLS_IMAGE"
  docker build -q -t "$TOOLS_IMAGE" -f "$ROOT/docker/tools.Dockerfile" "$ROOT/docker" >/dev/null
}

# CPU-side Python (data prep, generation client, downloads) in the pinned tools image.
run_tools() {
  ensure_tools_image
  docker run --rm --network host --user "$HOST_UID:$HOST_GID" -e HOME=/tmp -e HF_HOME=/tmp/hf \
    -e HF_TOKEN -e VLLM_API_KEY \
    -v "$ROOT/src:/src:ro" -v "$RAW_DIR:/raw" -v "$DATA_DIR:/data" -v "$MODEL_DIR:/model:ro" \
    "$TOOLS_IMAGE" "$@"
}

# Prints the AI Hub key on stdout for capture into a variable; never echo the result.
aihub_key() {
  if [ -n "${AIHUB_API_KEY:-}" ]; then
    printf '%s' "$AIHUB_API_KEY"
  elif [ -n "${AIHUB_API_KEY_FILE:-}" ]; then
    tr -d '\r\n' < "$(abspath "$AIHUB_API_KEY_FILE")"
  elif [ -n "${AIHUB_API_KEY_CMD:-}" ]; then
    bash -c "$AIHUB_API_KEY_CMD" | tr -d '\r\n'
  else
    return 1
  fi
}

aihub_key_source() {
  if [ -n "${AIHUB_API_KEY:-}" ]; then echo "AIHUB_API_KEY"
  elif [ -n "${AIHUB_API_KEY_FILE:-}" ]; then echo "AIHUB_API_KEY_FILE"
  elif [ -n "${AIHUB_API_KEY_CMD:-}" ]; then echo "AIHUB_API_KEY_CMD"
  else echo "not set"; fi
}

# curl against the vLLM server, adding the bearer token without putting it on the command line.
api() {
  if [ -n "$VLLM_API_KEY" ]; then
    curl -sS -H @<(printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY") "$@"
  else
    curl -sS "$@"
  fi
}
