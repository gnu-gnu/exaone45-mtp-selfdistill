#!/usr/bin/env bash
# usage: train.sh NAME [train_mtp.py args...]
# Runs src/train_mtp.py on the GPU with the target model frozen; output in $WORK_DIR/ckpt/NAME, log in logs/NAME.log.
# Inside the container: /model (MODEL_DIR, read-only), /data (WORK_DIR/data, read-only), /ckpt (WORK_DIR/ckpt).
. "$(dirname "$0")/common.sh"
name=${1:?usage: train.sh NAME [train_mtp.py args...]}; shift
require_gpu_free "$TRAIN_CONTAINER"
running "$TRAIN_CONTAINER" && die "$TRAIN_CONTAINER is already running"
docker rm -f "$TRAIN_CONTAINER" >/dev/null 2>&1 || true
mem=()
[ -n "${TRAIN_MEM_LIMIT:-}" ] && mem=(--memory "$TRAIN_MEM_LIMIT" --memory-swap "$TRAIN_MEM_LIMIT")
frac=()
[ -n "${TRAIN_MEM_FRAC:-}" ] && frac=(--mem-frac "$TRAIN_MEM_FRAC")
log "train $name: $*"
docker run -d --name "$TRAIN_CONTAINER" --gpus "$(gpus_arg)" --ipc host "${mem[@]}" \
  --user "$HOST_UID:$HOST_GID" -e HOME=/tmp -e HF_HOME=/tmp/hf -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$MODEL_DIR:/model:ro" -v "$DATA_DIR:/data:ro" -v "$CKPT_DIR:/ckpt" -v "$ROOT/src:/src:ro" \
  --entrypoint python3 "$VLLM_IMAGE" -I /src/train_mtp.py --out "/ckpt/$name" "${frac[@]}" "$@" >/dev/null
# Host memory watchdog for unified-memory machines (GB10): stop the job before the host runs out.
min_free=${TRAIN_MIN_FREE_GB:-0}
while running "$TRAIN_CONTAINER"; do
  if [ "$min_free" -gt 0 ]; then
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$avail" -lt "$min_free" ]; then
      log "watchdog: MemAvailable ${avail}G < ${min_free}G, stopping $TRAIN_CONTAINER"
      docker kill "$TRAIN_CONTAINER" >/dev/null
    fi
  fi
  sleep 2
done
docker logs "$TRAIN_CONTAINER" > "$LOG_DIR/$name.log" 2>&1
rc=$(docker inspect -f '{{.State.ExitCode}}' "$TRAIN_CONTAINER")
docker rm "$TRAIN_CONTAINER" >/dev/null
log "train $name exit=$rc (log: logs/$name.log, metrics: ckpt/$name/log.jsonl)"
[ "$rc" = 0 ]
