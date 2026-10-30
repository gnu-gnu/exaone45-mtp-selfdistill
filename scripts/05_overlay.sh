#!/usr/bin/env bash
# usage: 05_overlay.sh NAME [MTP_SAFETENSORS]
# Builds $WORK_DIR/serve/NAME: MODEL_DIR with only the mtp.* tensors replaced (default ckpt/NAME/mtp.safetensors).
# Unchanged files are hard links when WORK_DIR and MODEL_DIR share a filesystem, copies otherwise (about 33 GB).
. "$(dirname "$0")/common.sh"
name=${1:?usage: 05_overlay.sh NAME [MTP_SAFETENSORS]}
mtp="$(abspath "${2:-$CKPT_DIR/$name/mtp.safetensors}")"
[ -f "$mtp" ] || die "$mtp not found"
out="$SERVE_DIR/$name"
rm -rf "$out"
mkdir -p "$out"
shards=$(jq -r '.weight_map | to_entries[] | select(.key | startswith("mtp.")) | .value' "$MODEL_DIR/model.safetensors.index.json" | sort -u)
if cp -al "$MODEL_DIR/." "$out/" 2>/dev/null; then
  log "linked $MODEL_DIR into serve/$name"
else
  rm -rf "$out"; mkdir -p "$out"
  log "MODEL_DIR is on another filesystem; copying instead of linking"
  cp -a "$MODEL_DIR/." "$out/"
fi
for s in $shards; do rm -f "$out/$s"; done
docker run --rm --user "$HOST_UID:$HOST_GID" -e HOME=/tmp \
  -v "$MODEL_DIR:/model:ro" -v "$(dirname "$mtp"):/mtp:ro" -v "$out:/out" -v "$ROOT/src:/src:ro" \
  --entrypoint python3 "$VLLM_IMAGE" -I /src/build_overlay.py --shards-only \
  --src /model --mtp "/mtp/$(basename "$mtp")" --out /out
log "serve/$name ready; start it with: scripts/serve.sh up mtp $out"
