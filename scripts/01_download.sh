#!/usr/bin/env bash
# Download the prompt sources at pinned revisions into $WORK_DIR/raw and check them against data/SHA256SUMS.raw.
# AI Hub 271 needs an approved application and an API key (docs/configuration.md); SKIP_AIHUB=1 skips it.
. "$(dirname "$0")/common.sh"

hf_get() { # repo revision dir [include...]
  local repo=$1 rev=$2 dir=$3 inc=()
  shift 3
  for p in "$@"; do inc+=(--include "$p"); done
  log "hf: $repo@${rev:0:8} -> raw/$dir"
  run_tools hf download --quiet --repo-type dataset "$repo" --revision "$rev" --local-dir "/raw/$dir" "${inc[@]}" >/dev/null
}

hf_get allenai/WildChat-1M 7d6490e462285cf85d91eabea0f9a954fbddcd1f wildchat "data/train-0000[0-3]-of-00014.parquet"
hf_get maywell/koVast b0704da17669a3f32b5f11bb11c5b27620d27677 kovast "data/train-00000-of-00003-*.parquet"
hf_get m-a-p/CodeFeedback-Filtered-Instruction a08c213a9748c66c15d0225814be80a2e77adf4a codefeedback CodeFeedback-Filtered-Instruction.jsonl
hf_get nlpai-lab/kullm-v2 cddcb73c259269928e974e0ce141f123eb068030 kullm kullm-v2.jsonl
hf_get beomi/KoAlpaca-v1.1a 03d3122c530b1e47195c08a3d851eeadddad9689 koalpaca "data/train-00000-of-00001-*.parquet"

if [ "${SKIP_AIHUB:-0}" = 1 ]; then
  log "SKIP_AIHUB=1: skipping AI Hub 271"
elif [ -f "$RAW_DIR/aihub_emodial/emodial.xlsx" ]; then
  log "AI Hub 271: raw/aihub_emodial/emodial.xlsx already present"
else
  key="$(aihub_key)" || die "AI Hub key not configured; set AIHUB_API_KEY, AIHUB_API_KEY_FILE or AIHUB_API_KEY_CMD"
  [ -n "$key" ] || die "AI Hub key source ($(aihub_key_source)) returned an empty value"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  log "AI Hub 271: downloading (key from $(aihub_key_source))"
  code=$(curl -sS -L -o "$tmp/download.tar" -H @<(printf 'apikey:%s\n' "$key") -w '%{http_code}' \
    "https://api.aihub.or.kr/down/0.6/271.do?fileSn=50327")
  unset key
  if [ "$code" != 200 ] || ! tar -tf "$tmp/download.tar" >/dev/null 2>&1; then
    echo "AI Hub response (HTTP $code):" >&2
    head -c 500 "$tmp/download.tar" >&2
    echo >&2
    die "AI Hub download failed; check that the dataset application is approved and the key is valid"
  fi
  mkdir -p "$tmp/x"
  tar -xf "$tmp/download.tar" -C "$tmp/x" --no-same-owner --no-same-permissions
  # Large AI Hub files arrive split as NAME.partN; join them like aihubshell does.
  find "$tmp/x" -name '*.part[0-9]*' | sed 's/\.part[0-9]*$//' | sort -u | while read -r base; do
    find "$(dirname "$base")" -maxdepth 1 -name "$(basename "$base").part*" -print0 | sort -zV | xargs -0 cat > "$base"
    rm -f "$base".part*
  done
  xlsx="$(find "$tmp/x" -name '*.xlsx' | head -1)"
  [ -n "$xlsx" ] || die "no .xlsx in the AI Hub archive"
  mkdir -p "$RAW_DIR/aihub_emodial"
  cp "$xlsx" "$RAW_DIR/aihub_emodial/emodial.xlsx"
fi

log "checking raw checksums"
sums="$ROOT/data/SHA256SUMS.raw"
if [ "${SKIP_AIHUB:-0}" = 1 ]; then
  sums="$(mktemp)"
  grep -v aihub_emodial "$ROOT/data/SHA256SUMS.raw" > "$sums"
fi
(cd "$RAW_DIR" && sha256sum --quiet -c "$sums") && log "raw data matches data/SHA256SUMS.raw"
