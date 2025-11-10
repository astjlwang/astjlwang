#!/usr/bin/env bash

set -Eeuo pipefail

readonly REQUIRED_ENV_VARS=(ELO EHI M1 M2)

missing_vars=()
for var in "${REQUIRED_ENV_VARS[@]}"; do
  if [[ -z "${!var-}" ]]; then
    missing_vars+=("$var")
  fi
done

if (( ${#missing_vars[@]} )); then
  printf 'Error: missing required environment variable(s): %s\n' "${missing_vars[*]}" >&2
  exit 1
fi

emit_last_line_from_file() {
  local file_path=$1

  if [[ -s "$file_path" ]]; then
    tail -n 1 "$file_path"
  elif [[ ! -e "$file_path" ]]; then
    printf 'Warning: expected log file "%s" not found.\n' "$file_path" >&2
  fi
}

emit_last_line_from_buffer() {
  local buffer=$1

  if [[ -n "$buffer" ]]; then
    printf '%s\n' "$buffer" | tail -n 1
  fi
}

run_command_with_log() {
  local log_file=$1
  shift

  "$@" &> "$log_file"
  emit_last_line_from_file "$log_file"
}

run_command_capture() {
  local output

  if ! output=$("$@" 2>&1); then
    printf '%s\n' "$output" >&2
    return 1
  fi

  emit_last_line_from_buffer "$output"
}

# Step 1
run_command_with_log \
  mosspectra_2.log \
  mosspectra \
  eventfile=mos2S002-allevc.fits \
  keepinterfiles=no \
  withregion=yes \
  regionfile=poly2.txt \
  pattern=12 \
  withsrcrem=no \
  elow="$ELO" \
  ehigh="$EHI" \
  ccds="T T T T F T T" \
  -V=7

# Step 2
run_command_with_log \
  mosback_2.log \
  mosback \
  inspecfile=mos2S002-fovt.pi \
  elow="$ELO" \
  ehigh="$EHI" \
  ccds="T T T T F T T"

# Step 3
run_command_capture \
  protonscale \
  mode=1 \
  maskfile="${M2}-fovimspdet.fits" \
  specfile="${M2}-fovt.pi"

# Step 4
run_command_capture \
  grppha \
  mos2S002-fovt.pi \
  mos2S002-grp.pi \
  "chkey BACKFILE mos2S002-bkg.pi & chkey RESPFILE mos2S002.rmf & chkey ANCRFILE mos2S002.arf & group min 20 & exit"

# Step 5
run_command_with_log \
  mosspectra_1.log \
  mosspectra \
  eventfile=mos1S001-allevc.fits \
  keepinterfiles=no \
  withregion=yes \
  regionfile=poly1.txt \
  pattern=12 \
  withsrcrem=no \
  elow="$ELO" \
  ehigh="$EHI" \
  ccds="T T F F T F T" \
  -V=7

# Step 6
run_command_with_log \
  mosback_1.log \
  mosback \
  inspecfile=mos1S001-fovt.pi \
  elow="$ELO" \
  ehigh="$EHI" \
  ccds="T T F F T F T"

# Step 7
run_command_capture \
  protonscale \
  mode=1 \
  maskfile="${M1}-fovimspdet.fits" \
  specfile="${M1}-fovt.pi"

# Step 8
run_command_capture \
  grppha \
  mos1S001-fovt.pi \
  mos1S001-grp.pi \
  "chkey BACKFILE mos1S001-bkg.pi & chkey RESPFILE mos1S001.rmf & chkey ANCRFILE mos1S001.arf & group min 20 & exit"

