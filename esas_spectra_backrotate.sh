#!/usr/bin/env bash
set -Eeuo pipefail

# This script runs (mos|pn)spectra -> (mos|pn)back -> protonscale -> rotdet2sky
# for MOS1, MOS2, and PN for a single energy band [ELO, EHI] (in eV).
#
# Required environment variables (examples):
#   export ELO=350 EHI=1100
#   export M1="mos1S001"  M2="mos2S002"  PN="pnS003"
#   export M1ON='T T F T T F T'          # MOS1 CCD selection from emanom
#   export M2ON='T T T T F T T'          # MOS2 CCD selection from emanom
#   export PNON='T T T T'                # PN quad selection
#
# Region files (detector coordinates) expected in current directory by default:
#   regmos1.txt regmos2.txt regpn.txt
#
# Notes:
# - We enable point-source removal masks in mosspectra/pnspectra (withsrcrem=yes).
# - We keep intermediate files (keepinterfiles=yes), as per your reference.

readonly REQUIRED_ENV_VARS=(ELO EHI M1 M2 PN M1ON M2ON PNON)

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

# ----- MOS2 -----
run_command_with_log \
  mosspectra_2.log \
  mosspectra \
  eventfile="${M2}-allevc.fits" \
  keepinterfiles=yes \
  withregion=yes \
  regionfile=regmos2.txt \
  pattern=12 \
  withsrcrem=yes \
  maskdet="${M2}-bkgregtdet.fits" \
  masksky="${M2}-bkgregtsky.fits" \
  elow="${ELO}" \
  ehigh="${EHI}" \
  ccds="${M2ON}" \
  -V=7

run_command_with_log \
  mosback_2.log \
  mosback \
  inspecfile="${M2}-fovt.pi" \
  elow="${ELO}" \
  ehigh="${EHI}" \
  ccds="${M2ON}"

run_command_capture \
  protonscale \
  mode=1 \
  maskfile="${M2}-fovimspdet.fits" \
  specfile="${M2}-fovt.pi"

run_command_capture \
  rotdet2sky \
  intemplate="${M2}-fovimsky-${ELO}-${EHI}.fits" \
  inimage="${M2}-bkgimdet-${ELO}-${EHI}.fits" \
  outimage="${M2}-bkgimsky-${ELO}-${EHI}.fits" \
  withdetxy=false \
  withskyxy=false

# ----- MOS1 -----
run_command_with_log \
  mosspectra_1.log \
  mosspectra \
  eventfile="${M1}-allevc.fits" \
  keepinterfiles=yes \
  withregion=yes \
  regionfile=regmos1.txt \
  pattern=12 \
  withsrcrem=yes \
  maskdet="${M1}-bkgregtdet.fits" \
  masksky="${M1}-bkgregtsky.fits" \
  elow="${ELO}" \
  ehigh="${EHI}" \
  ccds="${M1ON}" \
  -V=7

run_command_with_log \
  mosback_1.log \
  mosback \
  inspecfile="${M1}-fovt.pi" \
  elow="${ELO}" \
  ehigh="${EHI}" \
  ccds="${M1ON}"

run_command_capture \
  protonscale \
  mode=1 \
  maskfile="${M1}-fovimspdet.fits" \
  specfile="${M1}-fovt.pi"

run_command_capture \
  rotdet2sky \
  intemplate="${M1}-fovimsky-${ELO}-${EHI}.fits" \
  inimage="${M1}-bkgimdet-${ELO}-${EHI}.fits" \
  outimage="${M1}-bkgimsky-${ELO}-${EHI}.fits" \
  withdetxy=false \
  withskyxy=false

# ----- PN -----
run_command_with_log \
  pnspectra_0.log \
  pnspectra \
  eventfile="${PN}-allevc.fits" \
  ootevtfile="${PN}-allevcoot.fits" \
  keepinterfiles=yes \
  withregion=yes \
  regionfile=regpn.txt \
  pattern=0 \
  withsrcrem=yes \
  maskdet="${PN}-bkgregtdet.fits" \
  masksky="${PN}-bkgregtsky.fits" \
  elow="${ELO}" \
  ehigh="${EHI}" \
  quads="${PNON}" \
  -V=7

run_command_with_log \
  pnback_0.log \
  pnback \
  inspecfile="${PN}-fovt.pi" \
  inspecoot="${PN}-fovtoot.pi" \
  elow="${ELO}" \
  ehigh="${EHI}" \
  quads="${PNON}"

run_command_capture \
  protonscale \
  mode=1 \
  maskfile="${PN}-fovimspdet.fits" \
  specfile="${PN}-fovt.pi"

run_command_capture \
  rotdet2sky \
  intemplate="${PN}-fovimsky-${ELO}-${EHI}.fits" \
  inimage="${PN}-bkgimdet-${ELO}-${EHI}.fits" \
  outimage="${PN}-bkgimsky-${ELO}-${EHI}.fits" \
  withdetxy=false \
  withskyxy=false

printf 'Done. Generated spectra/background products for %s-%s eV.\n' "${ELO}" "${EHI}"
