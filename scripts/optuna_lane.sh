#!/usr/bin/env bash
# Usage: lane.sh <name> <study>[:device[:enqueue_from]] ...  (runs studies one after another)
cd "$(dirname "$0")/.."
name=$1; shift
for spec in "$@"; do
  IFS=: read -r study device enqueue <<< "$spec"
  extra=""; [ -n "$enqueue" ] && extra="--enqueue-from $enqueue"
  echo "[$(date '+%F %T')] start $study (${device:-cpu}) $extra"
  PYTHONPATH=src .venv/Scripts/python.exe -m benchmarks.official.bohb_tune --study "$study" --device "${device:-cpu}" $extra 2>&1 | grep -v -i warn
  echo "[$(date '+%F %T')] end $study"
done
