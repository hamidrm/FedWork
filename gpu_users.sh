#!/usr/bin/env bash
set -euo pipefail

# Build GPU UUID -> index map
declare -A UUID2IDX
while IFS=, read -r idx uuid; do
  idx="$(echo "$idx"  | xargs)"
  uuid="$(echo "$uuid" | xargs)"
  UUID2IDX["$uuid"]="$idx"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

# One row per (GPU, PID)
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits \
| while IFS=, read -r uuid pid cmd mem; do
    uuid="$(echo "$uuid" | xargs)"
    pid="$(echo "$pid"  | xargs)"
    cmd="$(echo "$cmd"  | xargs)"
    mem="$(echo "$mem"  | xargs)"     # MiB, units removed by nounits
    gpu="${UUID2IDX[$uuid]:-?}"

    # Make sure the PID still exists
    if ps -p "$pid" >/dev/null 2>&1; then
      user=$(ps -o user=   -p "$pid" | xargs)
      etime=$(ps -o etime= -p "$pid" | xargs)   # [[dd-]hh:]mm:ss since start
      etimes=$(ps -o etimes= -p "$pid" | xargs) # elapsed seconds (numeric)
      start=$(ps -o lstart= -p "$pid" | xargs)  # start timestamp

      printf "GPU=%s  PID=%s  USER=%s  ELAPSED=%s (%ss)  START=\"%s\"  MEM=%sMiB  CMD=%s\n" \
        "$gpu" "$pid" "$user" "$etime" "$etimes" "$start" "$mem" "$cmd"
    fi
  done
