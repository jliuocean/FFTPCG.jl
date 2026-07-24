#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 DIRECTION" >&2
    exit 2
fi

direction="${1^^}"
case "${direction}" in
    S|SE|E|NE|N|NW|W|SW) ;;
    *)
        echo "Invalid wind direction: ${direction}" >&2
        exit 2
        ;;
esac

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
    echo "Waiting for PID ${WAIT_FOR_PID} before rendering ${direction}"
    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
        sleep 30
    done
fi

project_root="${HOME}/Projects/FFTPCG.jl"
experiment="${project_root}/experiments/buoyancy_flow_mamd_smooth_rotated_ccw_H200.jl"
log_dir="${project_root}/jobs/logs/rotated_compass_winds_96h"
mkdir -p "${log_dir}"

echo "Starting CPU-only ${direction} animation rendering"
exec nice -n 10 ionice -c 2 -n 7 \
    env JULIA_LOAD_PATH="${HOME}/Projects/Oceananigans.jl:${HOME}/Projects/FFTPCG.jl/env/h200:@stdlib" \
    julia --startup-file=no "${experiment}" \
    --arch CPU \
    --plot-only \
    --wind-case "${direction,,}" \
    --wind-speed 6 \
    --wind-start-time 43200 \
    --wind-ramp-time 43200 \
    --stop-time 345600
