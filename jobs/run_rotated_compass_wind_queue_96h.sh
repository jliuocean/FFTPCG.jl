#!/usr/bin/env bash

set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: $0 DIRECTION [DIRECTION ...]" >&2
    exit 2
fi

project_root="${HOME}/Projects/FFTPCG.jl"
experiment="${project_root}/experiments/buoyancy_flow_mamd_smooth_rotated_ccw_H200.jl"
log_dir="${project_root}/jobs/logs/rotated_compass_winds_96h"
mkdir -p "${log_dir}"

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
    echo "Waiting for PID ${WAIT_FOR_PID} before starting this queue"
    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
        sleep 30
    done
    echo "PID ${WAIT_FOR_PID} finished; starting queue"
fi

for direction in "$@"; do
    case "${direction}" in
        S|SE|E|NE|N|NW|W|SW) ;;
        *)
            echo "Invalid wind direction: ${direction}" >&2
            exit 2
            ;;
    esac

    lower_direction="${direction,,}"
    log_file="${log_dir}/${direction}.log"
    echo "Starting ${direction} wind experiment; log: ${log_file}"
    env JULIA_LOAD_PATH="${HOME}/Projects/Oceananigans.jl:${HOME}/Projects/FFTPCG.jl/env/h200:@stdlib" \
        julia --startup-file=no "${experiment}" \
        --arch GPU \
        --simulation-only \
        --wind-case "${lower_direction}" \
        --wind-speed 6 \
        --wind-start-time 43200 \
        --wind-ramp-time 43200 \
        --stop-time 345600 \
        --checkpoint-interval 21600 \
        --pickup false \
        >"${log_file}" 2>&1
    echo "Completed ${direction} wind experiment"
done

echo "Completed queue: $*"
