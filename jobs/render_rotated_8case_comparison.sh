#!/usr/bin/env bash

set -euo pipefail

if (( $# < 4 || $# > 5 )); then
    echo "Usage: $0 WIND_SPEED SANTEE_DISCHARGE WINYAH_DISCHARGE PLOT_MODE [RIVER_GEOMETRY]" >&2
    exit 2
fi

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
    echo "Waiting for PID ${WAIT_FOR_PID} before starting eight-case rendering"
    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
        sleep 30
    done
fi

project_root="${HOME}/Projects/FFTPCG.jl"
experiment="${project_root}/experiments/plot_rotated_compass_8case_comparison.jl"

exec nice -n 10 ionice -c 2 -n 7 \
    env JULIA_LOAD_PATH="${HOME}/Projects/Oceananigans.jl:${HOME}/Projects/FFTPCG.jl/env/h200:@stdlib" \
    julia --startup-file=no "${experiment}" "$@"
