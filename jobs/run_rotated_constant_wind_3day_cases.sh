#!/usr/bin/env bash
set -euo pipefail

model="experiments/buoyancy_flow_mamd_smooth_rotated_ccw_H200.jl"
log_dir="experiments/constant_wind_logs"
mkdir -p "${log_dir}"

run_case() {
    local wind_case="$1"
    local log_file="${log_dir}/${wind_case}_4ms_3day.out"

    env JULIA_LOAD_PATH="$HOME/Projects/Oceananigans.jl:$HOME/Projects/FFTPCG.jl/env/h200:@stdlib" \
        julia --startup-file=no "${model}" \
        --arch GPU \
        --wind-case "${wind_case}" \
        --wind-speed 4 \
        --stop-time 259200 \
        2>&1 | tee "${log_file}"
}

run_case westerly
run_case southerly
run_case northerly
