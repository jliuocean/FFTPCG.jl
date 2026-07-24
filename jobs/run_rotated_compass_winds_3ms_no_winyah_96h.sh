#!/usr/bin/env bash

set -euo pipefail

project_root="${HOME}/Projects/FFTPCG.jl"
experiment="${project_root}/experiments/buoyancy_flow_mamd_smooth_rotated_ccw_H200.jl"
log_dir="${project_root}/jobs/logs/rotated_compass_winds_3ms_no_winyah_96h"
mkdir -p "${log_dir}"

directions=(S SE E NE N NW W SW)

for direction in "${directions[@]}"; do
    lower_direction="${direction,,}"
    log_file="${log_dir}/${direction}.log"

    echo "Starting ${direction}: Santee 500 m3/s, no Winyah Bay, wind 3 m/s"
    env JULIA_LOAD_PATH="${HOME}/Projects/Oceananigans.jl:${HOME}/Projects/FFTPCG.jl/env/h200:@stdlib" \
        julia --startup-file=no "${experiment}" \
        --arch GPU \
        --simulation-only \
        --no-winyah-bay \
        --santee-discharge 500 \
        --winyah-discharge 0 \
        --wind-case "${lower_direction}" \
        --wind-speed 3 \
        --wind-start-time 43200 \
        --wind-ramp-time 43200 \
        --stop-time 345600 \
        --checkpoint-interval 21600 \
        --pickup false \
        >"${log_file}" 2>&1

    echo "Completed ${direction} no-Winyah experiment"
done

echo "Completed all eight 3 m/s no-Winyah experiments"
