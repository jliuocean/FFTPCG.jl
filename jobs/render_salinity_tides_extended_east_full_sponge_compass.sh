#!/usr/bin/env bash

set -euo pipefail

project_root="/home/jliu1/Projects/FFTPCG.jl"
oceananigans_root="/home/jliu1/Projects/Oceananigans.jl"
data_root="/mnt/workdir/jliu1/FFTPCG/Data/SalinityTides_compass_Qs457_Qw343_dx100_dz1_xe30_es3_full_v_sponge"
output_dir="$data_root/animations"
log_dir="$project_root/jobs/logs/salinity_tides_extended_east_full_sponge"

mkdir -p "$output_dir" "$log_dir"
cd "$project_root"

env JULIA_LOAD_PATH="$oceananigans_root:$project_root/env/h200:@stdlib" \
    julia --startup-file=no \
    experiments/plot_salinity_tides_compass_9case.jl \
    --data-root "$data_root" \
    --output-dir "$output_dir" \
    --framerate 6 \
    > "$log_dir/render_surface.log" 2>&1

env JULIA_LOAD_PATH="$oceananigans_root:$project_root/env/h200:@stdlib" \
    julia --startup-file=no \
    experiments/plot_salinity_tides_compass_epv_velocity.jl \
    --data-root "$data_root" \
    --output-dir "$output_dir" \
    --framerate 6 \
    --frame-stride 1 \
    --max-frames 0 \
    --sample-count 5 \
    > "$log_dir/render_diagnostics.log" 2>&1

touch "$output_dir/.render_complete"
