#!/usr/bin/env bash

set -euo pipefail

project_root="/home/jliu1/Projects/FFTPCG.jl"
oceananigans_root="/home/jliu1/Projects/Oceananigans.jl"
model_script="$project_root/experiments/salinity_flow_mamd_tides_rotated_ccw_H200.jl"
trim_script="$project_root/experiments/trim_jld2_output_to_checkpoint.jl"
run_root="/mnt/workdir/jliu1/FFTPCG/Data/SalinityTides_compass_Qs457_Qw343_dx100_dz1_xe30_es3_full_v_sponge"
log_dir="$project_root/jobs/logs/salinity_tides_extended_east_full_sponge"
queue_log="$log_dir/queue.log"

mkdir -p "$run_root" "$log_dir"
cd "$project_root"

exec >> "$queue_log" 2>&1

echo "[$(date --iso-8601=seconds)] Starting remaining extended-east, full-v-sponge cases."

# SW has already been completed with these numerical settings. Keep a
# convenient reference beside the remaining cases without copying its data.
completed_sw="/mnt/workdir/jliu1/FFTPCG/Data/SalinityTides_Qs457_Qw343_dx100p0_dz1p0_xe30p0km_es3p0km_cg0p5_tauSW0p03_delay1p0M2_T74p4h"
if [[ ! -e "$run_root/SW" ]]; then
    ln -s "$completed_sw" "$run_root/SW"
fi

directions=(CALM S SE E NE N NW W)

for direction in "${directions[@]}"; do
    if [[ "$direction" == "CALM" ]]; then
        wind_stress="0.0"
    else
        wind_stress="0.03"
    fi

    output_dir="$run_root/$direction"
    case_log="$log_dir/${direction}.log"
    complete_marker="$output_dir/.complete"

    if [[ -f "$complete_marker" ]]; then
        echo "[$(date --iso-8601=seconds)] Skipping completed $direction case."
        continue
    fi

    pickup="false"
    if [[ -s "$output_dir/instantaneous_fields.jld2" ]] &&
       compgen -G "$output_dir/checkpoint_iteration*.jld2" > /dev/null; then
        latest_checkpoint="$(
            find "$output_dir" -maxdepth 1 -type f \
                -name 'checkpoint_iteration*.jld2' -print |
            sort -V |
            tail -n 1
        )"
        backup_file="$output_dir/instantaneous_fields.pre_restart_backup.jld2"
        if [[ ! -f "$backup_file" ]]; then
            cp --reflink=auto "$output_dir/instantaneous_fields.jld2" \
                "$backup_file"
        fi
        env JULIA_LOAD_PATH="$oceananigans_root:$project_root/env/h200:@stdlib" \
            julia --startup-file=no "$trim_script" \
            "$output_dir/instantaneous_fields.jld2" "$latest_checkpoint"
        pickup="$latest_checkpoint"
        echo "[$(date --iso-8601=seconds)] Resuming $direction from $latest_checkpoint."
    fi

    echo "[$(date --iso-8601=seconds)] Starting $direction case (wind stress $wind_stress Pa)."

    env JULIA_LOAD_PATH="$oceananigans_root:$project_root/env/h200:@stdlib" \
        julia --startup-file=no "$model_script" \
        --arch GPU \
        --skip-setup-plot \
        --horizontal-resolution 100 \
        --vertical-levels 20 \
        --eastern-boundary-x 30000 \
        --east-sponge-width 3000 \
        --east-sponge-timescale 1800 \
        --east-gravity-wave-speed 0.5 \
        --east-tangential-sponge-factor 1.0 \
        --stop-time -1 \
        --pickup "$pickup" \
        --wind-direction "$direction" \
        --wind-stress "$wind_stress" \
        --m2-period 44640 \
        --buoyancy-delay-cycles 1 \
        --progress-interval 20 \
        --output-interval 3600 \
        --checkpoint-interval 44640 \
        --output-without-halos \
        --output-dir "$output_dir" \
        > "$case_log" 2>&1

    touch "$complete_marker"
    echo "[$(date --iso-8601=seconds)] Completed $direction case."
done

echo "[$(date --iso-8601=seconds)] All remaining cases completed."
