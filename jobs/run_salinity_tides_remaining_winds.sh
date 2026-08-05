#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_salinity_tides_remaining_winds.sh [--wait-pid PID]

Options:
  --wait-pid PID  Wait for PID to exit before starting the queued cases.
  -h, --help      Show this help message.

If --wait-pid is omitted, the queued cases start immediately. The
OCEANANIGANS_ROOT environment variable may be used to override the default
Oceananigans.jl checkout beside this repository.
EOF
}

wait_pid=""
while (( $# > 0 )); do
    case "$1" in
        --wait-pid)
            (( $# >= 2 )) || { echo "Error: --wait-pid requires a PID." >&2; exit 2; }
            wait_pid="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$wait_pid" && ! "$wait_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --wait-pid must be a positive integer." >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
oceananigans_root="${OCEANANIGANS_ROOT:-$project_root/../Oceananigans.jl}"
model_script="$project_root/experiments/salinity_flow_mamd_tides_rotated_ccw_H200.jl"
log_dir="$project_root/jobs/logs/salinity_tides_remaining_winds"
queue_log="$log_dir/queue.log"

mkdir -p "$log_dir"
cd "$project_root"

exec >> "$queue_log" 2>&1

if [[ -n "$wait_pid" ]]; then
    echo "[$(date --iso-8601=seconds)] Waiting for PID $wait_pid."
    while kill -0 "$wait_pid" 2>/dev/null; do
        sleep 60
    done
    echo "[$(date --iso-8601=seconds)] PID $wait_pid finished; starting queued cases."
else
    echo "[$(date --iso-8601=seconds)] No wait PID specified; starting queued cases."
fi

directions=(CALM S SW W NW N NE E)

for direction in "${directions[@]}"; do
    if [[ "$direction" == "CALM" ]]; then
        wind_stress="0.0"
    else
        wind_stress="0.03"
    fi

    case_log="$log_dir/${direction}.log"
    echo "[$(date --iso-8601=seconds)] Starting $direction case (wind stress $wind_stress Pa)."

    env JULIA_LOAD_PATH="$oceananigans_root:$project_root/env/h200:@stdlib" \
        julia --startup-file=no "$model_script" \
        --arch GPU \
        --horizontal-resolution 100 \
        --vertical-levels 20 \
        --stop-time -1 \
        --wind-direction "$direction" \
        --wind-stress "$wind_stress" \
        --m2-period 44640 \
        --buoyancy-delay-cycles 1 \
        --progress-interval 20 \
        --output-interval 3600 \
        --checkpoint-interval 44640 \
        --output-without-halos \
        > "$case_log" 2>&1

    echo "[$(date --iso-8601=seconds)] Completed $direction case."
done

echo "[$(date --iso-8601=seconds)] All queued salinity-tide cases completed."
