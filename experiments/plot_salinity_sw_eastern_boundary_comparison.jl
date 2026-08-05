using Oceananigans
using Oceanostics
using CairoMakie
using ArgParse
using Statistics
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Grids: inactive_cell

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--old-data-dir"
            help = "Original SW-case directory"
            arg_type = String
            required = true
        "--new-data-dir"
            help = "Revised eastern-boundary SW-case directory with the strong tangential sponge"
            arg_type = String
            required = true
        "--weak-data-dir"
            help = "Optional revised SW case with a weaker tangential sponge"
            arg_type = String
            default = ""
        "--output-dir"
            help = "Directory for comparison animations"
            arg_type = String
            required = true
        "--framerate"
            help = "Animation frame rate"
            arg_type = Int
            default = 6
        "--frame-stride"
            help = "Use every Nth common snapshot"
            arg_type = Int
            default = 1
        "--sample-count"
            help = "Number of representative frames used for color limits"
            arg_type = Int
            default = 7
    end
    return parse_args(settings)
end

args = parse_commandline()
const OLD_DIR = abspath(args["old-data-dir"])
const NEW_DIR = abspath(args["new-data-dir"])
const WEAK_DIR = isempty(args["weak-data-dir"]) ? "" :
                 abspath(args["weak-data-dir"])
const OUTPUT_DIR = abspath(args["output-dir"])
const FRAMERATE = args["framerate"]
const FRAME_STRIDE = args["frame-stride"]
const SAMPLE_COUNT = args["sample-count"]

FRAMERATE > 0 || error("--framerate must be positive")
FRAME_STRIDE > 0 || error("--frame-stride must be positive")
SAMPLE_COUNT > 0 || error("--sample-count must be positive")
mkpath(OUTPUT_DIR)

const CASES = isempty(WEAK_DIR) ? ("old", "new") :
              ("old", "strong", "weak")
const CASE_TITLES = Dict(
    "old" => "Original eastern boundary",
    "new" => "Extended eastern boundary + sponge",
    "strong" => "Extended boundary + full v sponge",
    "weak" => "Extended boundary + 25% v sponge",
)
const DATA_DIRS = isempty(WEAK_DIR) ?
    Dict("old" => OLD_DIR, "new" => NEW_DIR) :
    Dict("old" => OLD_DIR, "strong" => NEW_DIR, "weak" => WEAK_DIR)
const COMPARISON_TAG = isempty(WEAK_DIR) ?
    "eastern_boundary_comparison" :
    "eastern_boundary_3case_comparison"

data_file(case) = joinpath(DATA_DIRS[case], "instantaneous_fields.jld2")
for case in CASES
    isfile(data_file(case)) || error("Missing data file: $(data_file(case))")
end

function open_series(field)
    Dict(case => FieldTimeSeries(data_file(case), field) for case in CASES)
end

u_data = open_series("u")
v_data = open_series("v")
w_data = open_series("w")
S_data = open_series("S")

# Compare snapshots at the same saved times.
reference_times = S_data[first(CASES)].times
common_times = Float64[]
case_indices = Dict(case => Int[] for case in CASES)
for time in reference_times
    indices = Dict(
        case => findfirst(t -> isapprox(t, time; atol = 1e-6, rtol = 0),
                          S_data[case].times)
        for case in CASES
    )
    any(isnothing, values(indices)) && continue
    push!(common_times, time)
    for case in CASES
        push!(case_indices[case], indices[case]::Int)
    end
end

isempty(common_times) && error("The cases have no common saved times")
selection = collect(1:FRAME_STRIDE:length(common_times))

const gravitational_acceleration = 9.81
const haline_contraction = 7.8e-4
const reference_salinity = 34.0
const f₀ = 8e-5

mutable struct DiagnosticWorkspace
    grid
    u
    v
    w
    b
    epv
    u_center
    v_center
    active
    epv_valid
    x
    y
    z
end

function DiagnosticWorkspace(grid)
    Nx, Ny, Nz = size(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    b = CenterField(grid)
    model = (
        grid = grid,
        velocities = (u = u, v = v, w = w),
        tracers = (b = b,),
        coriolis = FPlane(f = f₀),
    )
    epv = Field(@at (Center, Center, Center) ErtelPotentialVorticity(model))
    u_center = Field(@at (Center, Center, Center) u)
    v_center = Field(@at (Center, Center, Center) v)

    active = falses(Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        active[i, j, k] = !inactive_cell(i, j, k, grid)
    end

    # EPV requires a complete wet six-neighbor stencil. Omitting the outermost
    # cells also avoids reconstructing unavailable physical-boundary halos.
    epv_valid = falses(Nx, Ny, Nz)
    for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx-1
        epv_valid[i, j, k] =
            active[i, j, k] &&
            active[i-1, j, k] && active[i+1, j, k] &&
            active[i, j-1, k] && active[i, j+1, k] &&
            active[i, j, k-1] && active[i, j, k+1]
    end

    x = collect(xnodes(grid, Center()))
    y = collect(ynodes(grid, Center()))
    z = collect(znodes(grid, Center()))
    return DiagnosticWorkspace(grid, u, v, w, b, epv, u_center, v_center,
                               active, epv_valid, x, y, z)
end

workspaces = Dict(case => DiagnosticWorkspace(S_data[case].grid)
                  for case in CASES)

function masked_slice(field, k, mask)
    values = Array(interior(field, :, :, k))
    values[.!view(mask, :, :, k)] .= NaN
    return values
end

function load_slices!(case, common_index)
    workspace = workspaces[case]
    n = case_indices[case][common_index]

    interior(workspace.u) .= Array(interior(u_data[case][n]))
    interior(workspace.v) .= Array(interior(v_data[case][n]))
    interior(workspace.w) .= Array(interior(w_data[case][n]))
    salinity = Array(interior(S_data[case][n]))
    interior(workspace.b) .=
        -gravitational_acceleration * haline_contraction .*
        (salinity .- reference_salinity)

    fill_halo_regions!((workspace.u, workspace.v, workspace.w, workspace.b))
    compute!(workspace.epv)
    compute!(workspace.u_center)
    compute!(workspace.v_center)

    Nz = size(workspace.grid, 3)
    surface_salinity = Array(interior(S_data[case][n], :, :, Nz))
    surface_salinity[.!view(workspace.active, :, :, Nz)] .= NaN

    return (
        salinity = surface_salinity,
        u = masked_slice(workspace.u_center, Nz, workspace.active),
        v = masked_slice(workspace.v_center, Nz, workspace.active),
        epv = masked_slice(workspace.epv, max(2, Nz - 1),
                           workspace.epv_valid),
    )
end

finite_values(values) = vec(values)[isfinite.(vec(values))]
function robust_symmetric_limit(samples, fallback)
    isempty(samples) && return fallback
    return max(quantile(samples, 0.995), eps(Float64))
end

sample_indices = unique(round.(Int,
    range(first(selection), last(selection);
          length = min(SAMPLE_COUNT, length(selection)))))
u_samples = Float64[]
v_samples = Float64[]
epv_samples = Float64[]

@info "Estimating shared color limits" sample_indices
for common_index in selection[sample_indices], case in CASES
    slices = load_slices!(case, common_index)
    append!(u_samples, abs.(finite_values(slices.u)))
    append!(v_samples, abs.(finite_values(slices.v)))
    append!(epv_samples, abs.(finite_values(slices.epv)))
end

u_limit = robust_symmetric_limit(u_samples, 0.1)
v_limit = robust_symmetric_limit(v_samples, 0.1)
epv_limit = robust_symmetric_limit(epv_samples, 1e-7)
@info "Shared comparison color limits" u_limit v_limit epv_limit

first_slices = Dict(case => load_slices!(case, first(selection))
                    for case in CASES)

function comparison_figure(title, field, colorrange, colorbar_label;
                           colormap = :balance)
    figure_width = isempty(WEAK_DIR) ? 1900 : 2600
    fig = Figure(size = (figure_width, 900), fontsize = 18,
                 backgroundcolor = :white)
    Label(fig[0, 1:length(CASES)], title; fontsize = 26)
    observations = Dict{String, Observable}()
    plots = Dict{String, Any}()

    for (column, case) in enumerate(CASES)
        workspace = workspaces[case]
        observations[case] = Observable(getproperty(first_slices[case], field))
        ax = Axis(fig[1, column],
                  title = CASE_TITLES[case],
                  xlabel = "Eastward x (km)",
                  ylabel = column == 1 ? "Northward y (km)" : "",
                  backgroundcolor = :lightgray,
                  aspect = DataAspect())
        plots[case] = heatmap!(
            ax, workspace.x ./ 1e3, workspace.y ./ 1e3,
            observations[case];
            colormap, colorrange, nan_color = :lightgray,
        )
        xlims!(ax, extrema(workspace.x ./ 1e3)...)
        ylims!(ax, extrema(workspace.y ./ 1e3)...)
        vlines!(ax, [maximum(workspace.x) / 1e3];
                color = :black, linewidth = 1.5, linestyle = :dash)
    end

    Colorbar(fig[1, length(CASES) + 1], plots[last(CASES)];
             label = colorbar_label)
    time_label = Observable(
        "t = $(round(common_times[first(selection)] / 3600; digits = 1)) h")
    Label(fig[2, 1:length(CASES)], time_label; fontsize = 22)
    return fig, observations, time_label
end

fig_S, obs_S, time_S = comparison_figure(
    "SW wind: surface salinity",
    :salinity, (12.0, 34.0), "Salinity"; colormap = :haline)
fig_u, obs_u, time_u = comparison_figure(
    "SW wind: near-surface eastward velocity",
    :u, (-u_limit, u_limit), "u (m s⁻¹)")
fig_v, obs_v, time_v = comparison_figure(
    "SW wind: near-surface northward velocity",
    :v, (-v_limit, v_limit), "v (m s⁻¹)")
fig_epv, obs_epv, time_epv = comparison_figure(
    "SW wind: near-surface Ertel potential vorticity",
    :epv, (-epv_limit, epv_limit), "Ertel PV (s⁻³)")

figures = (fig_S, fig_u, fig_v, fig_epv)
streams = map(fig -> Makie.VideoStream(
                  fig; format = "mp4", framerate = FRAMERATE,
                  backend = CairoMakie),
              figures)

@info "Rendering eastern-boundary comparison" frame_count = length(selection)
for (frame_number, common_index) in enumerate(selection)
    @info "Rendering frame" frame_number total = length(selection) time = common_times[common_index]
    for case in CASES
        slices = load_slices!(case, common_index)
        obs_S[case][] = slices.salinity
        obs_u[case][] = slices.u
        obs_v[case][] = slices.v
        obs_epv[case][] = slices.epv
    end
    label = "t = $(round(common_times[common_index] / 3600; digits = 1)) h"
    time_S[] = label
    time_u[] = label
    time_v[] = label
    time_epv[] = label
    foreach(Makie.recordframe!, streams)
end

output_files = (
    joinpath(OUTPUT_DIR, "sw_surface_salinity_$(COMPARISON_TAG).mp4"),
    joinpath(OUTPUT_DIR, "sw_near_surface_u_$(COMPARISON_TAG).mp4"),
    joinpath(OUTPUT_DIR, "sw_near_surface_v_$(COMPARISON_TAG).mp4"),
    joinpath(OUTPUT_DIR, "sw_surface_epv_$(COMPARISON_TAG).mp4"),
)

final_hour_label = replace(
    string(round(common_times[last(selection)] / 3600; digits = 1)),
    "." => "p",
)
snapshot_files = replace.(output_files,
                          ".mp4" => "_$(final_hour_label)h.png")
for (snapshot_file, figure) in zip(snapshot_files, figures)
    save(snapshot_file, figure)
    @info "Saved final comparison frame" snapshot_file
end

for (output_file, stream) in zip(output_files, streams)
    Makie.save(output_file, stream)
    @info "Saved eastern-boundary comparison animation" output_file
end
