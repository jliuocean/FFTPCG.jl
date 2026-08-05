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
        "--data-root"
            help = "Directory containing the nine SalinityTides case directories"
            arg_type = String
            default = "/mnt/workdir/jliu1/FFTPCG/Data"
        "--output-dir"
            help = "Directory for diagnostic animations"
            arg_type = String
            default = ""
        "--framerate"
            help = "Animation frame rate"
            arg_type = Int
            default = 6
        "--frame-stride"
            help = "Use every Nth saved snapshot"
            arg_type = Int
            default = 1
        "--max-frames"
            help = "Maximum number of frames; zero uses all selected snapshots"
            arg_type = Int
            default = 0
        "--sample-count"
            help = "Representative frames used to estimate robust color limits"
            arg_type = Int
            default = 5
    end
    return parse_args(settings)
end

args = parse_commandline()
const DATA_ROOT = abspath(args["data-root"])
const FRAMERATE = args["framerate"]
const FRAME_STRIDE = args["frame-stride"]
const MAX_FRAMES = args["max-frames"]
const SAMPLE_COUNT = args["sample-count"]
const OUTPUT_DIR = isempty(args["output-dir"]) ?
    joinpath(DATA_ROOT, "SalinityTides_compass_Qs457_Qw343_dx100_dz1") :
    abspath(args["output-dir"])

FRAMERATE > 0 || error("--framerate must be positive")
FRAME_STRIDE > 0 || error("--frame-stride must be positive")
MAX_FRAMES >= 0 || error("--max-frames must be nonnegative")
SAMPLE_COUNT > 0 || error("--sample-count must be positive")
mkpath(OUTPUT_DIR)

const CASES = ("NW", "N", "NE", "W", "CALM", "E", "SW", "S", "SE")
const PANEL_POSITION = Dict(
    "NW"   => (1, 1), "N"    => (1, 2), "NE" => (1, 3),
    "W"    => (2, 1), "CALM" => (2, 2), "E"  => (2, 3),
    "SW"   => (3, 1), "S"    => (3, 2), "SE" => (3, 3),
)
const CASE_TITLES = Dict(case => case == "CALM" ? "No wind" : "$case wind"
                         for case in CASES)

function case_directory(case)
    direct_directory = joinpath(DATA_ROOT, case)
    isfile(joinpath(direct_directory, "instantaneous_fields.jld2")) &&
        return direct_directory

    stress = case == "CALM" ? "0p0" : "0p03"
    directory = "SalinityTides_Qs457_Qw343_dx100p0_dz1p0_" *
                "tau$(case)$(stress)_delay1p0M2_T74p4h"
    return joinpath(DATA_ROOT, directory)
end

case_data_file(case) =
    joinpath(case_directory(case), "instantaneous_fields.jld2")

for case in CASES
    isfile(case_data_file(case)) ||
        error("Missing $case data: $(case_data_file(case))")
end

function open_case_series(field_name)
    return Dict(case => FieldTimeSeries(case_data_file(case), field_name)
                for case in CASES)
end

u_data = open_case_series("u")
v_data = open_case_series("v")
w_data = open_case_series("w")
S_data = open_case_series("S")

function common_times_and_indices(series_groups...)
    case_times = Dict(case => collect(first(series_groups)[case].times)
                      for case in CASES)

    for group in series_groups, case in CASES
        times = collect(group[case].times)
        length(times) == length(case_times[case]) ||
            error("Within-case time-count mismatch for $case")
        all(isapprox.(times, case_times[case]; atol = 1e-6, rtol = 0)) ||
            error("Within-case time-coordinate mismatch for $case")
    end

    reference_times = case_times[first(CASES)]
    common_times = Float64[]
    frame_indices = Dict(case => Int[] for case in CASES)

    for time in reference_times
        indices = Dict(case => findfirst(t -> isapprox(t, time;
                                                       atol = 1e-6,
                                                       rtol = 0),
                                         case_times[case])
                       for case in CASES)
        any(isnothing, values(indices)) && continue
        push!(common_times, time)
        for case in CASES
            push!(frame_indices[case], something(indices[case]))
        end
    end

    isempty(common_times) && error("No common output times found")
    @info "Aligned common diagnostic times" count = length(common_times)
    return common_times, frame_indices
end

times, frame_indices = common_times_and_indices(u_data, v_data, w_data, S_data)
frames = collect(1:FRAME_STRIDE:length(times))
MAX_FRAMES > 0 && (frames = frames[1:min(MAX_FRAMES, length(frames))])

####
#### Oceanostics working fields
####

grid = S_data["CALM"].grid
Nx, Ny, Nz = size(grid)
u = XFaceField(grid)
v = YFaceField(grid)
w = ZFaceField(grid)
b = CenterField(grid)

const gravitational_acceleration = 9.81
const haline_contraction = 7.8e-4
const reference_salinity = 34.0
const f₀ = 8e-5

diagnostic_model = (
    grid = grid,
    velocities = (u = u, v = v, w = w),
    tracers = (b = b,),
    coriolis = FPlane(f = f₀),
)

epv = Field(@at (Center, Center, Center) ErtelPotentialVorticity(diagnostic_model))
u_center = Field(@at (Center, Center, Center) u)
v_center = Field(@at (Center, Center, Center) v)

x = collect(xnodes(grid, Center()))
y = collect(ynodes(grid, Center()))
z = collect(znodes(grid, Center()))
z_w = collect(znodes(grid, Face()))
x_km, y_km = x ./ 1e3, y ./ 1e3
k_near_surface = Nz
k_epv_surface = max(2, Nz - 1)
k_w_near_surface = Nz

active = falses(Nx, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    active[i, j, k] = !inactive_cell(i, j, k, grid)
end

# EPV uses spatial derivatives. Exclude physical edges and cells without a
# complete six-neighbor wet stencil.
epv_valid = falses(Nx, Ny, Nz)
for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx-1
    epv_valid[i, j, k] =
        active[i, j, k] &&
        active[i-1, j, k] && active[i+1, j, k] &&
        active[i, j-1, k] && active[i, j+1, k] &&
        active[i, j, k-1] && active[i, j, k+1]
end

# For every horizontal wet column, select the deepest cell center with a valid
# EPV stencil. This is one cell above the first active bottom cell in regions
# where the bathymetry is resolved by more than two vertical cells.
bottom_epv_index = zeros(Int, Nx, Ny)
for j in 1:Ny, i in 1:Nx
    for k in 2:Nz-1
        if epv_valid[i, j, k]
            bottom_epv_index[i, j] = k
            break
        end
    end
end

function masked_horizontal_slice(field, k, mask)
    values = Array(interior(field, :, :, k))
    values[.!view(mask, :, :, k)] .= NaN
    return values
end

function bottom_following_slice(field)
    volume = Array(interior(field))
    values = fill(NaN, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        k = bottom_epv_index[i, j]
        k == 0 || (values[i, j] = volume[i, j, k])
    end
    return values
end

function masked_near_surface_w()
    values = Array(interior(w, :, :, k_w_near_surface))
    valid = view(active, :, :, Nz-1) .& view(active, :, :, Nz)
    values[.!valid] .= NaN
    return values
end

function load_case_diagnostics!(case, frame)
    interior(u) .= Array(interior(u_data[case][frame]))
    interior(v) .= Array(interior(v_data[case][frame]))
    interior(w) .= Array(interior(w_data[case][frame]))
    salinity = Array(interior(S_data[case][frame]))
    interior(b) .= -gravitational_acceleration * haline_contraction .*
                   (salinity .- reference_salinity)

    fill_halo_regions!((u, v, w, b))
    compute!(epv)
    compute!(u_center)
    compute!(v_center)

    return (
        epv_surface =
            masked_horizontal_slice(epv, k_epv_surface, epv_valid),
        epv_bottom = bottom_following_slice(epv),
        u_surface =
            masked_horizontal_slice(u_center, k_near_surface, active),
        v_surface =
            masked_horizontal_slice(v_center, k_near_surface, active),
        w_surface = masked_near_surface_w(),
    )
end

####
#### Robust, common color ranges
####

finite_values(values) = vec(values)[isfinite.(vec(values))]
robust_symmetric_limit(values, fallback) =
    isempty(values) ? fallback :
    max(quantile(values, 0.995), eps(Float64))

sample_frames =
    unique(round.(Int, range(first(frames), last(frames);
                              length = min(SAMPLE_COUNT, length(frames)))))
epv_surface_samples = Float64[]
epv_bottom_samples = Float64[]
u_samples = Float64[]
v_samples = Float64[]
w_samples = Float64[]

@info "Estimating common diagnostic color ranges" sample_frames
for frame in sample_frames, case in CASES
    slices = load_case_diagnostics!(case, frame_indices[case][frame])
    append!(epv_surface_samples,
            abs.(finite_values(slices.epv_surface)))
    append!(epv_bottom_samples,
            abs.(finite_values(slices.epv_bottom)))
    append!(u_samples, abs.(finite_values(slices.u_surface)))
    append!(v_samples, abs.(finite_values(slices.v_surface)))
    append!(w_samples, abs.(finite_values(slices.w_surface)))
end

epv_surface_limit =
    robust_symmetric_limit(epv_surface_samples, 1e-7)
epv_bottom_limit =
    robust_symmetric_limit(epv_bottom_samples, 1e-7)
u_limit = robust_symmetric_limit(u_samples, 0.1)
v_limit = robust_symmetric_limit(v_samples, 0.1)
w_limit = robust_symmetric_limit(w_samples, 1e-3)

@info "Common diagnostic color ranges" epv_surface_limit epv_bottom_limit u_limit v_limit w_limit

####
#### Compass figures and synchronized video streams
####

function compass_figure(title, colorbar_label, colorrange)
    fig = Figure(size = (1900, 2300), fontsize = 18,
                 backgroundcolor = :white)
    Label(fig[0, 1:3], title; fontsize = 28)
    time_label =
        Observable("t = $(round(times[first(frames)] / 3600; digits = 1)) h")
    Label(fig[4, 1:3], time_label; fontsize = 24)

    axes = Dict{String, Axis}()
    fields = Dict(case => Observable(fill(NaN, Nx, Ny)) for case in CASES)
    plots = Dict{String, Any}()

    for case in CASES
        row, column = PANEL_POSITION[case]
        ax = Axis(fig[row, column];
                  title = CASE_TITLES[case],
                  xlabel = row == 3 ? "Eastward x (km)" : "",
                  ylabel = column == 1 ? "Northward y (km)" : "",
                  backgroundcolor = :lightgray,
                  aspect = DataAspect())
        xlims!(ax, extrema(x_km)...)
        ylims!(ax, extrema(y_km)...)
        row == 3 || hidexdecorations!(ax; grid = false)
        column == 1 || hideydecorations!(ax; grid = false)
        plots[case] =
            heatmap!(ax, x_km, y_km, fields[case];
                     colormap = :balance, colorrange,
                     nan_color = :lightgray)
        vlines!(ax, [-5.0, 0.0, 5.0];
                color = (:gray35, 0.65), linewidth = 0.8,
                linestyle = :dash)
        hlines!(ax, [-10.0, 0.0, 10.0];
                color = (:gray35, 0.65), linewidth = 0.8,
                linestyle = :dash)
        axes[case] = ax
    end

    Colorbar(fig[1:3, 4], plots["CALM"];
             label = colorbar_label)
    return fig, fields, time_label
end

fig_epv_surface, obs_epv_surface, time_epv_surface =
    compass_figure(
        "Near-surface Ertel potential vorticity, z = $(round(z[k_epv_surface]; digits = 1)) m",
        "Ertel PV (s⁻³)",
        (-epv_surface_limit, epv_surface_limit),
    )

fig_epv_bottom, obs_epv_bottom, time_epv_bottom =
    compass_figure(
        "Near-bottom Ertel potential vorticity",
        "Ertel PV (s⁻³)",
        (-epv_bottom_limit, epv_bottom_limit),
    )

fig_u, obs_u, time_u =
    compass_figure(
        "Near-surface eastward velocity u, z = $(round(z[k_near_surface]; digits = 1)) m",
        "u (m s⁻¹)",
        (-u_limit, u_limit),
    )

fig_v, obs_v, time_v =
    compass_figure(
        "Near-surface northward velocity v, z = $(round(z[k_near_surface]; digits = 1)) m",
        "v (m s⁻¹)",
        (-v_limit, v_limit),
    )

fig_w, obs_w, time_w =
    compass_figure(
        "Near-surface vertical velocity w, z = $(round(z_w[k_w_near_surface]; digits = 1)) m",
        "w (m s⁻¹)",
        (-w_limit, w_limit),
    )

figures = (fig_epv_surface, fig_epv_bottom, fig_u, fig_v, fig_w)
streams = map(fig -> Makie.VideoStream(fig; format = "mp4",
                                        framerate = FRAMERATE,
                                        backend = CairoMakie),
              figures)

@info "Rendering five synchronized compass animations" frame_count = length(frames)
for (frame_number, frame) in enumerate(frames)
    @info "Rendering diagnostic frame" frame_number total = length(frames) time = times[frame]
    for case in CASES
        slices = load_case_diagnostics!(case, frame_indices[case][frame])
        obs_epv_surface[case][] = slices.epv_surface
        obs_epv_bottom[case][] = slices.epv_bottom
        obs_u[case][] = slices.u_surface
        obs_v[case][] = slices.v_surface
        obs_w[case][] = slices.w_surface
    end

    label = "t = $(round(times[frame] / 3600; digits = 1)) h"
    time_epv_surface[] = label
    time_epv_bottom[] = label
    time_u[] = label
    time_v[] = label
    time_w[] = label
    foreach(Makie.recordframe!, streams)
end

output_files = (
    joinpath(OUTPUT_DIR, "surface_epv_compass_9cases.mp4"),
    joinpath(OUTPUT_DIR, "bottom_epv_compass_9cases.mp4"),
    joinpath(OUTPUT_DIR, "near_surface_u_compass_9cases.mp4"),
    joinpath(OUTPUT_DIR, "near_surface_v_compass_9cases.mp4"),
    joinpath(OUTPUT_DIR, "near_surface_w_compass_9cases.mp4"),
)

for (output_file, stream) in zip(output_files, streams)
    Makie.save(output_file, stream)
    @info "Saved diagnostic compass animation" output_file
end
