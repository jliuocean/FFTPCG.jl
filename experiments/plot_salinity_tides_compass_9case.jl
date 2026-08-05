using Oceananigans
using CairoMakie
using ArgParse

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--data-root"
            help = "Directory containing the nine SalinityTides case directories"
            arg_type = String
            default = "/mnt/workdir/jliu1/FFTPCG/Data"
        "--output-dir"
            help = "Directory for the comparison animations"
            arg_type = String
            default = ""
        "--framerate"
            help = "Animation frame rate"
            arg_type = Int
            default = 6
        "--tracer-cutoff"
            help = "Tracer concentrations at or below this value are transparent"
            arg_type = Float64
            default = 1e-4
    end
    return parse_args(settings)
end

args = parse_commandline()
const DATA_ROOT = abspath(args["data-root"])
const FRAMERATE = args["framerate"]
const TRACER_CUTOFF = args["tracer-cutoff"]
const OUTPUT_DIR = isempty(args["output-dir"]) ?
    joinpath(DATA_ROOT, "SalinityTides_compass_Qs457_Qw343_dx100_dz1") :
    abspath(args["output-dir"])

FRAMERATE > 0 || error("--framerate must be positive")
TRACER_CUTOFF >= 0 || error("--tracer-cutoff must be nonnegative")
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

function common_times_and_indices(series_groups...)
    case_times = Dict(case => collect(first(series_groups)[case].times)
                      for case in CASES)

    # Fields within a case must share a time axis, but different cases may
    # contain additional snapshots (the SW run includes half-hourly output).
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
    @info "Aligned common compass times" count = length(common_times)
    return common_times, frame_indices
end

####
#### Rotated bathymetry mask
####

const inlet_width = 1000.0
const inlet_depth = 5.0
const santee_center_y = -5e3
const winyah_center_y = 5e3
const original_y₀ = -7.5e3
const original_y₁ = 7.5e3
const original_river_mouth_y = original_y₁ - 200.0
const nearshore_slope_length = 3e3
const nearshore_slope_depth = 5.0
const slope_depth = 15.0
const jetty_length = 3e3
const jetty_width = 100.0
const original_jetty_south_y = original_river_mouth_y - jetty_length
const original_winyah_jetty_x = (winyah_center_y - inlet_width / 2,
                                  winyah_center_y + inlet_width / 2)
const original_santee_channel_south_y =
    original_river_mouth_y - 3e3
const original_santee_channel_x = (santee_center_y - inlet_width / 2,
                                    santee_center_y + inlet_width / 2)

@inline in_original_santee_embayment(x) =
    abs(x - santee_center_y) <= inlet_width / 2
@inline in_original_winyah_embayment(x) =
    abs(x - winyah_center_y) <= inlet_width / 2
@inline in_original_embayment(x) =
    in_original_santee_embayment(x) || in_original_winyah_embayment(x)

@inline function in_original_winyah_jetty(x, y)
    along = original_jetty_south_y <= y <= original_river_mouth_y
    on_south_side =
        abs(x - original_winyah_jetty_x[1]) <= jetty_width / 2
    on_north_side =
        abs(x - original_winyah_jetty_x[2]) <= jetty_width / 2
    return along && (on_south_side || on_north_side)
end

@inline in_original_winyah_channel(x, y) =
    original_winyah_jetty_x[1] < x < original_winyah_jetty_x[2] &&
    original_jetty_south_y <= y <= original_river_mouth_y

@inline in_original_santee_channel(x, y) =
    original_santee_channel_x[1] < x < original_santee_channel_x[2] &&
    original_santee_channel_south_y <= y <= original_river_mouth_y

@inline function original_shelf_depth(x, y)
    if y > original_river_mouth_y
        return in_original_embayment(x) ? inlet_depth : 0.0
    end

    offshore_distance = original_river_mouth_y - y
    if offshore_distance <= nearshore_slope_length
        return nearshore_slope_depth *
               clamp(offshore_distance / nearshore_slope_length, 0.0, 1.0)
    end

    outer_length =
        original_river_mouth_y - original_y₀ - nearshore_slope_length
    outer_fraction = clamp((offshore_distance - nearshore_slope_length) /
                           outer_length, 0.0, 1.0)
    return nearshore_slope_depth +
           (slope_depth - nearshore_slope_depth) * outer_fraction
end

@inline function original_water_depth(x, y)
    in_original_winyah_jetty(x, y) && return 0.0
    in_original_winyah_channel(x, y) && return inlet_depth
    in_original_santee_channel(x, y) && return inlet_depth
    return original_shelf_depth(x, y)
end

@inline water_depth(x, y) = original_water_depth(y, -x)
@inline is_wet(x, y, z) =
    water_depth(x, y) > 0 && z >= -water_depth(x, y)

function surface_geometry(series)
    x = collect(xnodes(series.grid, Center()))
    y = collect(ynodes(series.grid, Center()))
    z = collect(znodes(series.grid, Center()))
    k_surface = lastindex(z)
    wet = [is_wet(xᵢ, yⱼ, z[k_surface]) for xᵢ in x, yⱼ in y]
    return x, y, z, k_surface, wet
end

function masked_surface(series, frame, k_surface, wet;
                        cutoff = nothing)
    values = Array(interior(series[frame], :, :, k_surface))
    values[.!wet] .= NaN
    if !isnothing(cutoff)
        values[values .<= cutoff] .= NaN
    end
    return values
end

function compass_figure(title, times, x_km, y_km;
                        size = (1900, 2300))
    fig = Figure(size = size, fontsize = 18, backgroundcolor = :white)
    Label(fig[0, 1:3], title; fontsize = 28)
    time_label = Observable("t = $(round(times[1] / 3600; digits = 1)) h")
    Label(fig[4, 1:3], time_label; fontsize = 24)

    axes = Dict{String, Axis}()
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
        axes[case] = ax
    end

    return fig, axes, time_label
end

function add_top_grid!(axes)
    for ax in values(axes)
        vlines!(ax, [-5.0, 0.0, 5.0];
                color = (:gray35, 0.65), linewidth = 0.8,
                linestyle = :dash)
        hlines!(ax, [-10.0, 0.0, 10.0];
                color = (:gray35, 0.65), linewidth = 0.8,
                linestyle = :dash)
    end
    return nothing
end

function save_surface_salinity_compass()
    salinity = open_case_series("S")
    times, frame_indices = common_times_and_indices(salinity)
    reference = salinity["CALM"]
    x, y, z, k_surface, wet = surface_geometry(reference)
    x_km, y_km = x ./ 1e3, y ./ 1e3

    frame = Observable(1)
    fields = Dict(
        case => lift(nn -> masked_surface(salinity[case],
                                          frame_indices[case][nn],
                                          k_surface, wet), frame)
        for case in CASES
    )

    fig, axes, time_label =
        compass_figure("Surface salinity: wind-direction comparison",
                       times, x_km, y_km)
    plots = Dict(
        case => heatmap!(axes[case], x_km, y_km, fields[case];
                         colormap = :haline, colorrange = (12.0, 34.0),
                         nan_color = :lightgray)
        for case in CASES
    )
    add_top_grid!(axes)
    Colorbar(fig[1:3, 4], plots["CALM"]; label = "Surface salinity")

    output_file =
        joinpath(OUTPUT_DIR, "surface_salinity_compass_9cases.mp4")
    CairoMakie.record(fig, output_file, eachindex(times);
                      framerate = FRAMERATE) do nn
        frame[] = nn
        time_label[] = "t = $(round(times[nn] / 3600; digits = 1)) h"
    end
    @info "Saved compass surface-salinity animation" output_file
    return output_file
end

function save_surface_tracer_compass()
    santee = open_case_series("c_santee")
    winyah = open_case_series("c_winyah")
    times, frame_indices = common_times_and_indices(santee, winyah)
    reference = santee["CALM"]
    x, y, z, k_surface, wet = surface_geometry(reference)
    x_km, y_km = x ./ 1e3, y ./ 1e3

    frame = Observable(1)
    santee_fields = Dict(
        case => lift(nn -> masked_surface(santee[case],
                                          frame_indices[case][nn], k_surface,
                                          wet; cutoff = TRACER_CUTOFF),
                     frame)
        for case in CASES
    )
    winyah_fields = Dict(
        case => lift(nn -> masked_surface(winyah[case],
                                          frame_indices[case][nn], k_surface,
                                          wet; cutoff = TRACER_CUTOFF),
                     frame)
        for case in CASES
    )

    fig, axes, time_label =
        compass_figure("Surface river tracers: wind-direction comparison",
                       times, x_km, y_km; size = (2050, 2300))
    santee_plots = Dict{String, Any}()
    winyah_plots = Dict{String, Any}()
    for case in CASES
        santee_plots[case] =
            heatmap!(axes[case], x_km, y_km, santee_fields[case];
                     colormap = :viridis, colorrange = (0.0, 1.0),
                     nan_color = :transparent, alpha = 0.72)
        winyah_plots[case] =
            heatmap!(axes[case], x_km, y_km, winyah_fields[case];
                     colormap = :magma, colorrange = (0.0, 1.0),
                     nan_color = :transparent, alpha = 0.62)
    end
    add_top_grid!(axes)
    Colorbar(fig[1:3, 4], santee_plots["CALM"];
             label = "Santee tracer")
    Colorbar(fig[1:3, 5], winyah_plots["CALM"];
             label = "Winyah tracer")

    output_file =
        joinpath(OUTPUT_DIR, "surface_tracers_compass_9cases.mp4")
    CairoMakie.record(fig, output_file, eachindex(times);
                      framerate = FRAMERATE) do nn
        frame[] = nn
        time_label[] = "t = $(round(times[nn] / 3600; digits = 1)) h"
    end
    @info "Saved compass surface-tracer animation" output_file
    return output_file
end

save_surface_salinity_compass()
save_surface_tracer_compass()
