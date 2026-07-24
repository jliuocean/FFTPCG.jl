using Oceananigans
using CairoMakie

const WIND_SPEED = isempty(ARGS) ? 6.0 : parse(Float64, first(ARGS))
const SANTEE_DISCHARGE = length(ARGS) < 2 ? 1000.0 : parse(Float64, ARGS[2])
const WINYAH_DISCHARGE = length(ARGS) < 3 ? 750.0 : parse(Float64, ARGS[3])
const PLOT_MODE = length(ARGS) < 4 ? "density-and-both-tracers" : lowercase(ARGS[4])
const RIVER_GEOMETRY = length(ARGS) < 5 ? "both-rivers" : lowercase(ARGS[5])
const NO_SANTEE_RIVER = RIVER_GEOMETRY == "no-santee-river"
WIND_SPEED > 0 || error("Wind speed must be positive")
SANTEE_DISCHARGE >= 0 || error("Santee discharge must be nonnegative")
WINYAH_DISCHARGE > 0 || error("Winyah discharge must be positive")
PLOT_MODE in ("density-and-both-tracers", "density-and-winyah", "single-tracers", "all") ||
    error("Plot mode must be density-and-both-tracers, density-and-winyah, single-tracers, or all")
RIVER_GEOMETRY in ("both-rivers", "no-santee-river") ||
    error("River geometry must be both-rivers or no-santee-river")
NO_SANTEE_RIVER && SANTEE_DISCHARGE != 0 &&
    error("no-santee-river geometry requires zero Santee discharge")

speed_label(speed) = replace(string(round(speed; digits = 1)), "." => "p")
const WIND_SPEED_LABEL = speed_label(WIND_SPEED)
const WIND_SPEED_FILE_LABEL = isinteger(WIND_SPEED) ? string(Int(WIND_SPEED)) :
                              WIND_SPEED_LABEL
const SANTEE_DISCHARGE_LABEL = isinteger(SANTEE_DISCHARGE) ?
                                string(Int(SANTEE_DISCHARGE)) :
                                speed_label(SANTEE_DISCHARGE)
const WINYAH_DISCHARGE_LABEL = isinteger(WINYAH_DISCHARGE) ?
                               string(Int(WINYAH_DISCHARGE)) :
                               speed_label(WINYAH_DISCHARGE)
const DATA_ROOT = get(ENV, "FFTPCG_DATA_ROOT", "/mnt/workdir/jliu1/FFTPCG/Data")
const RIVER_GEOMETRY_SUFFIX = NO_SANTEE_RIVER ? "_NoSanteeRiver" : ""
const OUTPUT_DIR = joinpath(DATA_ROOT,
                            "MAMD_RotatedCCW_8wind_comparison_" *
                            "Winyah$(WINYAH_DISCHARGE_LABEL)_" *
                            "Santee$(SANTEE_DISCHARGE_LABEL)_" *
                            "$(WIND_SPEED_LABEL)ms" *
                            RIVER_GEOMETRY_SUFFIX)
mkpath(OUTPUT_DIR)

const CASES = ("NW", "N", "NE", "W", "E", "SW", "S", "SE")
const PANEL_POSITION = Dict(
    "NW" => (1, 1), "N" => (1, 2), "NE" => (1, 3),
    "W"  => (2, 1),                  "E"  => (2, 3),
    "SW" => (3, 1), "S" => (3, 2), "SE" => (3, 3),
)

const CASE_PREFIX = "MAMD_RotatedCCW_Lx20km_Ly40km_" *
                    "Winyah$(WINYAH_DISCHARGE_LABEL)_" *
                    "Santee$(SANTEE_DISCHARGE_LABEL)_" *
                    (NO_SANTEE_RIVER ? "NoSanteeRiver_" : "") *
                    "SouthOutflow_Nx200_Ny400_Nz20_Wind$(WIND_SPEED_LABEL)ms_"
const CASE_SUFFIX = "_calm12h_ramp12h_hold72h_total96h"
const ρ₀ = 1025.0
const shelf_N² = 1e-5

# Geometry constants used only to mask dry surface cells consistently with the
# simulation's existing single-case surface-density animation.
const Δy = 100.0
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
const jetty_width = Δy
const original_jetty_south_y = original_river_mouth_y - jetty_length
const original_winyah_jetty_x = (winyah_center_y - inlet_width / 2,
                                  winyah_center_y + inlet_width / 2)
const original_santee_channel_south_y = original_river_mouth_y - 3e3
const original_santee_channel_x = (santee_center_y - inlet_width / 2,
                                    santee_center_y + inlet_width / 2)

@inline in_original_santee_embayment(x) = abs(x - santee_center_y) <= inlet_width / 2
@inline in_original_winyah_embayment(x) = abs(x - winyah_center_y) <= inlet_width / 2
@inline in_original_embayment(x) = (!NO_SANTEE_RIVER && in_original_santee_embayment(x)) ||
                                   in_original_winyah_embayment(x)

@inline function in_original_winyah_jetty(x, y)
    along = original_jetty_south_y <= y <= original_river_mouth_y
    south_side = abs(x - original_winyah_jetty_x[1]) <= jetty_width / 2
    north_side = abs(x - original_winyah_jetty_x[2]) <= jetty_width / 2
    return along && (south_side || north_side)
end

@inline in_original_winyah_channel(x, y) =
    original_winyah_jetty_x[1] < x < original_winyah_jetty_x[2] &&
    original_jetty_south_y <= y <= original_river_mouth_y

@inline in_original_santee_channel(x, y) =
    !NO_SANTEE_RIVER &&
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

    outer_length = original_river_mouth_y - original_y₀ - nearshore_slope_length
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
@inline is_wet(x, y, z) = water_depth(x, y) > 0 && z >= -water_depth(x, y)
@inline ambient_buoyancy(z) = shelf_N² * z

case_directory(case) = joinpath(DATA_ROOT, CASE_PREFIX * case * CASE_SUFFIX)
case_data_file(case) = joinpath(case_directory(case), "instantaneous_fields.jld2")

for case in CASES
    isfile(case_data_file(case)) || error("Missing data for $case: $(case_data_file(case))")
end

function open_case_series(field_name)
    return Dict(case => FieldTimeSeries(case_data_file(case), field_name) for case in CASES)
end

function validate_common_times(series_groups...)
    reference = first(series_groups)["NW"].times
    for series_group in series_groups, case in CASES
        times = series_group[case].times
        length(times) == length(reference) ||
            error("Time-count mismatch for $case")
        all(isapprox.(times, reference; atol = 1e-8, rtol = 0)) ||
            error("Time-coordinate mismatch for $case")
    end
    return reference
end

function comparison_figure(title, times; size = (1800, 2000))
    fig = Figure(size = size, fontsize = 18, backgroundcolor = :white)
    Label(fig[0, 1:3], title, fontsize = 28)
    time_text = Observable("t = $(round(times[1] / 3600; digits = 1)) hour")
    santee_description = NO_SANTEE_RIVER ? "Santee: removed" :
                          "Santee: $(SANTEE_DISCHARGE_LABEL) m³ s⁻¹"
    center_text = lift(time_text) do time_label
        "$time_label\n\n$santee_description\n" *
        "Winyah: $(WINYAH_DISCHARGE_LABEL) m³ s⁻¹\n\n" *
        "0–12 h: no wind\n12–24 h: ramp to $(WIND_SPEED) m s⁻¹\n" *
        "24–96 h: constant wind"
    end
    Label(fig[2, 2], center_text, fontsize = 23,
          tellwidth = false, tellheight = false, justification = :center)
    return fig, time_text
end

function case_axes!(fig, xC, yC)
    axes = Dict{String, Axis}()
    for case in CASES
        row, column = PANEL_POSITION[case]
        ax = Axis(fig[row, column], title = "$case wind",
                  xlabel = row == 3 ? "Eastward x (km)" : "",
                  ylabel = column == 1 ? "Northward y (km)" : "",
                  backgroundcolor = :white, aspect = DataAspect())
        xlims!(ax, extrema(xC)...)
        ylims!(ax, extrema(yC)...)
        row == 3 || hidexdecorations!(ax; grid = false)
        column == 1 || hideydecorations!(ax; grid = false)
        axes[case] = ax
    end
    return axes
end

function save_surface_density_comparison()
    buoyancy = open_case_series("b")
    times = validate_common_times(buoyancy)
    reference = buoyancy["NW"]
    x_m = collect(xnodes(reference.grid, Center()))
    y_m = collect(ynodes(reference.grid, Center()))
    z_m = collect(znodes(reference.grid, Center()))
    xC, yC = x_m ./ 1e3, y_m ./ 1e3
    k_surface = length(z_m)
    z_surface = z_m[k_surface]
    wet = [is_wet(x, y, z_surface) for x in x_m, y in y_m]

    function density_surface(snapshot)
        b = Array(interior(snapshot, :, :, k_surface))
        density = -ρ₀ .* (b .- ambient_buoyancy(z_surface)) ./ 9.81
        density[.!wet] .= NaN
        return density
    end

    n = Observable(1)
    title = "Surface density anomaly: eight wind directions, $(WIND_SPEED) m s⁻¹"
    fig, time_text = comparison_figure(title, times)
    axes = case_axes!(fig, xC, yC)
    density = Dict(case => lift(nn -> density_surface(buoyancy[case][nn]), n)
                   for case in CASES)
    plots = Dict(case => heatmap!(axes[case], xC, yC, density[case];
                                  colormap = :balance,
                                  colorrange = (-12.0, 12.0),
                                  nan_color = :transparent)
                 for case in CASES)
    Colorbar(fig[1:3, 4], plots["NW"]; label = "Density anomaly (kg m⁻³)")

    output_file = joinpath(OUTPUT_DIR,
                           "surface_density_8cases_Winyah$(WINYAH_DISCHARGE_LABEL)_" *
                           "Santee$(SANTEE_DISCHARGE_LABEL)_" *
                           "$(WIND_SPEED_FILE_LABEL)ms.mp4")
    CairoMakie.record(fig, output_file, 1:length(times); framerate = 6) do nn
        n[] = nn
        time_text[] = "t = $(round(times[nn] / 3600; digits = 1)) hour"
    end
    @info "Saved eight-case surface density animation" output_file
end

function masked_tracer(snapshot, k_surface; cutoff = 1e-6)
    values = Array(interior(snapshot, :, :, k_surface))
    values[values .<= cutoff] .= NaN
    return values
end

function save_surface_tracer_comparison()
    santee = open_case_series("c_santee")
    winyah = open_case_series("c_winyah")
    times = validate_common_times(santee, winyah)
    reference = santee["NW"]
    x_m = collect(xnodes(reference.grid, Center()))
    y_m = collect(ynodes(reference.grid, Center()))
    z_m = collect(znodes(reference.grid, Center()))
    xC, yC = x_m ./ 1e3, y_m ./ 1e3
    k_surface = length(z_m)

    n = Observable(1)
    title = "Surface river tracers: eight wind directions, $(WIND_SPEED) m s⁻¹"
    fig, time_text = comparison_figure(title, times; size = (1900, 2000))
    axes = case_axes!(fig, xC, yC)
    santee_surface = Dict(case => lift(nn -> masked_tracer(santee[case][nn], k_surface), n)
                           for case in CASES)
    winyah_surface = Dict(case => lift(nn -> masked_tracer(winyah[case][nn], k_surface), n)
                          for case in CASES)

    santee_plots = Dict{String, Any}()
    winyah_plots = Dict{String, Any}()
    for case in CASES
        santee_plots[case] = heatmap!(axes[case], xC, yC, santee_surface[case];
                                      colormap = :viridis, colorrange = (0.0, 1.0),
                                      nan_color = :transparent, alpha = 0.72)
        winyah_plots[case] = heatmap!(axes[case], xC, yC, winyah_surface[case];
                                     colormap = :magma, colorrange = (0.0, 1.0),
                                     nan_color = :transparent, alpha = 0.62)
    end
    Colorbar(fig[1:3, 4], santee_plots["NW"]; label = "Santee tracer")
    Colorbar(fig[1:3, 5], winyah_plots["NW"]; label = "Winyah tracer")

    output_file = joinpath(OUTPUT_DIR,
                           "surface_tracers_8cases_Winyah$(WINYAH_DISCHARGE_LABEL)_" *
                           "Santee$(SANTEE_DISCHARGE_LABEL)_" *
                           "$(WIND_SPEED_FILE_LABEL)ms.mp4")
    CairoMakie.record(fig, output_file, 1:length(times); framerate = 6) do nn
        n[] = nn
        time_text[] = "t = $(round(times[nn] / 3600; digits = 1)) hour"
    end
    @info "Saved eight-case surface tracer animation" output_file
end

function save_single_river_surface_tracer_comparison(river)
    river in (:winyah, :santee) || error("River must be :winyah or :santee")
    field_name = river == :winyah ? "c_winyah" : "c_santee"
    river_name = river == :winyah ? "Winyah" : "Santee"
    colormap = river == :winyah ? :magma : :viridis
    tracer = open_case_series(field_name)
    times = validate_common_times(tracer)
    reference = tracer["NW"]
    x_m = collect(xnodes(reference.grid, Center()))
    y_m = collect(ynodes(reference.grid, Center()))
    z_m = collect(znodes(reference.grid, Center()))
    xC, yC = x_m ./ 1e3, y_m ./ 1e3
    k_surface = length(z_m)

    n = Observable(1)
    title = "Surface $river_name tracer: eight wind directions, $(WIND_SPEED) m s⁻¹"
    fig, time_text = comparison_figure(title, times)
    axes = case_axes!(fig, xC, yC)
    surface = Dict(case => lift(nn -> masked_tracer(tracer[case][nn], k_surface), n)
                   for case in CASES)
    plots = Dict(case => heatmap!(axes[case], xC, yC, surface[case];
                                  colormap, colorrange = (0.0, 1.0),
                                  nan_color = :transparent, alpha = 0.9)
                 for case in CASES)
    Colorbar(fig[1:3, 4], plots["NW"]; label = "$river_name tracer")

    river_file_label = lowercase(river_name)
    output_file = joinpath(OUTPUT_DIR,
                           "surface_$(river_file_label)_tracer_8cases_" *
                           "Winyah$(WINYAH_DISCHARGE_LABEL)_" *
                           "Santee$(SANTEE_DISCHARGE_LABEL)_" *
                           "$(WIND_SPEED_FILE_LABEL)ms.mp4")
    CairoMakie.record(fig, output_file, 1:length(times); framerate = 6) do nn
        n[] = nn
        time_text[] = "t = $(round(times[nn] / 3600; digits = 1)) hour"
    end
    @info "Saved eight-case single-river surface tracer animation" river_name output_file
end

if PLOT_MODE in ("density-and-both-tracers", "density-and-winyah", "all")
    save_surface_density_comparison()
end

if PLOT_MODE in ("density-and-both-tracers", "all")
    save_surface_tracer_comparison()
end

if PLOT_MODE in ("single-tracers", "all")
    save_single_river_surface_tracer_comparison(:winyah)
    save_single_river_surface_tracer_comparison(:santee)
end

PLOT_MODE == "density-and-winyah" &&
    save_single_river_surface_tracer_comparison(:winyah)
