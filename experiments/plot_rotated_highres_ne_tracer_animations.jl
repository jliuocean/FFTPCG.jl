using Oceananigans
using CairoMakie

const DEFAULT_DATA_DIR =
    "/mnt/workdir/jliu1/FFTPCG/Data/" *
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee500_" *
    "SouthOutflow_Nx800_Ny1600_Nz40_Wind3p0ms_NE_" *
    "calm12h_ramp12h_hold0h_total24h"
const DATA_DIR = get(ENV, "FFTPCG_DATA_DIR", DEFAULT_DATA_DIR)
const DATA_FILE = joinpath(DATA_DIR, "instantaneous_fields.jld2")
const SURFACE_ONLY = "--surface-only" in ARGS
const PROFILES_ONLY = "--profiles-only" in ARGS
SURFACE_ONLY && PROFILES_ONLY &&
    error("--surface-only and --profiles-only cannot be combined")
isfile(DATA_FILE) || error("Missing field data: $DATA_FILE")

const ρ₀ = 1025.0
const inlet_width = 1000.0
const inlet_depth = 5.0
const santee_center_y = -5e3
const winyah_center_y = 5e3
const original_y₀ = -7.5e3
const original_y₁ = 7.5e3
const original_river_mouth_y = original_y₁ - 200.0
const rotated_coast_x = -original_river_mouth_y
const nearshore_slope_length = 3e3
const nearshore_slope_depth = 5.0
const slope_depth = 15.0
const jetty_length = 3e3
const jetty_width = 100.0
const original_jetty_south_y = original_river_mouth_y - jetty_length
const original_winyah_jetty_x = (winyah_center_y - inlet_width / 2,
                                  winyah_center_y + inlet_width / 2)
const original_santee_channel_south_y = original_river_mouth_y - 3e3
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
    south_side = abs(x - original_winyah_jetty_x[1]) <= jetty_width / 2
    north_side = abs(x - original_winyah_jetty_x[2]) <= jetty_width / 2
    return along && (south_side || north_side)
end

@inline function in_original_winyah_channel(x, y)
    return original_winyah_jetty_x[1] < x < original_winyah_jetty_x[2] &&
           original_jetty_south_y <= y <= original_river_mouth_y
end

@inline function in_original_santee_channel(x, y)
    return original_santee_channel_x[1] < x < original_santee_channel_x[2] &&
           original_santee_channel_south_y <= y <= original_river_mouth_y
end

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
nearest_index(nodes, value) = argmin(abs.(nodes .- value))

c_santee_data = FieldTimeSeries(DATA_FILE, "c_santee")
c_winyah_data = FieldTimeSeries(DATA_FILE, "c_winyah")
length(c_santee_data.times) == length(c_winyah_data.times) ||
    error("Santee and Winyah time counts differ")
all(isapprox.(c_santee_data.times, c_winyah_data.times;
              atol = 1e-8, rtol = 0)) ||
    error("Santee and Winyah time coordinates differ")

const Nt = length(c_santee_data.times)
Nt > 0 || error("No snapshots found in $DATA_FILE")
const times = c_santee_data.times
const x_m = collect(xnodes(c_santee_data.grid, Center()))
const y_m = collect(ynodes(c_santee_data.grid, Center()))
const z_m = collect(znodes(c_santee_data.grid, Center()))
const x_km = x_m ./ 1e3
const y_km = y_m ./ 1e3
const k_surface = length(z_m)
const depth = [water_depth(x, y) for x in x_m, y in y_m]
const wet_surface = [depth[i, j] > 0 &&
                     z_m[k_surface] >= -depth[i, j]
                     for i in eachindex(x_m), j in eachindex(y_m)]

const DISTANCES = (("Mouth", 0.0),
                   ("2 km offshore", 2e3),
                   ("5 km offshore", 5e3))
const RIVERS = (("Santee", santee_center_y),
                ("Winyah", winyah_center_y))
const stations = [(
    river = river,
    location = location,
    distance = distance,
    i = nearest_index(x_m, rotated_coast_x + distance),
    j = nearest_index(y_m, center_y),
) for (location, distance) in DISTANCES
  for (river, center_y) in RIVERS]

function masked_surface(snapshot; cutoff = 1e-6)
    values = Array(interior(snapshot, :, :, k_surface))
    values[.!wet_surface .| (values .<= cutoff)] .= NaN
    return values
end

function masked_profile(snapshot, station)
    values = Array(interior(snapshot, station.i, station.j, :))
    local_depth = depth[station.i, station.j]
    values[(z_m .< -local_depth) .| (values .< 0)] .= NaN
    return values
end

function time_string(nn)
    return "t = $(round(times[nn] / 3600; digits = 1)) hour"
end

function save_surface_tracer_animation()
    santee_surface = Observable(masked_surface(c_santee_data[1]))
    winyah_surface = Observable(masked_surface(c_winyah_data[1]))
    time_label = Observable(time_string(1))

    background_depth = copy(depth)
    background_depth[background_depth .<= 0] .= NaN

    fig = Figure(size = (1500, 1050), fontsize = 20,
                 backgroundcolor = :white)
    ax = Axis(fig[1, 1],
              title = "Surface river tracers",
              xlabel = "Eastward x (km)",
              ylabel = "Northward y (km)",
              backgroundcolor = :gray90,
              aspect = DataAspect())

    heatmap!(ax, x_km, y_km, background_depth;
             colormap = :deep, colorrange = (0.0, slope_depth),
             nan_color = :gray90)
    santee_plot = heatmap!(ax, x_km, y_km, santee_surface;
                           colormap = :viridis, colorrange = (0.0, 1.0),
                           nan_color = :transparent, alpha = 0.76)
    winyah_plot = heatmap!(ax, x_km, y_km, winyah_surface;
                           colormap = :magma, colorrange = (0.0, 1.0),
                           nan_color = :transparent, alpha = 0.64)

    for station in stations
        marker = station.river == "Santee" ? :circle : :rect
        scatter!(ax, [x_km[station.i]], [y_km[station.j]];
                 marker, markersize = 15, color = :white,
                 strokecolor = :black, strokewidth = 2)
    end

    vlines!(ax, -10:2.5:10; color = (:gray20, 0.55),
            linestyle = :dash, linewidth = 1)
    hlines!(ax, -20:5:20; color = (:gray20, 0.55),
            linestyle = :dash, linewidth = 1)
    xlims!(ax, extrema(x_km)...)
    ylims!(ax, extrema(y_km)...)

    Colorbar(fig[1, 2], santee_plot; label = "Santee tracer")
    Colorbar(fig[1, 3], winyah_plot; label = "Winyah tracer")
    Label(fig[0, 1:3], time_label, fontsize = 24)

    output_file = joinpath(DATA_DIR, "surface_tracers_xy.mp4")
    CairoMakie.record(fig, output_file, 1:Nt; framerate = 6) do nn
        santee_surface[] = masked_surface(c_santee_data[nn])
        winyah_surface[] = masked_surface(c_winyah_data[nn])
        time_label[] = time_string(nn)
    end

    @info "Saved high-resolution surface tracer animation" output_file
    return output_file
end

function save_vertical_profile_animation()
    santee_profiles = Dict(
        (station.river, station.location) =>
            Observable(masked_profile(c_santee_data[1], station))
        for station in stations
    )
    winyah_profiles = Dict(
        (station.river, station.location) =>
            Observable(masked_profile(c_winyah_data[1], station))
        for station in stations
    )
    time_label = Observable(time_string(1))

    fig = Figure(size = (1500, 1450), fontsize = 18,
                 backgroundcolor = :white)
    Label(fig[0, 1:2], time_label, fontsize = 24)

    for station in stations
        row = findfirst(entry -> entry[1] == station.location, DISTANCES)
        column = station.river == "Santee" ? 1 : 2
        station_x = round(x_km[station.i]; digits = 2)
        station_y = round(y_km[station.j]; digits = 2)
        local_depth = depth[station.i, station.j]

        ax = Axis(fig[row, column],
                  title = "$(station.river), $(station.location) " *
                          "(x=$(station_x) km, y=$(station_y) km)",
                  xlabel = "Tracer concentration",
                  ylabel = "z (m)",
                  xgridvisible = true,
                  ygridvisible = true,
                  xgridstyle = :dash,
                  ygridstyle = :dash)

        key = (station.river, station.location)
        lines!(ax, santee_profiles[key], z_m;
               color = :seagreen, linewidth = 3, label = "Santee")
        lines!(ax, winyah_profiles[key], z_m;
               color = :darkorange, linewidth = 3, label = "Winyah")
        hlines!(ax, [-local_depth]; color = :black,
                linestyle = :dot, linewidth = 2)
        xlims!(ax, 0.0, 1.0)
        ylims!(ax, -15.0, 0.0)
        row == 1 && column == 1 && axislegend(ax; position = :lb)
    end

    output_file = joinpath(DATA_DIR, "tracer_vertical_profiles.mp4")
    CairoMakie.record(fig, output_file, 1:Nt; framerate = 6) do nn
        santee_snapshot = c_santee_data[nn]
        winyah_snapshot = c_winyah_data[nn]
        for station in stations
            key = (station.river, station.location)
            santee_profiles[key][] = masked_profile(santee_snapshot, station)
            winyah_profiles[key][] = masked_profile(winyah_snapshot, station)
        end
        time_label[] = time_string(nn)
    end

    @info "Saved high-resolution tracer vertical-profile animation" output_file
    return output_file
end

SURFACE_ONLY || save_vertical_profile_animation()
PROFILES_ONLY || save_surface_tracer_animation()
