using Oceananigans
using CairoMakie

const DEFAULT_DATA_DIR =
    "/mnt/workdir/jliu1/FFTPCG/Data/" *
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee500_" *
    "SouthOutflow_Nx800_Ny1600_Nz40_Wind3p0ms_NE_" *
    "calm12h_ramp12h_hold0h_total24h"
const DATA_DIR = get(ENV, "FFTPCG_DATA_DIR", DEFAULT_DATA_DIR)
const DATA_FILE = joinpath(DATA_DIR, "instantaneous_fields.jld2")
isfile(DATA_FILE) || error("Missing field data: $DATA_FILE")

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

const times = c_santee_data.times
const Nt = length(times)
Nt > 0 || error("No snapshots found in $DATA_FILE")
const x_m = collect(xnodes(c_santee_data.grid, Center()))
const y_m = collect(ynodes(c_santee_data.grid, Center()))
const z_m = collect(znodes(c_santee_data.grid, Center()))
const x_km = x_m ./ 1e3
const y_km = y_m ./ 1e3

const XZ_SECTIONS = (
    ("Santee center", nearest_index(y_m, santee_center_y)),
    ("Between rivers", nearest_index(y_m, 0.0)),
    ("Winyah center", nearest_index(y_m, winyah_center_y)),
)
const YZ_SECTIONS = (
    ("Shoreline", nearest_index(x_m, rotated_coast_x)),
    ("2 km offshore", nearest_index(x_m, rotated_coast_x + 2e3)),
    ("5 km offshore", nearest_index(x_m, rotated_coast_x + 5e3)),
)

function masked_xz(snapshot, j; cutoff = 1e-6)
    values = Array(interior(snapshot, :, j, :))
    for k in eachindex(z_m), i in eachindex(x_m)
        wet = water_depth(x_m[i], y_m[j]) > 0 &&
              z_m[k] >= -water_depth(x_m[i], y_m[j])
        (!wet || values[i, k] <= cutoff) && (values[i, k] = NaN)
    end
    return values
end

function masked_yz(snapshot, i; cutoff = 1e-6)
    values = Array(interior(snapshot, i, :, :))
    for k in eachindex(z_m), j in eachindex(y_m)
        wet = water_depth(x_m[i], y_m[j]) > 0 &&
              z_m[k] >= -water_depth(x_m[i], y_m[j])
        (!wet || values[j, k] <= cutoff) && (values[j, k] = NaN)
    end
    return values
end

time_string(nn) = "t = $(round(times[nn] / 3600; digits = 1)) hour"

santee_xz = Dict(label => Observable(masked_xz(c_santee_data[1], j))
                 for (label, j) in XZ_SECTIONS)
winyah_xz = Dict(label => Observable(masked_xz(c_winyah_data[1], j))
                 for (label, j) in XZ_SECTIONS)
santee_yz = Dict(label => Observable(masked_yz(c_santee_data[1], i))
                 for (label, i) in YZ_SECTIONS)
winyah_yz = Dict(label => Observable(masked_yz(c_winyah_data[1], i))
                 for (label, i) in YZ_SECTIONS)
time_label = Observable(time_string(1))

fig = Figure(size = (2100, 1300), fontsize = 18, backgroundcolor = :white)
Label(fig[0, 1:3], time_label, fontsize = 26)

santee_plot = nothing
winyah_plot = nothing

for (column, (label, j)) in enumerate(XZ_SECTIONS)
    fixed_y = round(y_km[j]; digits = 2)
    ax = Axis(fig[1, column],
              title = "x–z at $label (y=$fixed_y km)",
              xlabel = "Eastward x (km)", ylabel = "z (m)",
              backgroundcolor = :gray92,
              xgridvisible = false, ygridvisible = false)

    global santee_plot = heatmap!(ax, x_km, z_m, santee_xz[label];
                                  colormap = :viridis,
                                  colorrange = (0.0, 1.0),
                                  nan_color = :transparent,
                                  alpha = 0.76)
    global winyah_plot = heatmap!(ax, x_km, z_m, winyah_xz[label];
                                  colormap = :magma,
                                  colorrange = (0.0, 1.0),
                                  nan_color = :transparent,
                                  alpha = 0.64)
    bottom = [-water_depth(x, y_m[j]) for x in x_m]
    lines!(ax, x_km, bottom; color = :black, linewidth = 2)
    vlines!(ax, -10:2.5:10; color = (:gray20, 0.55),
            linestyle = :dash, linewidth = 1)
    hlines!(ax, -15:2.5:0; color = (:gray20, 0.55),
            linestyle = :dash, linewidth = 1)
    xlims!(ax, extrema(x_km)...)
    ylims!(ax, -15.0, 0.0)
end

for (column, (label, i)) in enumerate(YZ_SECTIONS)
    fixed_x = round(x_km[i]; digits = 2)
    ax = Axis(fig[2, column],
              title = "y–z at $label (x=$fixed_x km)",
              xlabel = "Northward y (km)", ylabel = "z (m)",
              backgroundcolor = :gray92,
              xgridvisible = false, ygridvisible = false)

    heatmap!(ax, y_km, z_m, santee_yz[label];
             colormap = :viridis, colorrange = (0.0, 1.0),
             nan_color = :transparent, alpha = 0.76)
    heatmap!(ax, y_km, z_m, winyah_yz[label];
             colormap = :magma, colorrange = (0.0, 1.0),
             nan_color = :transparent, alpha = 0.64)
    bottom = [-water_depth(x_m[i], y) for y in y_m]
    lines!(ax, y_km, bottom; color = :black, linewidth = 2)
    vlines!(ax, -20:5:20; color = (:gray20, 0.55),
            linestyle = :dash, linewidth = 1)
    hlines!(ax, -15:2.5:0; color = (:gray20, 0.55),
            linestyle = :dash, linewidth = 1)
    xlims!(ax, extrema(y_km)...)
    ylims!(ax, -15.0, 0.0)
end

Colorbar(fig[1:2, 4], santee_plot; label = "Santee tracer")
Colorbar(fig[1:2, 5], winyah_plot; label = "Winyah tracer")

output_file = joinpath(DATA_DIR, "tracer_xz_yz_transects.mp4")
CairoMakie.record(fig, output_file, 1:Nt; framerate = 6) do nn
    santee_snapshot = c_santee_data[nn]
    winyah_snapshot = c_winyah_data[nn]

    for (label, j) in XZ_SECTIONS
        santee_xz[label][] = masked_xz(santee_snapshot, j)
        winyah_xz[label][] = masked_xz(winyah_snapshot, j)
    end
    for (label, i) in YZ_SECTIONS
        santee_yz[label][] = masked_yz(santee_snapshot, i)
        winyah_yz[label][] = masked_yz(winyah_snapshot, i)
    end
    time_label[] = time_string(nn)
end

@info "Saved high-resolution x-z and y-z tracer transect animation" output_file
