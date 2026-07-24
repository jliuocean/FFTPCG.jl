using Oceananigans
using CairoMakie

const DATA_ROOT = get(ENV, "FFTPCG_DATA_ROOT", "/mnt/workdir/jliu1/FFTPCG/Data")
const CASES = ("NW", "N", "NE", "W", "E", "SW", "S", "SE")
const PANEL_POSITION = Dict(
    "NW" => (1, 1), "N" => (1, 2), "NE" => (1, 3),
    "W"  => (2, 1),                 "E"  => (2, 3),
    "SW" => (3, 1), "S" => (3, 2), "SE" => (3, 3),
)

const SHARED_PREFIX = "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_"
const WITH_SANTEE_PREFIX = SHARED_PREFIX *
                            "Santee500_SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_"
const WITHOUT_SANTEE_PREFIX = SHARED_PREFIX *
                               "Santee0_NoSanteeRiver_SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_"
const CASE_SUFFIX = "_calm12h_ramp12h_hold72h_total96h"
const OUTPUT_DIR = joinpath(DATA_ROOT,
                            "MAMD_RotatedCCW_8wind_comparison_" *
                            "Winyah375_WithSantee500_vs_NoSantee_3p0ms")
mkpath(OUTPUT_DIR)

case_file(prefix, case) = joinpath(DATA_ROOT, prefix * case * CASE_SUFFIX,
                                   "instantaneous_fields.jld2")

for case in CASES
    isfile(case_file(WITH_SANTEE_PREFIX, case)) ||
        error("Missing with-Santee data for $case")
    isfile(case_file(WITHOUT_SANTEE_PREFIX, case)) ||
        error("Missing no-Santee data for $case")
end

with_santee = Dict(case => FieldTimeSeries(case_file(WITH_SANTEE_PREFIX, case), "b")
                    for case in CASES)
without_santee = Dict(case => FieldTimeSeries(case_file(WITHOUT_SANTEE_PREFIX, case), "b")
                       for case in CASES)

times = with_santee["NW"].times
for group in (with_santee, without_santee), case in CASES
    length(group[case].times) == length(times) || error("Time-count mismatch for $case")
    all(isapprox.(group[case].times, times; atol = 1e-8, rtol = 0)) ||
        error("Time-coordinate mismatch for $case")
end

reference = with_santee["NW"]
x_m = collect(xnodes(reference.grid, Center()))
y_m = collect(ynodes(reference.grid, Center()))
z_m = collect(znodes(reference.grid, Center()))
x_km = x_m ./ 1e3
y_km = y_m ./ 1e3
k_surface = length(z_m)
z_surface = z_m[k_surface]

const ρ₀ = 1025.0
const shelf_N² = 1e-5
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

@inline function in_original_winyah_jetty(x, y)
    along = original_jetty_south_y <= y <= original_river_mouth_y
    south_side = abs(x - original_winyah_jetty_x[1]) <= jetty_width / 2
    north_side = abs(x - original_winyah_jetty_x[2]) <= jetty_width / 2
    return along && (south_side || north_side)
end

@inline in_original_winyah_channel(x, y) =
    original_winyah_jetty_x[1] < x < original_winyah_jetty_x[2] &&
    original_jetty_south_y <= y <= original_river_mouth_y

@inline function original_shelf_depth(x, y, no_santee)
    if y > original_river_mouth_y
        in_embayment = in_original_winyah_embayment(x) ||
                       (!no_santee && in_original_santee_embayment(x))
        return in_embayment ? inlet_depth : 0.0
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

@inline function original_water_depth(x, y, no_santee)
    in_original_winyah_jetty(x, y) && return 0.0
    in_original_winyah_channel(x, y) && return inlet_depth
    in_santee_channel = !no_santee &&
                         original_santee_channel_x[1] < x < original_santee_channel_x[2] &&
                         original_santee_channel_south_y <= y <= original_river_mouth_y
    in_santee_channel && return inlet_depth
    return original_shelf_depth(x, y, no_santee)
end

@inline water_depth(x, y, no_santee) = original_water_depth(y, -x, no_santee)
wet_with_santee = [water_depth(x, y, false) > 0 &&
                    z_surface >= -water_depth(x, y, false) for x in x_m, y in y_m]
wet_without_santee = [water_depth(x, y, true) > 0 &&
                       z_surface >= -water_depth(x, y, true) for x in x_m, y in y_m]

function density_surface(snapshot, wet)
    buoyancy = Array(interior(snapshot, :, :, k_surface))
    density = -ρ₀ .* (buoyancy .- shelf_N² * z_surface) ./ 9.81
    density[.!wet] .= NaN
    return density
end

n = Observable(1)
with_surface = Dict(case => lift(nn -> density_surface(with_santee[case][nn],
                                                       wet_with_santee), n)
                    for case in CASES)
without_surface = Dict(case => lift(nn -> density_surface(without_santee[case][nn],
                                                          wet_without_santee), n)
                       for case in CASES)

fig = Figure(size = (2300, 2100), fontsize = 16, backgroundcolor = :white)
Label(fig[0, 1:3],
      "Surface density anomaly: influence of removing the Santee River",
      fontsize = 28)

plots = Dict{String, Any}()
for case in CASES
    row, column = PANEL_POSITION[case]
    panel = GridLayout(fig[row, column])
    ax_with = Axis(panel[1, 1], title = "$case wind — with Santee",
                   backgroundcolor = :white, aspect = DataAspect(),
                   xticks = -10:5:10, yticks = -20:10:20,
                   xgridvisible = false, ygridvisible = false)
    ax_without = Axis(panel[1, 2], title = "$case wind — no Santee",
                      backgroundcolor = :white, aspect = DataAspect(),
                      xticks = -10:5:10, yticks = -20:10:20,
                      xgridvisible = false, ygridvisible = false)

    for ax in (ax_with, ax_without)
        xlims!(ax, extrema(x_km)...)
        ylims!(ax, extrema(y_km)...)
    end
    hideydecorations!(ax_without; grid = false)
    if row != 3
        hidexdecorations!(ax_with; grid = false)
        hidexdecorations!(ax_without; grid = false)
    end
    column == 1 || hideydecorations!(ax_with; grid = false)

    plots[case] = heatmap!(ax_with, x_km, y_km, with_surface[case];
                           colormap = :balance, colorrange = (-12.0, 12.0),
                           nan_color = :transparent)
    heatmap!(ax_without, x_km, y_km, without_surface[case];
             colormap = :balance, colorrange = (-12.0, 12.0),
             nan_color = :transparent)
    for ax in (ax_with, ax_without)
        vlines!(ax, -10:5:10; color = (:gray30, 0.75),
                linestyle = :dash, linewidth = 1)
        hlines!(ax, -20:10:20; color = (:gray30, 0.75),
                linestyle = :dash, linewidth = 1)
    end
end

time_text = Observable("t = $(round(times[1] / 3600; digits = 1)) hour")
center_text = lift(time_text) do label
    "$label\n\nLeft: Santee 500 + Winyah 375 m³ s⁻¹\n" *
    "Right: Santee removed + Winyah 375 m³ s⁻¹"
end
Label(fig[2, 2], center_text, fontsize = 20,
      tellwidth = false, tellheight = false, justification = :center)
Colorbar(fig[1:3, 4], plots["NW"]; label = "Density anomaly (kg m⁻³)")

output_file = joinpath(OUTPUT_DIR,
                       "surface_density_8cases_" *
                       "with_Santee500_vs_no_Santee_3ms_no_wind_text.mp4")
CairoMakie.record(fig, output_file, 1:length(times); framerate = 6) do nn
    n[] = nn
    time_text[] = "t = $(round(times[nn] / 3600; digits = 1)) hour"
end

@info "Saved with-vs-without-Santee surface density comparison" output_file
