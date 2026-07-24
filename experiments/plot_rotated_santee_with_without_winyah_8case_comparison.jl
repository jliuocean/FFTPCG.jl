using Oceananigans
using CairoMakie

const DATA_ROOT = get(ENV, "FFTPCG_DATA_ROOT", "/mnt/workdir/jliu1/FFTPCG/Data")
const CASES = ("NW", "N", "NE", "W", "E", "SW", "S", "SE")
const PANEL_POSITION = Dict(
    "NW" => (1, 1), "N" => (1, 2), "NE" => (1, 3),
    "W"  => (2, 1),                 "E"  => (2, 3),
    "SW" => (3, 1), "S" => (3, 2), "SE" => (3, 3),
)

const WITH_WINYAH_PREFIX =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee500_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_"
const WITHOUT_WINYAH_PREFIX =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah0_Santee500_NoWinyahBay_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_"
const CASE_SUFFIX = "_calm12h_ramp12h_hold72h_total96h"
const OUTPUT_DIR = joinpath(DATA_ROOT,
                            "MAMD_RotatedCCW_8wind_comparison_" *
                            "Santee500_WithWinyah375_vs_NoWinyah_3p0ms")
mkpath(OUTPUT_DIR)

case_file(prefix, case) = joinpath(DATA_ROOT, prefix * case * CASE_SUFFIX,
                                   "instantaneous_fields.jld2")

for case in CASES
    isfile(case_file(WITH_WINYAH_PREFIX, case)) ||
        error("Missing with-Winyah data for $case")
    isfile(case_file(WITHOUT_WINYAH_PREFIX, case)) ||
        error("Missing no-Winyah data for $case")
end

with_winyah = Dict(case => FieldTimeSeries(case_file(WITH_WINYAH_PREFIX, case),
                                            "c_santee") for case in CASES)
without_winyah = Dict(case => FieldTimeSeries(case_file(WITHOUT_WINYAH_PREFIX, case),
                                               "c_santee") for case in CASES)

times = with_winyah["NW"].times
for group in (with_winyah, without_winyah), case in CASES
    length(group[case].times) == length(times) || error("Time-count mismatch for $case")
    all(isapprox.(group[case].times, times; atol = 1e-8, rtol = 0)) ||
        error("Time-coordinate mismatch for $case")
end

reference = with_winyah["NW"]
x_m = collect(xnodes(reference.grid, Center()))
y_m = collect(ynodes(reference.grid, Center()))
z_m = collect(znodes(reference.grid, Center()))
x_km = x_m ./ 1e3
y_km = y_m ./ 1e3
k_surface = length(z_m)

function masked_surface(snapshot; cutoff = 1e-6)
    values = Array(interior(snapshot, :, :, k_surface))
    values[values .<= cutoff] .= NaN
    return values
end

n = Observable(1)
with_surface = Dict(case => lift(nn -> masked_surface(with_winyah[case][nn]), n)
                    for case in CASES)
without_surface = Dict(case => lift(nn -> masked_surface(without_winyah[case][nn]), n)
                       for case in CASES)

fig = Figure(size = (2300, 2100), fontsize = 16, backgroundcolor = :white)
Label(fig[0, 1:3],
      "Surface Santee tracer: influence of removing Winyah Bay",
      fontsize = 28)

plots = Dict{String, Any}()
for case in CASES
    row, column = PANEL_POSITION[case]
    panel = GridLayout(fig[row, column])
    ax_with = Axis(panel[1, 1], title = "$case wind — with Winyah",
                   backgroundcolor = :white, aspect = DataAspect(),
                   xticks = -10:5:10, yticks = -20:10:20,
                   xgridvisible = false, ygridvisible = false)
    ax_without = Axis(panel[1, 2], title = "$case wind — no Winyah",
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
                           colormap = :viridis, colorrange = (0.0, 1.0),
                           nan_color = :transparent)
    heatmap!(ax_without, x_km, y_km, without_surface[case];
             colormap = :viridis, colorrange = (0.0, 1.0),
             nan_color = :transparent)

    # Draw the grid after the tracer so it stays visible above the data.
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
    "Right: Santee 500 m³ s⁻¹, Winyah Bay removed\n\n" *
    "0–12 h: no wind\n12–24 h: ramp to 3 m s⁻¹\n" *
    "24–96 h: constant wind"
end
Label(fig[2, 2], center_text, fontsize = 20,
      tellwidth = false, tellheight = false, justification = :center)
Colorbar(fig[1:3, 4], plots["NW"]; label = "Santee tracer")

output_file = joinpath(OUTPUT_DIR,
                       "surface_santee_tracer_8cases_" *
                       "with_Winyah375_vs_no_Winyah_3ms.mp4")
CairoMakie.record(fig, output_file, 1:length(times); framerate = 6) do nn
    n[] = nn
    time_text[] = "t = $(round(times[nn] / 3600; digits = 1)) hour"
end

@info "Saved with-vs-without-Winyah Santee tracer comparison" output_file
