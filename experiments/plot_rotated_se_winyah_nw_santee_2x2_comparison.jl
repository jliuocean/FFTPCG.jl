using Oceananigans
using CairoMakie

const DATA_ROOT = get(ENV, "FFTPCG_DATA_ROOT", "/mnt/workdir/jliu1/FFTPCG/Data")
const OUTPUT_ROOT = get(ENV, "FFTPCG_OUTPUT_ROOT",
                        joinpath(@__DIR__, "..", "Data"))
const CASE_SUFFIX = "_calm12h_ramp12h_hold72h_total96h"

const WITH_SANTEE_SE =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee500_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_SE" * CASE_SUFFIX
const WITHOUT_SANTEE_SE =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee0_NoSanteeRiver_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_SE" * CASE_SUFFIX
const WITH_WINYAH_NW =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee500_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_NW" * CASE_SUFFIX
const WITHOUT_WINYAH_NW =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah0_Santee500_NoWinyahBay_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_NW" * CASE_SUFFIX

data_file(directory) = joinpath(DATA_ROOT, directory, "instantaneous_fields.jld2")

const INPUTS = (
    (WITH_SANTEE_SE, "c_winyah"),
    (WITHOUT_SANTEE_SE, "c_winyah"),
    (WITH_WINYAH_NW, "c_santee"),
    (WITHOUT_WINYAH_NW, "c_santee"),
)

for (directory, tracer) in INPUTS
    filepath = data_file(directory)
    isfile(filepath) || error("Missing $tracer data: $filepath")
end

series = [FieldTimeSeries(data_file(directory), tracer)
          for (directory, tracer) in INPUTS]
times = series[1].times
for data in series[2:end]
    length(data.times) == length(times) || error("Time-count mismatch")
    all(isapprox.(data.times, times; atol = 1e-8, rtol = 0)) ||
        error("Time-coordinate mismatch")
end

x_m = collect(xnodes(series[1].grid, Center()))
y_m = collect(ynodes(series[1].grid, Center()))
z_m = collect(znodes(series[1].grid, Center()))
x_km = x_m ./ 1e3
y_km = y_m ./ 1e3
k_surface = length(z_m)

function masked_surface(snapshot; cutoff = 1e-6)
    values = Array(interior(snapshot, :, :, k_surface))
    values[values .<= cutoff] .= NaN
    return values
end

n = Observable(1)
surfaces = [lift(nn -> masked_surface(data[nn]), n) for data in series]

fig = Figure(size = (1450, 1500), fontsize = 20, backgroundcolor = :white)
title_text = Observable(
    "SE Winyah plume and NW Santee plume comparisons — " *
    "t = $(round(times[1] / 3600; digits = 1)) hour"
)
Label(fig[0, 1:2], title_text, fontsize = 28)

titles = (
    "SE wind — Winyah tracer — with Santee",
    "SE wind — Winyah tracer — no Santee",
    "NW wind — Santee tracer — with Winyah",
    "NW wind — Santee tracer — no Winyah",
)
positions = ((1, 1), (1, 2), (2, 1), (2, 2))
colormaps = (:magma, :magma, :viridis, :viridis)
axes = Axis[]
plots = Any[]

for index in eachindex(series)
    row, column = positions[index]
    ax = Axis(fig[row, column], title = titles[index],
              xlabel = row == 2 ? "x (km)" : "",
              ylabel = column == 1 ? "northward y (km)" : "",
              backgroundcolor = :white, aspect = DataAspect(),
              xticks = -10:5:10, yticks = -20:10:20,
              xgridvisible = false, ygridvisible = false)
    xlims!(ax, extrema(x_km)...)
    ylims!(ax, extrema(y_km)...)
    row == 1 && hidexdecorations!(ax; grid = false)
    column == 2 && hideydecorations!(ax; grid = false)

    plot = heatmap!(ax, x_km, y_km, surfaces[index];
                    colormap = colormaps[index], colorrange = (0.0, 1.0),
                    nan_color = :transparent)
    vlines!(ax, -10:5:10; color = (:gray30, 0.75),
            linestyle = :dash, linewidth = 1)
    hlines!(ax, -20:10:20; color = (:gray30, 0.75),
            linestyle = :dash, linewidth = 1)
    push!(axes, ax)
    push!(plots, plot)
end

Colorbar(fig[1, 3], plots[1]; label = "Winyah tracer")
Colorbar(fig[2, 3], plots[3]; label = "Santee tracer")

output_dir = joinpath(OUTPUT_ROOT,
                      "MAMD_RotatedCCW_SE_Winyah_NW_Santee_2x2_comparison_3p0ms")
mkpath(output_dir)
output_file = joinpath(output_dir,
                       "surface_tracers_SE_Winyah_NW_Santee_2x2_3ms.mp4")

CairoMakie.record(fig, output_file, 1:length(times); framerate = 6) do nn
    n[] = nn
    title_text[] = "SE Winyah plume and NW Santee plume comparisons — " *
                   "t = $(round(times[nn] / 3600; digits = 1)) hour"
end

@info "Saved SE Winyah and NW Santee 2×2 comparison" output_file
