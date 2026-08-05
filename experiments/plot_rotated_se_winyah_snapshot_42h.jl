using Oceananigans
using CairoMakie

const DATA_ROOT = get(ENV, "FFTPCG_DATA_ROOT", "/mnt/workdir/jliu1/FFTPCG/Data")
const OUTPUT_ROOT = get(ENV, "FFTPCG_OUTPUT_ROOT", joinpath(@__DIR__, "..", "Data"))
const CASE_SUFFIX = "_calm12h_ramp12h_hold72h_total96h"
const TARGET_TIME = 72 * 3600.0

const WITH_SANTEE_SE =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee500_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_SE" * CASE_SUFFIX
const WITHOUT_SANTEE_SE =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee0_NoSanteeRiver_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_SE" * CASE_SUFFIX

data_file(directory) = joinpath(DATA_ROOT, directory, "instantaneous_fields.jld2")
with_santee = FieldTimeSeries(data_file(WITH_SANTEE_SE), "c_winyah")
without_santee = FieldTimeSeries(data_file(WITHOUT_SANTEE_SE), "c_winyah")

frame = argmin(abs.(with_santee.times .- TARGET_TIME))
snapshot_time = with_santee.times[frame]
isapprox(snapshot_time, TARGET_TIME; atol = 1e-8, rtol = 0) ||
    error("No output exists at t = 72.0 h; nearest time is $(snapshot_time / 3600) h")
isapprox(without_santee.times[frame], TARGET_TIME; atol = 1e-8, rtol = 0) ||
    error("No matching no-Santee output exists at t = 72.0 h")

x_m = collect(xnodes(with_santee.grid, Center()))
y_m = collect(ynodes(with_santee.grid, Center()))
z_m = collect(znodes(with_santee.grid, Center()))
x_km = x_m ./ 1e3
y_km = y_m ./ 1e3
k_surface = length(z_m)

function masked_surface(snapshot; cutoff = 1e-6)
    values = Array(interior(snapshot, :, :, k_surface))
    values[values .<= cutoff] .= NaN
    return values
end

with_surface = masked_surface(with_santee[frame])
without_surface = masked_surface(without_santee[frame])

fig = Figure(size = (1050, 700), fontsize = 20, backgroundcolor = :white)
titles = ("With Santee River Plume", "Without Santee River Plume")
values = (with_surface, without_surface)
plots = Any[]

for column in 1:2
    ax = Axis(fig[1, column], title = titles[column],
              xlabel = "Cross-shore distance [km]",
              ylabel = column == 1 ? "Along-shore distance [km]" : "",
              backgroundcolor = :white, aspect = DataAspect(),
              xticks = -10:5:10, yticks = -20:10:20,
              xgridvisible = false, ygridvisible = false)
    xlims!(ax, extrema(x_km)...)
    ylims!(ax, extrema(y_km)...)
    column == 2 && hideydecorations!(ax; grid = false)

    plot = heatmap!(ax, x_km, y_km, values[column];
                    colormap = :magma, colorrange = (0.0, 1.0),
                    nan_color = :transparent)
    vlines!(ax, -10:5:10; color = (:gray30, 0.75),
            linestyle = :dash, linewidth = 1)
    hlines!(ax, -20:10:20; color = (:gray30, 0.75),
            linestyle = :dash, linewidth = 1)
    # Meteorological SE wind blows toward the northwest: (-x, +y).
    arrows!(ax, [6.5], [-14.0], [-3.5], [3.5];
            color = :white, linewidth = 7, arrowsize = 24)
    arrows!(ax, [6.5], [-14.0], [-3.5], [3.5];
            color = :black, linewidth = 4, arrowsize = 20)
    text!(ax, 6.5, -15.5; text = "Wind", color = :black,
          fontsize = 18, align = (:center, :top))
    push!(plots, plot)
end

Colorbar(fig[1, 3], plots[1]; label = "Winyah tracer")
colgap!(fig.layout, 1, 8)

output_dir = joinpath(OUTPUT_ROOT,
                      "MAMD_RotatedCCW_SE_Winyah_NW_Santee_2x2_comparison_3p0ms")
mkpath(output_dir)
output_file = joinpath(output_dir, "surface_winyah_tracer_SE_72h.png")
save(output_file, fig; px_per_unit = 2)

@info "Saved SE Winyah tracer snapshot" output_file snapshot_time
