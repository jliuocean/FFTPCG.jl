using Oceananigans
using CairoMakie

const DATA_ROOT = get(ENV, "FFTPCG_DATA_ROOT", "/mnt/workdir/jliu1/FFTPCG/Data")
const OUTPUT_ROOT = get(ENV, "FFTPCG_OUTPUT_ROOT", joinpath(@__DIR__, "..", "Data"))
const OUTPUT_DIR = joinpath(OUTPUT_ROOT, "Complementary_modeling_panels_D_to_G")
const CASE_SUFFIX = "_calm12h_ramp12h_hold72h_total96h"
const FIGURE_SIZE = (1200, 900) # Same 4:3 aspect ratio for PDF panels D--G.

const WITH_BOTH_SE =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee500_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_SE" * CASE_SUFFIX
const NO_SANTEE_SE =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee0_NoSanteeRiver_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_SE" * CASE_SUFFIX
const WITH_BOTH_NW =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah375_Santee500_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_NW" * CASE_SUFFIX
const NO_WINYAH_NW =
    "MAMD_RotatedCCW_Lx20km_Ly40km_Winyah0_Santee500_NoWinyahBay_" *
    "SouthOutflow_Nx200_Ny400_Nz20_Wind3p0ms_NW" * CASE_SUFFIX

data_file(directory) = joinpath(DATA_ROOT, directory, "instantaneous_fields.jld2")

function surface_at_time(directory, tracer, target_hours)
    data = FieldTimeSeries(data_file(directory), tracer)
    target_time = target_hours * 3600.0
    frame = argmin(abs.(data.times .- target_time))
    actual_time = data.times[frame]
    isapprox(actual_time, target_time; atol = 1e-8, rtol = 0) ||
        error("No output at t = $target_hours h for $directory")

    x_m = collect(xnodes(data.grid, Center()))
    y_m = collect(ynodes(data.grid, Center()))
    z_m = collect(znodes(data.grid, Center()))
    values = Array(interior(data[frame], :, :, length(z_m)))
    values[values .<= 1e-6] .= NaN
    return x_m ./ 1e3, y_m ./ 1e3, values
end

function save_panel(filename, title, directory, tracer, target_hours,
                    colormap, colorbar_label, wind_case)
    x_km, y_km, values = surface_at_time(directory, tracer, target_hours)

    fig = Figure(size = FIGURE_SIZE, fontsize = 28, backgroundcolor = :white)
    ax = Axis(fig[1, 1], title = title,
              xlabel = "Cross-shore distance [km]",
              ylabel = "Along-shore distance [km]",
              backgroundcolor = :white,
              xticks = -10:5:10, yticks = -20:10:20,
              xgridvisible = false, ygridvisible = false)
    xlims!(ax, extrema(x_km)...)
    ylims!(ax, extrema(y_km)...)

    hm = heatmap!(ax, x_km, y_km, values;
                  colormap, colorrange = (0.0, 1.0),
                  nan_color = :transparent)
    vlines!(ax, -10:5:10; color = (:gray30, 0.75),
            linestyle = :dash, linewidth = 1.5)
    hlines!(ax, -20:10:20; color = (:gray30, 0.75),
            linestyle = :dash, linewidth = 1.5)

    if wind_case == :SE
        # SE wind blows northwestward: (-x, +y).
        origin_x, origin_y, vector_x, vector_y = 6.5, -14.0, -3.5, 3.5
        text_x, text_y, text_align = 6.5, -15.5, (:center, :top)
    elseif wind_case == :NW
        # NW wind blows southeastward: (+x, -y).
        origin_x, origin_y, vector_x, vector_y = 3.0, 15.0, 3.5, -3.5
        text_x, text_y, text_align = 2.7, 16.0, (:center, :bottom)
    else
        error("Unsupported wind case: $wind_case")
    end

    arrows!(ax, [origin_x], [origin_y], [vector_x], [vector_y];
            color = :white, linewidth = 9, arrowsize = 30)
    arrows!(ax, [origin_x], [origin_y], [vector_x], [vector_y];
            color = :black, linewidth = 5, arrowsize = 25)
    text!(ax, text_x, text_y; text = "Wind", color = :black,
          fontsize = 26, align = text_align)

    Colorbar(fig[1, 2], hm; label = colorbar_label)
    colgap!(fig.layout, 1, 18)
    save(joinpath(OUTPUT_DIR, filename), fig; px_per_unit = 1)
end

mkpath(OUTPUT_DIR)

# Match the single/interacting ordering described by the PDF caption.
save_panel("panel_D_Winyah_single_plume_SE_72h.png",
           "Without Santee River Plume", NO_SANTEE_SE, "c_winyah", 72,
           :magma, "Winyah tracer", :SE)
save_panel("panel_E_Winyah_interacting_plumes_SE_72h.png",
           "With Santee River Plume", WITH_BOTH_SE, "c_winyah", 72,
           :magma, "Winyah tracer", :SE)
save_panel("panel_F_Santee_single_plume_NW_42h.png",
           "Without Winyah Bay Plume", NO_WINYAH_NW, "c_santee", 42,
           :viridis, "Santee tracer", :NW)
save_panel("panel_G_Santee_interacting_plumes_NW_42h.png",
           "With Winyah Bay Plume", WITH_BOTH_NW, "c_santee", 42,
           :viridis, "Santee tracer", :NW)

@info "Saved complementary modeling panels D--G" OUTPUT_DIR FIGURE_SIZE
