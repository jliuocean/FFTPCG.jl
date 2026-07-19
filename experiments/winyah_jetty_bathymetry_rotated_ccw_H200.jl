# Reuse the verified 40 km Winyah/Santee bathymetry, then rotate the map
# 90 degrees counter-clockwise: x_rot = -y and y_rot = x.
include(joinpath(@__DIR__, "winyah_jetty_bathymetry_H200.jl"))

const Lx_rotated = 20e3
const Ly_rotated = Lx
const Nx_rotated = Int(Lx_rotated / Δy)
const Ny_rotated = Int(Ly_rotated / Δx)
const x_rotated_0 = -Lx_rotated / 2
const x_rotated_1 =  Lx_rotated / 2
const y_rotated_0 = -Ly_rotated / 2
const y_rotated_1 =  Ly_rotated / 2

x_rotated_m = collect(range(x_rotated_0 + Δy / 2,
                            x_rotated_1 - Δy / 2;
                            length = Nx_rotated))
y_rotated_m = collect(range(y_rotated_0 + Δx / 2,
                            y_rotated_1 - Δx / 2;
                            length = Ny_rotated))

x_rotated_km = x_rotated_m ./ 1e3
y_rotated_km = y_rotated_m ./ 1e3

# The inverse of (x_rot, y_rot) = (-y, x) is (x, y) = (y_rot, -x_rot).
rotated_depth = [water_depth(y_rotated, -x_rotated)
                 for x_rotated in x_rotated_m,
                     y_rotated in y_rotated_m]

rotated_fig = Figure(size = (1500, 900), fontsize = 18)

rotated_full_ax = Axis(rotated_fig[1, 1],
                       title = "Bathymetry rotated 90° counter-clockwise",
                       xlabel = "Rotated x / eastward (km)",
                       ylabel = "Rotated y / northward (km)",
                       aspect = DataAspect())

rotated_hm = heatmap!(rotated_full_ax,
                      x_rotated_km,
                      y_rotated_km,
                      rotated_depth;
                      colormap = :deep,
                      colorrange = (0, slope_depth))

rotated_zoom_ax = Axis(rotated_fig[1, 2],
                       title = "Rotated river mouths and 3 km extensions",
                       xlabel = "Rotated x / eastward (km)",
                       ylabel = "Rotated y / northward (km)",
                       aspect = DataAspect())

heatmap!(rotated_zoom_ax,
         x_rotated_km,
         y_rotated_km,
         rotated_depth;
         colormap = :deep,
         colorrange = (0, slope_depth))

rotated_coast_x = -river_mouth_y
rotated_jetty_x = (-river_mouth_y, -jetty_south_y)

for ax in (rotated_full_ax, rotated_zoom_ax)
    vlines!(ax, [rotated_coast_x / 1e3];
            color = :black,
            linewidth = 2,
            linestyle = :dot)

    for jetty_y in winyah_jetty_x
        lines!(ax,
               collect(rotated_jetty_x) ./ 1e3,
               fill(jetty_y / 1e3, 2);
               color = :red,
               linewidth = 3)
    end

    lines!(ax,
           [-river_mouth_y, -santee_channel_south_y,
            -santee_channel_south_y, -river_mouth_y] ./ 1e3,
           [santee_channel_x[1], santee_channel_x[1],
            santee_channel_x[2], santee_channel_x[2]] ./ 1e3;
           color = :dodgerblue,
           linewidth = 3,
           linestyle = :dash)
end

xlims!(rotated_zoom_ax,
       (rotated_coast_x - 0.5e3) / 1e3,
       (-jetty_south_y + 0.5e3) / 1e3)
ylims!(rotated_zoom_ax, -8, 8)

text!(rotated_zoom_ax, -6.8, inlet_centers[1] / 1e3;
      text = "Santee", color = :dodgerblue, align = (:left, :bottom))
text!(rotated_zoom_ax, -6.8, inlet_centers[2] / 1e3;
      text = "Winyah", color = :red, align = (:left, :bottom))

Colorbar(rotated_fig[1, 3], rotated_hm; label = "Water depth (m)")

rotated_output_file = joinpath(@__DIR__,
                               "winyah_jetty_bathymetry_rotated_ccw.png")
save(rotated_output_file, rotated_fig)

@info "Saved 90-degree counter-clockwise rotated bathymetry map" rotated_output_file
