# Plot the analytic initial buoyancy field on the 90-degree counter-clockwise
# rotated, 20 km east-west by 40 km north-south domain.
include(joinpath(@__DIR__, "winyah_jetty_bathymetry_rotated_ccw_H200.jl"))

const buoyancy_Lz = 20.0
const buoyancy_Nz = 20
const buoyancy_Δz = buoyancy_Lz / buoyancy_Nz
const inlet_depth = 5.0
const shelf_N² = 1e-5
const santee_N² = 8e-3
const winyah_N² = 1.6e-2
const river_bottom_buoyancy = -shelf_N² * inlet_depth
const river_shelf_transition_length = 1e3
const channel_edge_transition_width = 200.0

z_m = collect(range(-buoyancy_Lz + buoyancy_Δz / 2,
                    -buoyancy_Δz / 2;
                    length = buoyancy_Nz))

@inline ambient_buoyancy(z) = shelf_N² * z
@inline santee_buoyancy_profile(z) =
    river_bottom_buoyancy + santee_N² * (z + inlet_depth)
@inline winyah_buoyancy_profile(z) =
    river_bottom_buoyancy + winyah_N² * (z + inlet_depth)

@inline function buoyancy_smoothstep(η)
    η = clamp(η, 0.0, 1.0)
    return η^2 * (3 - 2η)
end

@inline function initial_river_weight(x, y, z, channel_center)
    depth = water_depth(x, y)
    (depth > 0 && z >= -depth && z >= -inlet_depth) || return 0.0

    inner_half_width = inlet_width / 2 - channel_edge_transition_width / 2
    cross_channel_coordinate = (abs(x - channel_center) - inner_half_width) /
                               channel_edge_transition_width
    cross_channel_weight = 1 - buoyancy_smoothstep(cross_channel_coordinate)

    shelfward_edge = river_mouth_y - river_shelf_transition_length
    along_channel_coordinate = (y - shelfward_edge) / (y₁ - shelfward_edge)
    along_channel_weight = buoyancy_smoothstep(along_channel_coordinate)

    return cross_channel_weight * along_channel_weight
end

@inline function initial_buoyancy(x, y, z)
    background = ambient_buoyancy(z)
    santee_weight = initial_river_weight(x, y, z, santee_center_x)
    winyah_weight = initial_river_weight(x, y, z, winyah_center_x)

    return background +
           santee_weight * (santee_buoyancy_profile(z) - background) +
           winyah_weight * (winyah_buoyancy_profile(z) - background)
end

# The inverse rotation is (x, y) = (y_rotated, -x_rotated).
@inline rotated_initial_buoyancy(x_rotated, y_rotated, z) =
    initial_buoyancy(y_rotated, -x_rotated, z)

function masked_rotated_buoyancy(x_rotated, y_rotated, z)
    x = y_rotated
    y = -x_rotated
    depth = water_depth(x, y)
    return depth > 0 && z >= -depth ? initial_buoyancy(x, y, z) : NaN
end

surface_z = z_m[end]
surface_buoyancy = [masked_rotated_buoyancy(x, y, surface_z)
                    for x in x_rotated_m, y in y_rotated_m]

santee_y_rotated = santee_center_x
winyah_y_rotated = winyah_center_x
santee_section = [masked_rotated_buoyancy(x, santee_y_rotated, z)
                  for x in x_rotated_m, z in z_m]
winyah_section = [masked_rotated_buoyancy(x, winyah_y_rotated, z)
                  for x in x_rotated_m, z in z_m]

buoyancy_range = (-1e-4, winyah_buoyancy_profile(surface_z))
buoyancy_fig = Figure(size = (1550, 1100), fontsize = 18)

surface_ax = Axis(buoyancy_fig[1:2, 1],
                  title = "Initial surface buoyancy at z = $(surface_z) m",
                  xlabel = "Eastward x (km)",
                  ylabel = "Northward y (km)",
                  aspect = DataAspect())

surface_hm = heatmap!(surface_ax,
                      x_rotated_km,
                      y_rotated_km,
                      surface_buoyancy;
                      colormap = :thermal,
                      colorrange = buoyancy_range,
                      nan_color = :lightgray)

vlines!(surface_ax, [rotated_coast_x / 1e3];
        color = :white, linewidth = 2, linestyle = :dot)
hlines!(surface_ax, [santee_y_rotated / 1e3];
        color = :dodgerblue, linewidth = 2, linestyle = :dash)
hlines!(surface_ax, [winyah_y_rotated / 1e3];
        color = :red, linewidth = 2, linestyle = :dash)

santee_ax = Axis(buoyancy_fig[2, 2],
                 title = "Santee centerline, y = -5 km",
                 xlabel = "Eastward x (km)",
                 ylabel = "z (m)")

winyah_ax = Axis(buoyancy_fig[1, 2],
                 title = "Winyah centerline, y = 5 km",
                 xlabel = "Eastward x (km)",
                 ylabel = "z (m)")

heatmap!(santee_ax, x_rotated_km, z_m, santee_section;
         colormap = :thermal, colorrange = buoyancy_range,
         nan_color = :lightgray)
heatmap!(winyah_ax, x_rotated_km, z_m, winyah_section;
         colormap = :thermal, colorrange = buoyancy_range,
         nan_color = :lightgray)

for ax in (santee_ax, winyah_ax)
    vlines!(ax, [rotated_coast_x / 1e3];
            color = :white, linewidth = 2, linestyle = :dot)
    xlims!(ax, -10, 10)
    ylims!(ax, -20, 0)
end

Colorbar(buoyancy_fig[:, 3], surface_hm; label = "Buoyancy b (m s⁻²)")

buoyancy_output_file = joinpath(@__DIR__,
                                "winyah_initial_buoyancy_rotated_ccw.png")
save(buoyancy_output_file, buoyancy_fig)

santee_surface_buoyancy = santee_buoyancy_profile(surface_z)
winyah_surface_buoyancy = winyah_buoyancy_profile(surface_z)
@info "Saved rotated initial buoyancy field" buoyancy_output_file santee_surface_buoyancy winyah_surface_buoyancy
