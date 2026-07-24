using CairoMakie

####
#### Rotated 20 km × 40 km domain
####

const Lx = 20e3
const Ly = 40e3
const Δx = 100.0
const Δy = 100.0
const Nx = Int(Lx / Δx)
const Ny = Int(Ly / Δy)
const x₀, x₁ = -Lx / 2, Lx / 2
const y₀, y₁ = -Ly / 2, Ly / 2

const inlet_width = 1000.0
const inlet_depth = 5.0
const santee_center_y = -5e3

####
#### Geometry in the original orientation
####

const original_y₀ = -7.5e3
const original_y₁ = 7.5e3
const original_river_mouth_y = original_y₁ - 200.0
const rotated_coast_x = -original_river_mouth_y
const nearshore_slope_length = 3e3
const nearshore_slope_depth = 5.0
const slope_depth = 15.0

const santee_channel_length = 3e3
const original_santee_channel_south_y = original_river_mouth_y - santee_channel_length
const original_santee_channel_x = (santee_center_y - inlet_width / 2,
                                    santee_center_y + inlet_width / 2)

@inline in_original_santee_embayment(x) = abs(x - santee_center_y) <= inlet_width / 2

@inline function in_original_santee_navigation_channel(x, y)
    within_channel = original_santee_channel_x[1] < x < original_santee_channel_x[2]
    along_channel = original_santee_channel_south_y <= y <= original_river_mouth_y
    return within_channel && along_channel
end

@inline function original_shelf_depth(x, y)
    if y > original_river_mouth_y
        return in_original_santee_embayment(x) ? inlet_depth : 0.0
    end

    offshore_distance = original_river_mouth_y - y
    if offshore_distance <= nearshore_slope_length
        fraction = clamp(offshore_distance / nearshore_slope_length, 0.0, 1.0)
        return nearshore_slope_depth * fraction
    end

    outer_length = original_river_mouth_y - original_y₀ - nearshore_slope_length
    outer_fraction = clamp((offshore_distance - nearshore_slope_length) /
                           outer_length, 0.0, 1.0)
    return nearshore_slope_depth +
           (slope_depth - nearshore_slope_depth) * outer_fraction
end

@inline function original_water_depth_without_winyah_bay(x, y)
    in_original_santee_navigation_channel(x, y) && return inlet_depth

    # There is no Winyah embayment, river mouth, navigation channel, or jetty.
    # The former Winyah location follows the ordinary coastline and shelf.
    return original_shelf_depth(x, y)
end

# Inverse of the physical 90° counter-clockwise rotation:
# (x_original, y_original) = (y_rotated, -x_rotated).
@inline water_depth(x, y) = original_water_depth_without_winyah_bay(y, -x)

@assert water_depth(rotated_coast_x, santee_center_y) == inlet_depth
@assert water_depth(-original_santee_channel_south_y, santee_center_y) == inlet_depth
@assert water_depth(rotated_coast_x, 5e3) == 0.0

####
#### Bathymetry figure
####

x_m = collect(range(x₀ + Δx / 2, x₁ - Δx / 2; length = Nx))
y_m = collect(range(y₀ + Δy / 2, y₁ - Δy / 2; length = Ny))
x_km = x_m ./ 1e3
y_km = y_m ./ 1e3
depth = [water_depth(x, y) for x in x_m, y in y_m]

fig = Figure(size = (1550, 950), fontsize = 20, backgroundcolor = :white)

ax_full = Axis(fig[1, 1],
               title = "Rotated bathymetry — Winyah Bay removed",
               xlabel = "Eastward x (km)",
               ylabel = "Northward y (km)",
               aspect = DataAspect())

hm = heatmap!(ax_full, x_km, y_km, depth;
              colormap = :deep, colorrange = (0, slope_depth))

ax_zoom = Axis(fig[1, 2],
               title = "Santee river-mouth zoom",
               xlabel = "Eastward x (km)",
               ylabel = "Northward y (km)",
               aspect = DataAspect())

heatmap!(ax_zoom, x_km, y_km, depth;
         colormap = :deep, colorrange = (0, slope_depth))

rotated_channel_x = collect((-original_river_mouth_y,
                              -original_santee_channel_south_y)) ./ 1e3
for ax in (ax_full, ax_zoom)
    vlines!(ax, [rotated_coast_x / 1e3];
            color = :black, linewidth = 2, linestyle = :dot)

    lines!(ax,
           [rotated_channel_x[1], rotated_channel_x[2],
            rotated_channel_x[2], rotated_channel_x[1]] ,
           [original_santee_channel_x[1], original_santee_channel_x[1],
            original_santee_channel_x[2], original_santee_channel_x[2]] ./ 1e3;
           color = :dodgerblue, linewidth = 3, linestyle = :dash)
end

text!(ax_zoom, -4.05, santee_center_y / 1e3;
      text = "Santee channel", color = :dodgerblue,
      align = (:right, :center))

xlims!(ax_full, extrema(x_km)...)
ylims!(ax_full, extrema(y_km)...)
xlims!(ax_zoom, -8.0, -3.8)
ylims!(ax_zoom, -8.0, 8.0)

Colorbar(fig[1, 3], hm; label = "Water depth (m)")

output_file = joinpath(@__DIR__,
                       "santee_bathymetry_rotated_ccw_no_winyah_bay.png")
save(output_file, fig)

@info "Saved rotated bathymetry with Santee River only" output_file
