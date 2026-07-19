using CairoMakie

####
#### Domain and channel geometry (matching buoyancy_flow_mamd_smooth_H200.jl)
####

const Lx = 40e3
const Ly = 15e3
const Lz = 20.0

const Δx = 100.0
const Δy = 100.0
const Nx = Int(Lx / Δx)
const Ny = Int(Ly / Δy)

const x₀ = -Lx / 2
const x₁ =  Lx / 2
const y₀ = -Ly / 2
const y₁ =  Ly / 2

const inlet_width = 1000.0
const inlet_center_spacing = 10e3
const inlet_centers = (-inlet_center_spacing / 2, inlet_center_spacing / 2)
const santee_center_x = inlet_centers[1]
const winyah_center_x = inlet_centers[2]

const santee_channel_depth = 5.0
const winyah_channel_depth = 5.0
const embayment_length_y = 200.0
const river_mouth_y = y₁ - embayment_length_y
const nearshore_slope_length = 3e3
const nearshore_slope_depth = 5.0
const slope_depth = 15.0

####
#### Winyah jetties
####

const jetty_length = 3e3
const jetty_width = Δx
const jetty_south_y = river_mouth_y - jetty_length
const winyah_jetty_x = (winyah_center_x - inlet_width / 2,
                        winyah_center_x + inlet_width / 2)
const santee_channel_length = 3e3
const santee_channel_south_y = river_mouth_y - santee_channel_length
const santee_channel_x = (santee_center_x - inlet_width / 2,
                          santee_center_x + inlet_width / 2)

@inline in_left_embayment(x) = abs(x - santee_center_x) <= inlet_width / 2
@inline in_right_embayment(x) = abs(x - winyah_center_x) <= inlet_width / 2
@inline in_embayment(x) = in_left_embayment(x) || in_right_embayment(x)

@inline function in_winyah_jetty(x, y)
    along_jetty = jetty_south_y <= y <= river_mouth_y
    on_west_jetty = abs(x - winyah_jetty_x[1]) <= jetty_width / 2
    on_east_jetty = abs(x - winyah_jetty_x[2]) <= jetty_width / 2
    return along_jetty && (on_west_jetty || on_east_jetty)
end

@inline function in_winyah_navigation_channel(x, y)
    between_jetties = winyah_jetty_x[1] < x < winyah_jetty_x[2]
    along_jetties = jetty_south_y <= y <= river_mouth_y
    return between_jetties && along_jetties
end

@inline function in_santee_navigation_channel(x, y)
    within_channel = santee_channel_x[1] < x < santee_channel_x[2]
    along_channel = santee_channel_south_y <= y <= river_mouth_y
    return within_channel && along_channel
end

@inline function shelf_and_channel_depth(x, y)
    if y > river_mouth_y
        in_left_embayment(x) && return santee_channel_depth
        in_right_embayment(x) && return winyah_channel_depth
        return 0.0
    else
        offshore_distance = river_mouth_y - y

        if offshore_distance <= nearshore_slope_length
            nearshore_fraction = clamp(offshore_distance / nearshore_slope_length, 0.0, 1.0)
            return nearshore_slope_depth * nearshore_fraction
        else
            offshore_domain_length = river_mouth_y - y₀
            outer_shelf_length = offshore_domain_length - nearshore_slope_length
            outer_shelf_fraction = clamp((offshore_distance - nearshore_slope_length) /
                                         outer_shelf_length, 0.0, 1.0)
            return nearshore_slope_depth +
                   (slope_depth - nearshore_slope_depth) * outer_shelf_fraction
        end
    end
end

@inline function water_depth(x, y)
    in_winyah_jetty(x, y) && return 0.0
    in_winyah_navigation_channel(x, y) && return winyah_channel_depth
    in_santee_navigation_channel(x, y) && return santee_channel_depth
    return shelf_and_channel_depth(x, y)
end

@assert water_depth(0.0, river_mouth_y) == 0.0
@assert water_depth(0.0, river_mouth_y - nearshore_slope_length) == nearshore_slope_depth
@assert water_depth(0.0, y₀) == slope_depth
@assert water_depth(winyah_center_x, river_mouth_y) == winyah_channel_depth
@assert water_depth(winyah_center_x, jetty_south_y) == winyah_channel_depth
@assert water_depth(santee_center_x, river_mouth_y) == santee_channel_depth
@assert water_depth(santee_center_x, santee_channel_south_y) == santee_channel_depth

####
#### Bathymetry map
####

x_m = collect(range(x₀ + Δx / 2, x₁ - Δx / 2; length = Nx))
y_m = collect(range(y₀ + Δy / 2, y₁ - Δy / 2; length = Ny))
x_km = x_m ./ 1e3
y_km = y_m ./ 1e3

depth = [water_depth(x, y) for x in x_m, y in y_m]

fig = Figure(size = (1900, 680), fontsize = 18)

ax_full = Axis(fig[1, 1],
               title = "Bathymetry with 3 km Winyah jetties and Santee channel",
               xlabel = "x (km)",
               ylabel = "y (km)",
               aspect = DataAspect())

hm = heatmap!(ax_full, x_km, y_km, depth;
              colormap = :deep,
              colorrange = (0, slope_depth))

ax_santee = Axis(fig[1, 2],
                 title = "Santee 5 m channel, extended 3 km",
                 xlabel = "x (km)",
                 ylabel = "y (km)",
                 aspect = DataAspect())

heatmap!(ax_santee, x_km, y_km, depth;
         colormap = :deep,
         colorrange = (0, slope_depth))

ax_zoom = Axis(fig[1, 3],
               title = "Winyah river mouth (zoom)",
               xlabel = "x (km)",
               ylabel = "y (km)",
               aspect = DataAspect())

heatmap!(ax_zoom, x_km, y_km, depth;
         colormap = :deep,
         colorrange = (0, slope_depth))

for ax in (ax_full, ax_santee, ax_zoom)
    for jetty_x in winyah_jetty_x
        lines!(ax,
               fill(jetty_x / 1e3, 2),
               [jetty_south_y, river_mouth_y] ./ 1e3;
               color = :red,
               linewidth = 3)
    end

    hlines!(ax, [river_mouth_y / 1e3];
            color = :black,
            linewidth = 2,
            linestyle = :dot)

    lines!(ax,
           [santee_channel_x[1], santee_channel_x[1],
            santee_channel_x[2], santee_channel_x[2]] ./ 1e3,
           [river_mouth_y, santee_channel_south_y,
            santee_channel_south_y, river_mouth_y] ./ 1e3;
           color = :dodgerblue,
           linewidth = 3,
           linestyle = :dash)
end

xlims!(ax_santee, (santee_center_x - 2.5e3) / 1e3,
                   (santee_center_x + 2.5e3) / 1e3)
ylims!(ax_santee, (santee_channel_south_y - 0.5e3) / 1e3,
                   (river_mouth_y + 0.3e3) / 1e3)

xlims!(ax_zoom, (winyah_center_x - 2.5e3) / 1e3,
                (winyah_center_x + 2.5e3) / 1e3)
ylims!(ax_zoom, (jetty_south_y - 0.5e3) / 1e3,
                (river_mouth_y + 0.3e3) / 1e3)

Colorbar(fig[1, 4], hm; label = "Water depth (m)")

output_file = joinpath(@__DIR__, "winyah_jetty_bathymetry.png")
save(output_file, fig)

@info "Saved river-mouth bathymetry map" output_file jetty_length jetty_width winyah_jetty_x jetty_south_y santee_channel_length santee_channel_depth
