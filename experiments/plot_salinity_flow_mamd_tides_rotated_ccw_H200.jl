using Oceananigans
using CairoMakie
using ArgParse

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--data-dir"
            help = "Directory containing instantaneous_fields.jld2"
            arg_type = String
            required = true
        "--framerate"
            help = "Animation frame rate"
            arg_type = Int
            default = 6
        "--tracer-threshold"
            help = "Minimum tracer value shown in 3D point clouds"
            arg_type = Float64
            default = 0.05
        "--tracer-stride"
            help = "Horizontal stride for 3D tracer point clouds"
            arg_type = Int
            default = 2
    end
    return parse_args(settings)
end

args = parse_commandline()
const FILE_DIR = abspath(args["data-dir"])
const DATA_FILE = joinpath(FILE_DIR, "instantaneous_fields.jld2")
const FRAMERATE = args["framerate"]
const TRACER_THRESHOLD = args["tracer-threshold"]
const TRACER_STRIDE = args["tracer-stride"]

isfile(DATA_FILE) || error("Data file does not exist: $DATA_FILE")
FRAMERATE > 0 || error("--framerate must be positive")
TRACER_THRESHOLD >= 0 || error("--tracer-threshold must be nonnegative")
TRACER_STRIDE > 0 || error("--tracer-stride must be positive")

####
#### Rotated Winyah-Santee bathymetry used by the simulation
####

const inlet_width = 1000.0
const inlet_depth = 5.0
const inlet_center_spacing = 10e3
const santee_center_y = -inlet_center_spacing / 2
const winyah_center_y =  inlet_center_spacing / 2
const original_y₀ = -7.5e3
const original_y₁ =  7.5e3
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
const santee_channel_length = 3e3
const original_santee_channel_south_y =
    original_river_mouth_y - santee_channel_length
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
    on_south_side = abs(x - original_winyah_jetty_x[1]) <= jetty_width / 2
    on_north_side = abs(x - original_winyah_jetty_x[2]) <= jetty_width / 2
    return along && (on_south_side || on_north_side)
end

@inline in_original_winyah_channel(x, y) =
    original_winyah_jetty_x[1] < x < original_winyah_jetty_x[2] &&
    original_jetty_south_y <= y <= original_river_mouth_y

@inline in_original_santee_channel(x, y) =
    original_santee_channel_x[1] < x < original_santee_channel_x[2] &&
    original_santee_channel_south_y <= y <= original_river_mouth_y

@inline function original_shelf_depth(x, y)
    if y > original_river_mouth_y
        return in_original_embayment(x) ? inlet_depth : 0.0
    end
    offshore_distance = original_river_mouth_y - y
    if offshore_distance <= nearshore_slope_length
        return nearshore_slope_depth *
               clamp(offshore_distance / nearshore_slope_length, 0.0, 1.0)
    end
    outer_length =
        original_river_mouth_y - original_y₀ - nearshore_slope_length
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
@inline bathymetry(x, y) = -water_depth(x, y)
@inline is_wet(x, y, z) = water_depth(x, y) > 0 && z >= bathymetry(x, y)
nearest_index(nodes, value) = argmin(abs.(nodes .- value))
time_string(t) = "t = $(round(t / 3600; digits = 1)) hour"

function mask_surface(snapshot, x, y, z; cutoff = nothing)
    values = Array(snapshot)
    for j in eachindex(y), i in eachindex(x)
        if !is_wet(x[i], y[j], z) ||
           (!isnothing(cutoff) && values[i, j] <= cutoff)
            values[i, j] = NaN
        end
    end
    return values
end

function mask_xz(snapshot, x, y, z)
    values = Array(snapshot)
    for k in eachindex(z), i in eachindex(x)
        is_wet(x[i], y, z[k]) || (values[i, k] = NaN)
    end
    return values
end

function mask_yz(snapshot, x, y, z)
    values = Array(snapshot)
    for k in eachindex(z), j in eachindex(y)
        is_wet(x, y[j], z[k]) || (values[j, k] = NaN)
    end
    return values
end

####
#### Surface salinity and density
####

function save_surface_salinity_animation()
    data = FieldTimeSeries(DATA_FILE, "S")
    Nt = length(data.times)
    x = collect(xnodes(data.grid, Center()))
    y = collect(ynodes(data.grid, Center()))
    z = collect(znodes(data.grid, Center()))
    x_km, y_km = x ./ 1e3, y ./ 1e3
    k = lastindex(z)
    n = Observable(1)
    surface_salinity = lift(n) do nn
        mask_surface(interior(data[nn], :, :, k), x, y, z[k])
    end

    fig = Figure(size = (1000, 1050), fontsize = 18)
    ax = Axis(fig[1, 1], title = "Surface salinity",
              xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
              aspect = DataAspect())
    hm = heatmap!(ax, x_km, y_km, surface_salinity;
                  colormap = :haline, colorrange = (10, 34),
                  nan_color = :lightgray)
    Colorbar(fig[1, 2], hm; label = "Salinity")
    Label(fig[0, :], lift(nn -> time_string(data.times[nn]), n);
          fontsize = 22)
    output = joinpath(FILE_DIR, "surface_salinity_xy.mp4")
    CairoMakie.record(fig, output, 1:Nt; framerate = FRAMERATE) do nn
        n[] = nn
    end
    @info "Saved surface salinity animation" output
end

function save_surface_density_animation()
    data = FieldTimeSeries(DATA_FILE, "S")
    Nt = length(data.times)
    x = collect(xnodes(data.grid, Center()))
    y = collect(ynodes(data.grid, Center()))
    z = collect(znodes(data.grid, Center()))
    x_km, y_km = x ./ 1e3, y ./ 1e3
    k = lastindex(z)
    ρ₀, β, S₀ = 1025.0, 7.8e-4, 34.0
    n = Observable(1)
    density = lift(n) do nn
        S = mask_surface(interior(data[nn], :, :, k), x, y, z[k])
        ρ₀ * β .* (S .- S₀)
    end

    fig = Figure(size = (1000, 1050), fontsize = 18)
    ax = Axis(fig[1, 1], title = "Surface density anomaly",
              xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
              aspect = DataAspect())
    hm = heatmap!(ax, x_km, y_km, density;
                  colormap = :balance, colorrange = (-20, 5),
                  nan_color = :lightgray)
    Colorbar(fig[1, 2], hm; label = "ρ - ρ₀ (kg m⁻³)")
    Label(fig[0, :], lift(nn -> time_string(data.times[nn]), n);
          fontsize = 22)
    output = joinpath(FILE_DIR, "surface_density_xy.mp4")
    CairoMakie.record(fig, output, 1:Nt; framerate = FRAMERATE) do nn
        n[] = nn
    end
    @info "Saved surface density animation" output
end

####
#### Surface tracer panels and salinity transects
####

function save_surface_tracer_panels_animation()
    santee = FieldTimeSeries(DATA_FILE, "c_santee")
    winyah = FieldTimeSeries(DATA_FILE, "c_winyah")
    Nt = min(length(santee.times), length(winyah.times))
    x = collect(xnodes(santee.grid, Center()))
    y = collect(ynodes(santee.grid, Center()))
    z = collect(znodes(santee.grid, Center()))
    x_km, y_km = x ./ 1e3, y ./ 1e3
    k = lastindex(z)
    n = Observable(1)
    cs = lift(n) do nn
        mask_surface(interior(santee[nn], :, :, k), x, y, z[k]; cutoff = 1e-6)
    end
    cw = lift(n) do nn
        mask_surface(interior(winyah[nn], :, :, k), x, y, z[k]; cutoff = 1e-6)
    end

    fig = Figure(size = (1900, 1050), fontsize = 18)
    ax_both = Axis(fig[1, 1], title = "Surface Santee + Winyah tracers",
                   xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                   aspect = DataAspect())
    ax_w = Axis(fig[1, 2], title = "Surface Winyah tracer",
                xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                aspect = DataAspect())
    ax_s = Axis(fig[1, 3], title = "Surface Santee tracer",
                xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                aspect = DataAspect())
    heatmap!(ax_both, x_km, y_km, cs; colormap = :viridis,
             colorrange = (0, 1), nan_color = :transparent, alpha = 0.72)
    heatmap!(ax_both, x_km, y_km, cw; colormap = :magma,
             colorrange = (0, 1), nan_color = :transparent, alpha = 0.62)
    hw = heatmap!(ax_w, x_km, y_km, cw; colormap = :magma,
                  colorrange = (0, 1), nan_color = :transparent)
    hs = heatmap!(ax_s, x_km, y_km, cs; colormap = :viridis,
                  colorrange = (0, 1), nan_color = :transparent)
    Colorbar(fig[1, 4], hw; label = "Winyah tracer")
    Colorbar(fig[1, 5], hs; label = "Santee tracer")
    Label(fig[0, :], lift(nn -> time_string(santee.times[nn]), n);
          fontsize = 22)
    output = joinpath(FILE_DIR, "surface_tracers_panels_xy.mp4")
    CairoMakie.record(fig, output, 1:Nt; framerate = FRAMERATE) do nn
        n[] = nn
    end
    @info "Saved surface tracer panels animation" output
end

function save_salinity_transects_animation()
    data = FieldTimeSeries(DATA_FILE, "S")
    Nt = length(data.times)
    x = collect(xnodes(data.grid, Center()))
    y = collect(ynodes(data.grid, Center()))
    z = collect(znodes(data.grid, Center()))
    x_km, y_km = x ./ 1e3, y ./ 1e3
    j_s = nearest_index(y, santee_center_y)
    j_w = nearest_index(y, winyah_center_y)
    i_coast = nearest_index(x, rotated_coast_x)
    i_offshore = nearest_index(x, -5e3)
    n = Observable(1)
    S_w = lift(n) do nn
        mask_xz(interior(data[nn], :, j_w, :), x, y[j_w], z)
    end
    S_s = lift(n) do nn
        mask_xz(interior(data[nn], :, j_s, :), x, y[j_s], z)
    end
    S_coast = lift(n) do nn
        mask_yz(interior(data[nn], i_coast, :, :), x[i_coast], y, z)
    end
    S_offshore = lift(n) do nn
        mask_yz(interior(data[nn], i_offshore, :, :), x[i_offshore], y, z)
    end

    fig = Figure(size = (1600, 1050), fontsize = 18)
    ax_w = Axis(fig[1, 1], title = "Winyah centerline x-z",
                xlabel = "Eastward x (km)", ylabel = "z (m)")
    ax_s = Axis(fig[2, 1], title = "Santee centerline x-z",
                xlabel = "Eastward x (km)", ylabel = "z (m)")
    ax_c = Axis(fig[1, 2], title = "Shoreline y-z",
                xlabel = "Northward y (km)", ylabel = "z (m)")
    ax_o = Axis(fig[2, 2], title = "Nearshore y-z at x = -5 km",
                xlabel = "Northward y (km)", ylabel = "z (m)")
    hm = heatmap!(ax_w, x_km, z, S_w; colormap = :haline, colorrange = (10, 34),
                  nan_color = :lightgray)
    heatmap!(ax_s, x_km, z, S_s; colormap = :haline, colorrange = (10, 34),
             nan_color = :lightgray)
    heatmap!(ax_c, y_km, z, S_coast; colormap = :haline, colorrange = (10, 34),
             nan_color = :lightgray)
    heatmap!(ax_o, y_km, z, S_offshore; colormap = :haline,
             colorrange = (10, 34), nan_color = :lightgray)
    lines!(ax_w, x_km, [bathymetry(xi, y[j_w]) for xi in x];
           color = :black, linewidth = 3)
    lines!(ax_s, x_km, [bathymetry(xi, y[j_s]) for xi in x];
           color = :black, linewidth = 3)
    Colorbar(fig[:, 3], hm; label = "Salinity")
    Label(fig[0, :], lift(nn -> time_string(data.times[nn]), n);
          fontsize = 22)
    output = joinpath(FILE_DIR, "salinity_transects.mp4")
    CairoMakie.record(fig, output, 1:Nt; framerate = FRAMERATE) do nn
        n[] = nn
    end
    @info "Saved salinity transects animation" output
end

####
#### Three-dimensional tracer point clouds
####

function tracer_points(snapshot, x, y, z)
    values = Array(interior(snapshot, :, :, :))
    xs = Float64[]; ys = Float64[]; zs = Float64[]; cs = Float64[]
    for k in eachindex(z),
        j in firstindex(y):TRACER_STRIDE:lastindex(y),
        i in firstindex(x):TRACER_STRIDE:lastindex(x)
        c = values[i, j, k]
        if c >= TRACER_THRESHOLD &&
           is_wet(1e3 * x[i], 1e3 * y[j], z[k])
            push!(xs, x[i]); push!(ys, y[j]); push!(zs, z[k]); push!(cs, c)
        end
    end
    return xs, ys, zs, cs
end

function tracer_axis(fig, x, y, bottom, title)
    ax = Axis3(fig[1, 1], title = title,
               xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
               zlabel = "z (m)", azimuth = 1.2pi, elevation = 0.18pi,
               aspect = (1, 1, 0.45))
    surface!(ax, x, y, bottom; colormap = :deep,
             colorrange = (-slope_depth, 0), alpha = 0.55,
             transparency = true)
    xlims!(ax, extrema(x)...)
    ylims!(ax, extrema(y)...)
    zlims!(ax, -slope_depth, 2)
    return ax
end

function save_combined_3d_tracer_animation()
    santee = FieldTimeSeries(DATA_FILE, "c_santee")
    winyah = FieldTimeSeries(DATA_FILE, "c_winyah")
    Nt = min(length(santee.times), length(winyah.times))
    x_m = collect(xnodes(santee.grid, Center()))
    y_m = collect(ynodes(santee.grid, Center()))
    z = collect(znodes(santee.grid, Center()))
    x, y = x_m ./ 1e3, y_m ./ 1e3
    bottom = [bathymetry(xi, yi) for xi in x_m, yi in y_m]
    xs0, ys0, zs0, cs0 = tracer_points(santee[1], x, y, z)
    xw0, yw0, zw0, cw0 = tracer_points(winyah[1], x, y, z)
    xs = Observable(xs0); ys = Observable(ys0)
    zs = Observable(zs0); cs = Observable(cs0)
    xw = Observable(xw0); yw = Observable(yw0)
    zw = Observable(zw0); cw = Observable(cw0)
    fig = Figure(size = (1400, 900), fontsize = 18)
    ax = tracer_axis(fig, x, y, bottom, "3D Santee + Winyah tracer plumes")
    ps = scatter!(ax, xs, ys, zs; color = cs, colormap = :viridis,
                  colorrange = (0, 1), markersize = 8, alpha = 0.75)
    pw = scatter!(ax, xw, yw, zw; color = cw, colormap = :magma,
                  colorrange = (0, 1), markersize = 8, alpha = 0.65)
    Colorbar(fig[1, 2], ps; label = "Santee tracer")
    Colorbar(fig[1, 3], pw; label = "Winyah tracer")
    label = Observable(time_string(santee.times[1]))
    Label(fig[0, :], label; fontsize = 22)
    output = joinpath(FILE_DIR, "tracers_3d.mp4")
    CairoMakie.record(fig, output, 1:Nt; framerate = FRAMERATE) do nn
        xs[], ys[], zs[], cs[] = tracer_points(santee[nn], x, y, z)
        xw[], yw[], zw[], cw[] = tracer_points(winyah[nn], x, y, z)
        label[] = time_string(santee.times[nn])
    end
    @info "Saved combined 3D tracer animation" output
end

function save_single_3d_tracer_animation(field, river, colormap, filename)
    data = FieldTimeSeries(DATA_FILE, field)
    Nt = length(data.times)
    x_m = collect(xnodes(data.grid, Center()))
    y_m = collect(ynodes(data.grid, Center()))
    z = collect(znodes(data.grid, Center()))
    x, y = x_m ./ 1e3, y_m ./ 1e3
    bottom = [bathymetry(xi, yi) for xi in x_m, yi in y_m]
    x0, y0, z0, c0 = tracer_points(data[1], x, y, z)
    xs = Observable(x0); ys = Observable(y0)
    zs = Observable(z0); cs = Observable(c0)
    fig = Figure(size = (1200, 900), fontsize = 18)
    ax = tracer_axis(fig, x, y, bottom, "3D $river tracer plume")
    plume = scatter!(ax, xs, ys, zs; color = cs, colormap,
                     colorrange = (0, 1), markersize = 8, alpha = 0.75)
    Colorbar(fig[1, 2], plume; label = "$river tracer")
    label = Observable(time_string(data.times[1]))
    Label(fig[0, :], label; fontsize = 22)
    output = joinpath(FILE_DIR, filename)
    CairoMakie.record(fig, output, 1:Nt; framerate = FRAMERATE) do nn
        xs[], ys[], zs[], cs[] = tracer_points(data[nn], x, y, z)
        label[] = time_string(data.times[nn])
    end
    @info "Saved single-river 3D tracer animation" river output
end

@info "Plotting salinity-tide animations" FILE_DIR DATA_FILE
save_surface_salinity_animation()
save_surface_density_animation()
save_surface_tracer_panels_animation()
save_salinity_transects_animation()
save_combined_3d_tracer_animation()
save_single_3d_tracer_animation("c_winyah", "Winyah", :magma,
                                "Winyah_tracers_3d.mp4")
save_single_3d_tracer_animation("c_santee", "Santee", :viridis,
                                "Santee_tracers_3d.mp4")
@info "All salinity-tide animations complete" FILE_DIR
