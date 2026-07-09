using Oceananigans
using Printf
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using Oceananigans.Utils: launch!
using Oceananigans.Units
using KernelAbstractions: @kernel, @index
using Statistics
using CUDA
using ArgParse
using CairoMakie

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--arch"
            help = "Architecture to use: GPU or CPU"
            arg_type = String
            default = "GPU"
        "--setup-only"
            help = "Build the model and save initial condition plots without running the simulation"
            action = :store_true
        "--stop-time"
            help = "Simulation stop time in seconds"
            arg_type = Float64
            default = 21600minutes
    end

    return parse_args(s)
end

args = parse_commandline()
const ARCH = uppercase(args["arch"])
ARCH in ("GPU", "CPU") || error("Invalid --arch $(ARCH); must be GPU or CPU")

arch = ARCH == "GPU" ? GPU() : CPU()

#####
##### Domain and buoyant inflow parameters
#####

const Lx = 40e3
const Ly = 20e3
const Lz = 20.0

const Δx = 100.0
const Δy = 100.0
const Nz = 20
const Δz = Lz / Nz

const Nx = Int(Lx / Δx)
const Ny = Int(Ly / Δy)


const x₀ = -Lx / 2
const x₁ =  Lx / 2
const y₀ = -Ly / 2
const y₁ =  Ly / 2
const z₀ = -Lz
const z₁ = 0.0

const inlet_width = 1000.0
const inlet_center_spacing = 10e3
const inlet_centers = (-inlet_center_spacing / 2, inlet_center_spacing / 2)
const embayment_depth = 5.0
const inlet_depth = embayment_depth
const santee_inlet_speed = 0.002
const winyah_inlet_speed = 0.14
const santee_N² = 1e-2
const winyah_N² = 5e-3
const shelf_N² = 1e-5
const N² = shelf_N²
const plume_surface_buoyancy = max(santee_N², winyah_N²) * inlet_depth
const f₀ = 1e-4
const Cd = 2e-3
const embayment_length_y = 200.0
const river_mouth_y = y₁ - embayment_length_y
const slope_depth = 15.0
const initial_buoyancy_noise = 1e-6

const inlet_transport = -inlet_width * inlet_depth * (santee_inlet_speed + winyah_inlet_speed)
const shelf_open_boundary_length_y = river_mouth_y - y₀
const west_outlet_area = 0.5 * (embayment_depth + slope_depth) * shelf_open_boundary_length_y
const outlet_speed = abs(inlet_transport) / west_outlet_area
const sponge_width = 2e3
const sponge_timescale = 30minutes
const sponge_rate = 1 / sponge_timescale

const ν₄h = 1e5
const κ₄h = 1e5
const ν₄z = 1e-3
const κ₄z = 1e-3

filename = string("MAMD_WB700_Santee10_Nx_", Nx, "_Ny_", Ny, "_Nz_", Nz, "_spacing_", Int(inlet_center_spacing), "m_30min")
FILE_DIR = joinpath("Data", filename)
mkpath(FILE_DIR)

#####
##### Geometry and boundary conditions
#####

@inline function in_left_embayment(x)
    half_width = inlet_width / 2
    return abs(x - inlet_centers[1]) <= half_width
end

@inline function in_right_embayment(x)
    half_width = inlet_width / 2
    return abs(x - inlet_centers[2]) <= half_width
end

@inline in_embayment(x) = in_left_embayment(x) || in_right_embayment(x)

@inline function water_depth(x, y)
    if y > river_mouth_y
        return in_embayment(x) ? embayment_depth : 0.0
    else
        slope_fraction = clamp((river_mouth_y - y) / (river_mouth_y - y₀), 0.0, 1.0)
        return embayment_depth + (slope_depth - embayment_depth) * slope_fraction
    end
end

@inline bathymetry(x, y) = -water_depth(x, y)
@inline is_wet(x, y, z) = water_depth(x, y) > 0 && z >= bathymetry(x, y)

@inline function in_inlet(x, z)
    return in_embayment(x) && z >= -inlet_depth
end

@inline function inlet_speed_profile(x, z)
    z >= -inlet_depth || return 0.0
    in_left_embayment(x) && return santee_inlet_speed
    in_right_embayment(x) && return winyah_inlet_speed
    return 0.0
end

@inline function v_inflow_profile(x, z, t)
    return -inlet_speed_profile(x, z)
end

@inline function inlet_buoyancy_profile(x, z)
    z >= -inlet_depth || return shelf_N² * z
    in_left_embayment(x) && return plume_surface_buoyancy + santee_N² * z
    in_right_embayment(x) && return plume_surface_buoyancy + winyah_N² * z
    return shelf_N² * z
end

@inline function b_inflow_profile(x, z, t)
    return inlet_buoyancy_profile(x, z)
end

@inline function c_santee_inflow_profile(x, z, t)
    return in_left_embayment(x) && z >= -inlet_depth ? 1.0 : 0.0
end

@inline function c_winyah_inflow_profile(x, z, t)
    return in_right_embayment(x) && z >= -inlet_depth ? 1.0 : 0.0
end

@inline u_west_outflow(y, z, t) = -outlet_speed

@inline ambient_buoyancy(z) = shelf_N² * z

@inline function smoothstep(η)
    η = clamp(η, 0.0, 1.0)
    return η^2 * (3 - 2η)
end

@inline function river_to_mouth_buoyancy_weight(x, y, z)
    in_channel = in_embayment(x) && river_mouth_y <= y <= y₁
    in_channel || return 0.0
    is_wet(x, y, z) || return 0.0

    η = (y - river_mouth_y) / embayment_length_y
    return smoothstep(η)
end

@inline function b_initial(x, y, z)
    background = ambient_buoyancy(z)
    river_weight = river_to_mouth_buoyancy_weight(x, y, z)
    target_buoyancy = in_left_embayment(x) ? plume_surface_buoyancy + santee_N² * z : plume_surface_buoyancy + winyah_N² * z
    return background + river_weight * (target_buoyancy - background) + initial_buoyancy_noise * rand()
end

grid = RectilinearGrid(arch, Float64,
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (x₀, x₁),
                       y = (y₀, y₁),
                       z = (z₀, z₁),
                       topology = (Bounded, Bounded, Bounded))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

v_north_bc = NormalFlowBoundaryCondition(v_inflow_profile;
                                         scheme = PerturbationAdvection(target_transport = inlet_transport))
u_west_bc = NormalFlowBoundaryCondition(u_west_outflow;
                                        scheme = PerturbationAdvection())

b_north_bc = ValueBoundaryCondition(b_inflow_profile)
c_santee_north_bc = ValueBoundaryCondition(c_santee_inflow_profile)
c_winyah_north_bc = ValueBoundaryCondition(c_winyah_inflow_profile)

quadratic_drag = BulkDrag(coefficient = Cd)
no_slip_bc = ValueBoundaryCondition(0)

u_bcs = FieldBoundaryConditions(immersed = quadratic_drag,
                                bottom = quadratic_drag,
                                north = no_slip_bc,
                                west = u_west_bc)
v_bcs = FieldBoundaryConditions(immersed = quadratic_drag,
                                bottom = quadratic_drag,
                                north = v_north_bc)
w_bcs = FieldBoundaryConditions(north = no_slip_bc)
b_bcs = FieldBoundaryConditions(north = b_north_bc)
c_santee_bcs = FieldBoundaryConditions(north = c_santee_north_bc)
c_winyah_bcs = FieldBoundaryConditions(north = c_winyah_north_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs, b = b_bcs,
                       c_santee = c_santee_bcs, c_winyah = c_winyah_bcs)

 function sponge_mask(x, y)
    west = smoothstep((x₀ + sponge_width - x) / sponge_width)
    east = smoothstep((x - (x₁ - sponge_width)) / sponge_width)
    south = smoothstep((y₀ + sponge_width - y) / sponge_width)
    return max(west, east, south)
end

 sponge_relaxation(x, y, field, target) = -sponge_rate * sponge_mask(x, y) * (field - target)

 u_sponge(x, y, z, t, u) = sponge_relaxation(x, y, u, 0.0)
 v_sponge(x, y, z, t, v) = sponge_relaxation(x, y, v, 0.0)
 b_sponge(x, y, z, t, b) = sponge_relaxation(x, y, b, ambient_buoyancy(z))
 c_santee_sponge(x, y, z, t, c_santee) = sponge_relaxation(x, y, c_santee, 0.0)
 c_winyah_sponge(x, y, z, t, c_winyah) = sponge_relaxation(x, y, c_winyah, 0.0)

forcing = (u = Forcing(u_sponge, field_dependencies = :u),
           v = Forcing(v_sponge, field_dependencies = :v),
           b = Forcing(b_sponge, field_dependencies = :b),
           c_santee = Forcing(c_santee_sponge, field_dependencies = :c_santee),
           c_winyah = Forcing(c_winyah_sponge, field_dependencies = :c_winyah))

closure = Oceananigans.TurbulenceClosures.ModifiedAnisotropicMinimumDissipation()
pressure_solver = ConjugateGradientPoissonSolver(grid)
coriolis = FPlane(f = f₀)

model = NonhydrostaticModel(grid;
                            pressure_solver,
                            advection = WENO(order = 5),
                            tracers = (:b, :c_santee, :c_winyah),
                            coriolis,
                            closure,
                            forcing,
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions)

set!(model, b = b_initial)

#####

function nearest_index(nodes, value)
    return argmin(abs.(nodes .- value))
end

#####
##### Initial condition plots
#####

function save_initial_plots!(model)
    b = model.tracers.b

    x_m = collect(xnodes(model.grid, Center()))
    y_m = collect(ynodes(model.grid, Center()))
    z_m = collect(znodes(model.grid, Center()))

    xC = x_m ./ 1e3
    yC = y_m ./ 1e3

    depth_xy = [water_depth(x, y) for x in x_m, y in y_m]

    fig_depth = Figure(size = (900, 760), fontsize = 20)
    ax_depth = Axis(fig_depth[1, 1],
                    title = "Water depth",
                    xlabel = "x (km)",
                    ylabel = "y (km)",
                    aspect = DataAspect())
    hm_depth = heatmap!(ax_depth, xC, yC, depth_xy; colormap = :deep, colorrange = (0, slope_depth))
    for xc in inlet_centers
        vlines!(ax_depth, [xc / 1e3]; color = :white, linewidth = 2, linestyle = :dash)
    end
    hlines!(ax_depth, [river_mouth_y / 1e3]; color = :white, linewidth = 2, linestyle = :dot)
    Colorbar(fig_depth[1, 2], hm_depth; label = "Depth (m)")
    save(joinpath(FILE_DIR, "bathymetry_xy.png"), fig_depth)

    bxy = Array(interior(b, :, :, Nz))
    j_mouth = nearest_index(y_m, river_mouth_y)
    bxz = Array(interior(b, :, j_mouth, :))
    buoyancy_colorrange = plume_surface_buoyancy
    colorrange = (-buoyancy_colorrange, buoyancy_colorrange)
    i_left_channel = nearest_index(x_m, inlet_centers[1])
    i_right_channel = nearest_index(x_m, inlet_centers[2])
    byz_left = Array(interior(b, i_left_channel, :, :))
    byz_right = Array(interior(b, i_right_channel, :, :))

    for i in eachindex(x_m), j in eachindex(y_m)
        bxy[i, j] = is_wet(x_m[i], y_m[j], z_m[Nz]) ? bxy[i, j] : NaN
    end

    for i in eachindex(x_m), k in eachindex(z_m)
        bxz[i, k] = is_wet(x_m[i], y_m[j_mouth], z_m[k]) ? bxz[i, k] : NaN
    end

    for j in eachindex(y_m), k in eachindex(z_m)
        byz_left[j, k] = is_wet(x_m[i_left_channel], y_m[j], z_m[k]) ? byz_left[j, k] : NaN
        byz_right[j, k] = is_wet(x_m[i_right_channel], y_m[j], z_m[k]) ? byz_right[j, k] : NaN
    end

    fig_xy = Figure(size = (900, 760), fontsize = 20)
    ax_xy = Axis(fig_xy[1, 1],
                 title = "Initial surface buoyancy",
                 xlabel = "x (km)",
                 ylabel = "y (km)",
                 aspect = DataAspect())
    hm_xy = heatmap!(ax_xy, xC, yC, bxy; colormap = :balance, colorrange)
    Colorbar(fig_xy[1, 2], hm_xy; label = "Buoyancy")
    save(joinpath(FILE_DIR, "initial_surface_buoyancy_xy.png"), fig_xy)

    fig_xz = Figure(size = (980, 560), fontsize = 20)
    ax_xz = Axis(fig_xz[1, 1],
                 title = "Initial buoyancy x-z section at northern inlet",
                 xlabel = "x (km)",
                 ylabel = "z (m)")
    hm_xz = heatmap!(ax_xz, xC, z_m, bxz; colormap = :balance, colorrange)
    Colorbar(fig_xz[1, 2], hm_xz; label = "Buoyancy")
    save(joinpath(FILE_DIR, "initial_buoyancy_xz.png"), fig_xz)

    fig_yz = Figure(size = (1400, 560), fontsize = 20)
    ax_yz_left = Axis(fig_yz[1, 1],
                      title = string("Initial buoyancy y-z section, Santee channel x = ", round(x_m[i_left_channel] / 1e3; digits = 2), " km"),
                      xlabel = "y (km)",
                      ylabel = "z (m)")
    ax_yz_right = Axis(fig_yz[1, 2],
                       title = string("Initial buoyancy y-z section, Winyah channel x = ", round(x_m[i_right_channel] / 1e3; digits = 2), " km"),
                       xlabel = "y (km)",
                       ylabel = "z (m)")
    hm_yz = heatmap!(ax_yz_left, yC, z_m, byz_left; colormap = :balance, colorrange)
    heatmap!(ax_yz_right, yC, z_m, byz_right; colormap = :balance, colorrange)
    lines!(ax_yz_left, yC, [bathymetry(x_m[i_left_channel], y) for y in y_m]; color = :black, linewidth = 3)
    lines!(ax_yz_right, yC, [bathymetry(x_m[i_right_channel], y) for y in y_m]; color = :black, linewidth = 3)
    vlines!(ax_yz_left, [river_mouth_y / 1e3]; color = :black, linewidth = 2, linestyle = :dot)
    vlines!(ax_yz_right, [river_mouth_y / 1e3]; color = :black, linewidth = 2, linestyle = :dot)
    Colorbar(fig_yz[1, 3], hm_yz; label = "Buoyancy")
    save(joinpath(FILE_DIR, "initial_buoyancy_yz_channel.png"), fig_yz)

    fig_yz_right = Figure(size = (980, 560), fontsize = 20)
    ax_yz_right_only = Axis(fig_yz_right[1, 1],
                            title = string("Initial buoyancy y-z section, Winyah channel x = ", round(x_m[i_right_channel] / 1e3; digits = 2), " km"),
                            xlabel = "y (km)",
                            ylabel = "z (m)")
    hm_yz_right = heatmap!(ax_yz_right_only, yC, z_m, byz_right; colormap = :balance, colorrange)
    lines!(ax_yz_right_only, yC, [bathymetry(x_m[i_right_channel], y) for y in y_m]; color = :black, linewidth = 3)
    vlines!(ax_yz_right_only, [river_mouth_y / 1e3]; color = :black, linewidth = 2, linestyle = :dot)
    Colorbar(fig_yz_right[1, 2], hm_yz_right; label = "Buoyancy")
    save(joinpath(FILE_DIR, "initial_buoyancy_yz_right_channel.png"), fig_yz_right)

    j_y0 = nearest_index(y_m, 0.0)
    j_ymin = 1
    bxz_y0 = Array(interior(b, :, j_y0, :))
    bxz_ymin = Array(interior(b, :, j_ymin, :))

    for i in eachindex(x_m), k in eachindex(z_m)
        bxz_y0[i, k] = is_wet(x_m[i], y_m[j_y0], z_m[k]) ? bxz_y0[i, k] : NaN
        bxz_ymin[i, k] = is_wet(x_m[i], y_m[j_ymin], z_m[k]) ? bxz_ymin[i, k] : NaN
    end

    fig_sections = Figure(size = (1500, 1100), fontsize = 18)
    ax_sections_xy = Axis(fig_sections[1, 1],
                          title = "Initial surface buoyancy",
                          xlabel = "x (km)",
                          ylabel = "y (km)",
                          aspect = DataAspect())
    ax_sections_mouth = Axis(fig_sections[1, 2],
                             title = string("Initial x-z buoyancy at river mouth, y = ", round(y_m[j_mouth] / 1e3; digits = 2), " km"),
                             xlabel = "x (km)",
                             ylabel = "z (m)")
    ax_sections_y0 = Axis(fig_sections[2, 1],
                          title = string("Initial x-z buoyancy at y = ", round(y_m[j_y0] / 1e3; digits = 2), " km"),
                          xlabel = "x (km)",
                          ylabel = "z (m)")
    ax_sections_ymin = Axis(fig_sections[2, 2],
                            title = string("Initial x-z buoyancy at y = ", round(y_m[j_ymin] / 1e3; digits = 2), " km"),
                            xlabel = "x (km)",
                            ylabel = "z (m)")

    hm_sections = heatmap!(ax_sections_xy, xC, yC, bxy; colormap = :balance, colorrange)
    heatmap!(ax_sections_mouth, xC, z_m, bxz; colormap = :balance, colorrange)
    heatmap!(ax_sections_y0, xC, z_m, bxz_y0; colormap = :balance, colorrange)
    heatmap!(ax_sections_ymin, xC, z_m, bxz_ymin; colormap = :balance, colorrange)
    Colorbar(fig_sections[:, 3], hm_sections; label = "Buoyancy")
    save(joinpath(FILE_DIR, "initial_buoyancy_sections.png"), fig_sections)

    return nothing
end

save_initial_plots!(model)

if args["setup-only"]
    @info string("Setup complete. Initial condition plots saved in ", FILE_DIR, ".")
    exit()
end

#####
##### Simulation
#####

stop_time = args["stop-time"]
Δt = 1

simulation = Simulation(model; Δt, stop_time)
time_wizard = TimeStepWizard(cfl = 0.3, max_change = 1.05, max_Δt = 20.0)
simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

u, v, w = model.velocities
b = model.tracers.b
c_santee = model.tracers.c_santee
c_winyah = model.tracers.c_winyah

d = CenterField(grid)

@kernel function _divergence!(target_field, u, v, w, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds target_field[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function compute_flow_divergence!(target_field, model)
    grid = model.grid
    u, v, w = model.velocities
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _divergence!, target_field, u, v, w, grid)
    return nothing
end

wall_clock = Ref(time_ns())
progress_start_time = Ref(time_ns())

function progress(sim)
    now = time_ns()
    elapsed = 1e-9 * (now - wall_clock[])
    total_elapsed = 1e-9 * (now - progress_start_time[])
    percent_complete = 100 * sim.model.clock.time / sim.stop_time

    compute_flow_divergence!(d, sim.model)

    msg = string("i: ", iteration(sim),
                 ", t: ", prettytime(sim), " / ", prettytime(sim.stop_time),
                 " (", round(percent_complete; digits = 1), "%)",
                 ", wall Δt: ", prettytime(elapsed),
                 ", wall total: ", prettytime(total_elapsed),
                 ", model Δt: ", prettytime(sim.Δt),
                 ", max |u|: ", maximum(abs, sim.model.velocities.u),
                 ", max |v|: ", maximum(abs, sim.model.velocities.v),
                 ", max |w|: ", maximum(abs, sim.model.velocities.w),
                 ", min b: ", minimum(sim.model.tracers.b),
                 ", max b: ", maximum(sim.model.tracers.b),
                 ", max c_santee: ", maximum(abs, sim.model.tracers.c_santee),
                 ", max c_winyah: ", maximum(abs, sim.model.tracers.c_winyah),
                 ", max div: ", maximum(abs, d))

    wall_clock[] = now
    println(msg)

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(5))

simulation.output_writers[:jld2] = JLD2Writer(model, (; u, v, w, b, c_santee, c_winyah);
                                              filename = joinpath(FILE_DIR, "instantaneous_fields.jld2"),
                                              schedule = TimeInterval(1hour),
                                              with_halos = true,
                                              overwrite_existing = true)

function nearest_index(nodes, value)
    return argmin(abs.(nodes .- value))
end

function save_surface_buoyancy_animation()
    filepath = joinpath(FILE_DIR, "instantaneous_fields.jld2")
    b_data = FieldTimeSeries(filepath, "b")
    Nt = length(b_data.times)

    Nt == 0 && return nothing

    x_m = collect(xnodes(b_data.grid, Center()))
    y_m = collect(ynodes(b_data.grid, Center()))
    z_m = collect(znodes(b_data.grid, Center()))

    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    zC = z_m
    times = b_data.times

    j_river_mouth = nearest_index(y_m, river_mouth_y)
    j_y0 = nearest_index(y_m, 0.0)
    j_ymin = 1
    buoyancy_colorrange = plume_surface_buoyancy
    colorrange = (-buoyancy_colorrange, buoyancy_colorrange)

    fig = Figure(size = (1500, 1100), fontsize = 18)

    ax_xy = Axis(fig[1, 1],
                 title = "Surface buoyancy",
                 xlabel = "x (km)",
                 ylabel = "y (km)",
                 aspect = DataAspect())

    ax_river = Axis(fig[1, 2],
                    title = string("x-z buoyancy at river mouth, y = ", round(y_m[j_river_mouth] / 1e3; digits = 2), " km"),
                    xlabel = "x (km)",
                    ylabel = "z (m)")

    ax_y0 = Axis(fig[2, 1],
                 title = string("x-z buoyancy at y = ", round(y_m[j_y0] / 1e3; digits = 2), " km"),
                 xlabel = "x (km)",
                 ylabel = "z (m)")

    ax_ymin = Axis(fig[2, 2],
                   title = string("x-z buoyancy at y = ", round(y_m[j_ymin] / 1e3; digits = 2), " km"),
                   xlabel = "x (km)",
                   ylabel = "z (m)")

    n = Observable(1)

    b_surface = lift(n) do nn
        Array(interior(b_data[nn], :, :, Nz))
    end

    b_xz_river = lift(n) do nn
        Array(interior(b_data[nn], :, j_river_mouth, :))
    end

    b_xz_y0 = lift(n) do nn
        Array(interior(b_data[nn], :, j_y0, :))
    end

    b_xz_ymin = lift(n) do nn
        Array(interior(b_data[nn], :, j_ymin, :))
    end

    hm_xy = heatmap!(ax_xy, xC, yC, b_surface; colormap = :balance, colorrange)
    heatmap!(ax_river, xC, zC, b_xz_river; colormap = :balance, colorrange)
    heatmap!(ax_y0, xC, zC, b_xz_y0; colormap = :balance, colorrange)
    heatmap!(ax_ymin, xC, zC, b_xz_ymin; colormap = :balance, colorrange)

    Colorbar(fig[:, 3], hm_xy; label = "Buoyancy")

    time_label = lift(n) do nn
        string("t = ", round(times[nn] / 60; digits = 1), " min")
    end
    Label(fig[0, :], time_label, fontsize = 22)

    CairoMakie.record(fig, joinpath(FILE_DIR, "surface_buoyancy_xy.mp4"), 1:Nt; framerate = 6) do nn
        n[] = nn
    end

    return nothing
end

function save_surface_tracer_animation()
    filepath = joinpath(FILE_DIR, "instantaneous_fields.jld2")
    c_santee_data = FieldTimeSeries(filepath, "c_santee")
    c_winyah_data = FieldTimeSeries(filepath, "c_winyah")
    Nt = min(length(c_santee_data.times), length(c_winyah_data.times))

    Nt == 0 && return nothing

    x_m = collect(xnodes(c_santee_data.grid, Center()))
    y_m = collect(ynodes(c_santee_data.grid, Center()))
    z_m = collect(znodes(c_santee_data.grid, Center()))

    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    zC = z_m
    times = c_santee_data.times

    j_river_mouth = nearest_index(y_m, river_mouth_y)
    j_y0 = nearest_index(y_m, 0.0)
    j_ymin = 1
    colorrange = (0.0, 1.0)

    fig = Figure(size = (1650, 1100), fontsize = 18)

    ax_xy = Axis(fig[1, 1],
                 title = "Surface tracer concentration",
                 xlabel = "x (km)",
                 ylabel = "y (km)",
                 aspect = DataAspect())

    ax_river = Axis(fig[1, 2],
                    title = string("x-z tracer concentration at river mouth, y = ", round(y_m[j_river_mouth] / 1e3; digits = 2), " km"),
                    xlabel = "x (km)",
                    ylabel = "z (m)")

    ax_y0 = Axis(fig[2, 1],
                 title = string("x-z tracer concentration at y = ", round(y_m[j_y0] / 1e3; digits = 2), " km"),
                 xlabel = "x (km)",
                 ylabel = "z (m)")

    ax_ymin = Axis(fig[2, 2],
                   title = string("x-z tracer concentration at y = ", round(y_m[j_ymin] / 1e3; digits = 2), " km"),
                   xlabel = "x (km)",
                   ylabel = "z (m)")

    n = Observable(1)

    c_santee_surface = lift(n) do nn
        Array(interior(c_santee_data[nn], :, :, Nz))
    end

    c_winyah_surface = lift(n) do nn
        Array(interior(c_winyah_data[nn], :, :, Nz))
    end

    c_santee_xz_river = lift(n) do nn
        Array(interior(c_santee_data[nn], :, j_river_mouth, :))
    end

    c_winyah_xz_river = lift(n) do nn
        Array(interior(c_winyah_data[nn], :, j_river_mouth, :))
    end

    c_santee_xz_y0 = lift(n) do nn
        Array(interior(c_santee_data[nn], :, j_y0, :))
    end

    c_winyah_xz_y0 = lift(n) do nn
        Array(interior(c_winyah_data[nn], :, j_y0, :))
    end

    c_santee_xz_ymin = lift(n) do nn
        Array(interior(c_santee_data[nn], :, j_ymin, :))
    end

    c_winyah_xz_ymin = lift(n) do nn
        Array(interior(c_winyah_data[nn], :, j_ymin, :))
    end

    hm_santee = heatmap!(ax_xy, xC, yC, c_santee_surface; colormap = :viridis, colorrange, alpha = 0.72)
    hm_winyah = heatmap!(ax_xy, xC, yC, c_winyah_surface; colormap = :magma, colorrange, alpha = 0.62)

    heatmap!(ax_river, xC, zC, c_santee_xz_river; colormap = :viridis, colorrange, alpha = 0.72)
    heatmap!(ax_river, xC, zC, c_winyah_xz_river; colormap = :magma, colorrange, alpha = 0.62)
    heatmap!(ax_y0, xC, zC, c_santee_xz_y0; colormap = :viridis, colorrange, alpha = 0.72)
    heatmap!(ax_y0, xC, zC, c_winyah_xz_y0; colormap = :magma, colorrange, alpha = 0.62)
    heatmap!(ax_ymin, xC, zC, c_santee_xz_ymin; colormap = :viridis, colorrange, alpha = 0.72)
    heatmap!(ax_ymin, xC, zC, c_winyah_xz_ymin; colormap = :magma, colorrange, alpha = 0.62)

    Colorbar(fig[:, 3], hm_santee; label = "Santee tracer")
    Colorbar(fig[:, 4], hm_winyah; label = "Winyah tracer")

    time_label = lift(n) do nn
        string("t = ", round(times[nn] / 60; digits = 1), " min")
    end
    Label(fig[0, :], time_label, fontsize = 22)

    CairoMakie.record(fig, joinpath(FILE_DIR, "surface_tracers_xy.mp4"), 1:Nt; framerate = 6) do nn
        n[] = nn
    end

    return nothing
end

function tracer_points(c_snapshot, xC, yC, zC; threshold = 0.02)
    C = Array(interior(c_snapshot, :, :, :))

    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    cs = Float64[]

    for k in eachindex(zC), j in eachindex(yC), i in eachindex(xC)
        cval = C[i, j, k]
        if cval >= threshold
            push!(xs, xC[i])
            push!(ys, yC[j])
            push!(zs, zC[k])
            push!(cs, cval)
        end
    end

    return xs, ys, zs, cs
end

function save_3d_tracer_animation()
    filepath = joinpath(FILE_DIR, "instantaneous_fields.jld2")
    c_santee_data = FieldTimeSeries(filepath, "c_santee")
    c_winyah_data = FieldTimeSeries(filepath, "c_winyah")
    Nt = min(length(c_santee_data.times), length(c_winyah_data.times))

    Nt == 0 && return nothing

    x_m = collect(xnodes(c_santee_data.grid, Center()))
    y_m = collect(ynodes(c_santee_data.grid, Center()))
    z_m = collect(znodes(c_santee_data.grid, Center()))

    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    zC = z_m
    times = c_santee_data.times

    bottom = [bathymetry(x, y) for x in x_m, y in y_m]

    fig = Figure(size = (1300, 900), fontsize = 18)
    ax = Axis3(fig[1, 1],
               title = "3D tracer plumes",
               xlabel = "x (km)",
               ylabel = "y (km)",
               zlabel = "z (m)",
               azimuth = 0.7pi,
               elevation = 0.18pi,
               aspect = (1, 1, 0.45))

    surface!(ax, xC, yC, bottom; colormap = :deep, colorrange = (-slope_depth, 0), alpha = 0.55, transparency = true)

    xs_s0, ys_s0, zs_s0, cs_s0 = tracer_points(c_santee_data[1], xC, yC, zC)
    xs_w0, ys_w0, zs_w0, cs_w0 = tracer_points(c_winyah_data[1], xC, yC, zC)

    xs_s = Observable(xs_s0)
    ys_s = Observable(ys_s0)
    zs_s = Observable(zs_s0)
    cs_s = Observable(cs_s0)

    xs_w = Observable(xs_w0)
    ys_w = Observable(ys_w0)
    zs_w = Observable(zs_w0)
    cs_w = Observable(cs_w0)

    santee_plume = scatter!(ax, xs_s, ys_s, zs_s;
                            color = cs_s,
                            colormap = :viridis,
                            colorrange = (0.0, 1.0),
                            markersize = 9,
                            alpha = 0.75,
                            transparency = true)

    winyah_plume = scatter!(ax, xs_w, ys_w, zs_w;
                            color = cs_w,
                            colormap = :magma,
                            colorrange = (0.0, 1.0),
                            markersize = 9,
                            alpha = 0.65,
                            transparency = true)

    Colorbar(fig[1, 2], santee_plume; label = "Santee tracer")
    Colorbar(fig[1, 3], winyah_plume; label = "Winyah tracer")

    time_label = Observable(string("t = ", round(times[1] / 60; digits = 1), " min"))
    Label(fig[0, :], time_label, fontsize = 22)

    xlims!(ax, extrema(xC)...)
    ylims!(ax, extrema(yC)...)
    zlims!(ax, -slope_depth, 5)

    CairoMakie.record(fig, joinpath(FILE_DIR, "tracers_3d.mp4"), 1:Nt; framerate = 6) do nn
        new_xs_s, new_ys_s, new_zs_s, new_cs_s = tracer_points(c_santee_data[nn], xC, yC, zC)
        new_xs_w, new_ys_w, new_zs_w, new_cs_w = tracer_points(c_winyah_data[nn], xC, yC, zC)
        xs_s[] = new_xs_s
        ys_s[] = new_ys_s
        zs_s[] = new_zs_s
        cs_s[] = new_cs_s
        xs_w[] = new_xs_w
        ys_w[] = new_ys_w
        zs_w[] = new_zs_w
        cs_w[] = new_cs_w
        time_label[] = string("t = ", round(times[nn] / 60; digits = 1), " min")
    end

    return nothing
end

function save_3d_tracer_contour_animation()
    filepath = joinpath(FILE_DIR, "instantaneous_fields.jld2")
    c_santee_data = FieldTimeSeries(filepath, "c_santee")
    c_winyah_data = FieldTimeSeries(filepath, "c_winyah")
    Nt = min(length(c_santee_data.times), length(c_winyah_data.times))

    Nt == 0 && return nothing

    x_m = collect(xnodes(c_santee_data.grid, Center()))
    y_m = collect(ynodes(c_santee_data.grid, Center()))
    z_m = collect(znodes(c_santee_data.grid, Center()))

    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    zC = z_m
    times = c_santee_data.times

    bottom = [bathymetry(x, y) for x in x_m, y in y_m]
    j_mouth = nearest_index(y_m, river_mouth_y)
    j_y0 = nearest_index(y_m, 0.0)

    Xxy = [x for x in xC, y in yC]
    Yxy = [y for x in xC, y in yC]
    Zsurface = fill(0.5, length(xC), length(yC))

    Xxz = [x for x in xC, z in zC]
    Zxz = [z for x in xC, z in zC]
    Ymouth = fill(yC[j_mouth], length(xC), length(zC))
    Ymid = fill(yC[j_y0], length(xC), length(zC))

    fig = Figure(size = (1400, 900), fontsize = 18)
    ax = Axis3(fig[1, 1],
               title = "3D tracer concentration faces",
               xlabel = "x (km)",
               ylabel = "y (km)",
               zlabel = "z (m)",
               azimuth = 0.7pi,
               elevation = 0.18pi,
               aspect = (1, 1, 0.45))

    surface!(ax, xC, yC, bottom; colormap = :deep, colorrange = (-slope_depth, 0), alpha = 0.35, transparency = true)

    c_santee_surface = Observable(Array(interior(c_santee_data[1], :, :, Nz)))
    c_winyah_surface = Observable(Array(interior(c_winyah_data[1], :, :, Nz)))
    c_santee_mouth = Observable(Array(interior(c_santee_data[1], :, j_mouth, :)))
    c_winyah_mouth = Observable(Array(interior(c_winyah_data[1], :, j_mouth, :)))
    c_santee_mid = Observable(Array(interior(c_santee_data[1], :, j_y0, :)))
    c_winyah_mid = Observable(Array(interior(c_winyah_data[1], :, j_y0, :)))

    santee_face = surface!(ax, Xxy, Yxy, Zsurface;
                           color = c_santee_surface,
                           colormap = :viridis,
                           colorrange = (0.0, 1.0),
                           alpha = 0.62,
                           transparency = true)

    winyah_face = surface!(ax, Xxy, Yxy, Zsurface;
                           color = c_winyah_surface,
                           colormap = :magma,
                           colorrange = (0.0, 1.0),
                           alpha = 0.52,
                           transparency = true)

    surface!(ax, Xxz, Ymouth, Zxz;
             color = c_santee_mouth,
             colormap = :viridis,
             colorrange = (0.0, 1.0),
             alpha = 0.68,
             transparency = true)

    surface!(ax, Xxz, Ymouth, Zxz;
             color = c_winyah_mouth,
             colormap = :magma,
             colorrange = (0.0, 1.0),
             alpha = 0.58,
             transparency = true)

    surface!(ax, Xxz, Ymid, Zxz;
             color = c_santee_mid,
             colormap = :viridis,
             colorrange = (0.0, 1.0),
             alpha = 0.52,
             transparency = true)

    surface!(ax, Xxz, Ymid, Zxz;
             color = c_winyah_mid,
             colormap = :magma,
             colorrange = (0.0, 1.0),
             alpha = 0.44,
             transparency = true)

    Colorbar(fig[1, 2], santee_face; label = "Santee tracer")
    Colorbar(fig[1, 3], winyah_face; label = "Winyah tracer")

    time_label = Observable(string("t = ", round(times[1] / 60; digits = 1), " min"))
    Label(fig[0, :], time_label, fontsize = 22)

    xlims!(ax, extrema(xC)...)
    ylims!(ax, extrema(yC)...)
    zlims!(ax, -slope_depth, 5)

    CairoMakie.record(fig, joinpath(FILE_DIR, "tracers_3d_contours.mp4"), 1:Nt; framerate = 6) do nn
        c_santee_surface[] = Array(interior(c_santee_data[nn], :, :, Nz))
        c_winyah_surface[] = Array(interior(c_winyah_data[nn], :, :, Nz))
        c_santee_mouth[] = Array(interior(c_santee_data[nn], :, j_mouth, :))
        c_winyah_mouth[] = Array(interior(c_winyah_data[nn], :, j_mouth, :))
        c_santee_mid[] = Array(interior(c_santee_data[nn], :, j_y0, :))
        c_winyah_mid[] = Array(interior(c_winyah_data[nn], :, j_y0, :))
        time_label[] = string("t = ", round(times[nn] / 60; digits = 1), " min")
    end

    return nothing
end

run!(simulation)
save_surface_buoyancy_animation()
save_surface_tracer_animation()
save_3d_tracer_animation()
save_3d_tracer_contour_animation()
