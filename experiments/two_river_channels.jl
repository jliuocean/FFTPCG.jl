using Oceananigans
using Oceananigans.ImmersedBoundaries: GridFittedBottom
using Oceananigans.Grids: xnodes, ynodes, znodes
using Oceananigans.Units
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index
using CairoMakie
using JLD2
using CUDA
using Printf
using Statistics
using ArgParse

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
            default = 3600.0
    end

    return parse_args(s)
end

args = parse_commandline()
const ARCH = uppercase(args["arch"])
ARCH in ("GPU", "CPU") || error("Invalid --arch $(ARCH); must be GPU or CPU")

arch = ARCH == "GPU" ? GPU() : CPU()

#####
##### Domain, river, and shelf parameters
#####

const Lx = 15e3       # m
const Ly = 15e3       # m
const H  = 20.0       # m, maximum shelf depth
const river_mouth_depth = 10.0 # m, depth at the northern river mouths

const Δx = 100.0      # m
const Δy = 100.0      # m
const Δz = 0.5        # m

const Nx = Int(Lx / Δx)
const Ny = Int(Ly / Δy)
const Nz = Int(H  / Δz)

const x₀ = 0.0
const x₁ = Lx
const y₀ = 0.0
const y₁ = Ly
const z₀ = -H
const z₁ = 0.0

const S_shelf = 35.0
const S_river = 30.0

const river_width = 500.0
const river_centers = (2.5e3, 12.5e3) # 10 km center-to-center spacing
const river_initial_length = 1e3
const river_discharge = 1000.0        # m³/s per river mouth
const river_velocity = river_discharge / (river_width * river_mouth_depth)
const total_river_transport = -2river_discharge
const south_outflow_velocity = total_river_transport / (Lx * H)

const νh = 0.05
const κh = 0.05
const Cd = 2e-3

filename = string("two_river_channels_hydrostatic_land_channels_Nx_", Nx, "_Ny_", Ny, "_Nz_", Nz, "_Q_", Int(river_discharge), "m3s_Sriver_", Int(S_river), "_1hour")
FILE_DIR = joinpath("Data", filename)
mkpath(FILE_DIR)

#####
##### Geometry, initial condition, and boundary conditions
#####
const river_mouth_y = y₁ - river_initial_length

@inline function water_depth(x, y)
    if y >= river_mouth_y
        return in_river_mouth(x) ? river_mouth_depth : 0.0
    else
        return river_mouth_depth + (H - river_mouth_depth) * (river_mouth_y - y) / (river_mouth_y - y₀)
    end
end

@inline bathymetry(x, y) = -water_depth(x, y)
@inline is_wet(x, y, z) = water_depth(x, y) > 0 && z >= bathymetry(x, y)
@inline function in_river_mouth(x)
    half_width = river_width / 2
    in_first  = abs(x - river_centers[1]) <= half_width
    in_second = abs(x - river_centers[2]) <= half_width
    return in_first || in_second
end

@inline function in_initial_river_patch(x, y)
    return in_river_mouth(x) && y >= y₁ - river_initial_length
end
@inline river_v_velocity(x, z, t) = in_river_mouth(x) && z >= -river_mouth_depth ? -river_velocity : 0.0
@inline river_salinity(x, z, t) = in_river_mouth(x) && z >= -river_mouth_depth ? S_river : S_shelf
@inline south_outflow(y, z, t) = south_outflow_velocity
@inline initial_salinity(x, y, z) = in_initial_river_patch(x, y) && is_wet(x, y, z) ? S_river : S_shelf

grid = RectilinearGrid(arch, Float64,
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (x₀, x₁),
                       y = (y₀, y₁),
                       z = (z₀, z₁),
                       topology = (Bounded, Bounded, Bounded))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

v_north_bc = NormalFlowBoundaryCondition(-river_velocity;
                                         scheme = PerturbationAdvection(target_transport = total_river_transport))
v_south_bc = NormalFlowBoundaryCondition(south_outflow_velocity;
                                         scheme = PerturbationAdvection())
S_north_bc = ValueBoundaryCondition(S_river)

quadratic_drag = BulkDrag(coefficient = Cd)

u_bcs = FieldBoundaryConditions(immersed = quadratic_drag, bottom = quadratic_drag)
v_bcs = FieldBoundaryConditions(immersed = quadratic_drag,
                                north = v_north_bc,
                                south = v_south_bc,
                                bottom = quadratic_drag)
w_bcs = FieldBoundaryConditions()
S_bcs = FieldBoundaryConditions(north = S_north_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs, S = S_bcs)

closure = ScalarDiffusivity(ν = νh, κ = κh)
free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7)

model = HydrostaticFreeSurfaceModel(grid;
                                   free_surface,
                                   momentum_advection = VectorInvariant(),
                                   tracer_advection = WENO(order = 5),
                                   tracers = (:S,),
                                   closure,
                                   boundary_conditions)

set!(model, S = initial_salinity)

#####
##### Initial condition plots
#####

function save_initial_condition_plots!(model)
    S = model.tracers.S
    x_m = collect(xnodes(model.grid, Center()))
    y_m = collect(ynodes(model.grid, Center()))
    zC = collect(znodes(model.grid, Center()))
    xC = x_m ./ 1e3
    yC = y_m ./ 1e3

    Sxy = Array(interior(S, :, :, Nz))
    Sxz = Array(interior(S, :, Ny, :))
    depth_xy = [water_depth(x, y) for x in x_m, y in y_m]

    fig_bathy = Figure(size = (900, 760), fontsize = 20)
    ax_bathy = Axis(fig_bathy[1, 1],
                    title = "Water depth",
                    xlabel = "x (km)",
                    ylabel = "y (km)",
                    aspect = DataAspect())
    hm_bathy = heatmap!(ax_bathy, xC, yC, depth_xy; colormap = :deep, colorrange = (0, H))
    Colorbar(fig_bathy[1, 2], hm_bathy; label = "Depth (m)")
    save(joinpath(FILE_DIR, "bathymetry_xy.png"), fig_bathy)


    for i in eachindex(x_m), j in eachindex(y_m)
        Sxy[i, j] = is_wet(x_m[i], y_m[j], zC[Nz]) ? Sxy[i, j] : NaN
    end

    for i in eachindex(x_m), k in eachindex(zC)
        Sxz[i, k] = is_wet(x_m[i], y_m[Ny], zC[k]) ? Sxz[i, k] : NaN
    end

    fig_xy = Figure(size = (900, 760), fontsize = 20)
    ax_xy = Axis(fig_xy[1, 1],
                 title = "Initial surface salinity",
                 xlabel = "x (km)",
                 ylabel = "y (km)",
                 aspect = DataAspect())
    hm_xy = heatmap!(ax_xy, xC, yC, Sxy; colormap = :haline, colorrange = (0, 35))
    Colorbar(fig_xy[1, 2], hm_xy; label = "Salinity")
    save(joinpath(FILE_DIR, "initial_surface_salinity_xy.png"), fig_xy)

    fig_xz = Figure(size = (980, 560), fontsize = 20)
    ax_xz = Axis(fig_xz[1, 1],
                 title = "Initial salinity x-z section near northern river mouths",
                 xlabel = "x (km)",
                 ylabel = "z (m)")
    hm_xz = heatmap!(ax_xz, xC, zC, Sxz; colormap = :haline, colorrange = (0, 35))
    lines!(ax_xz, xC, [bathymetry(x, y_m[Ny]) for x in x_m]; color = :black, linewidth = 3)
    Colorbar(fig_xz[1, 2], hm_xz; label = "Salinity")
    save(joinpath(FILE_DIR, "initial_salinity_xz.png"), fig_xz)

    return nothing
end

save_initial_condition_plots!(model)

if args["setup-only"]
    @info "Setup complete. Initial condition plots saved in $(FILE_DIR)."
    exit()
end

#####
##### Simulation
#####

stop_time = args["stop-time"]
Δt = 20.0
simulation = Simulation(model; Δt, stop_time)
time_wizard = TimeStepWizard(cfl = 0.6, max_change = 1.05, max_Δt = 60.0)
simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

u, v, w = model.velocities
S = model.tracers.S

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("i: %d, t: %s, wall t: %s, Δt: %s",
                   iteration(sim), prettytime(sim), prettytime(elapsed), prettytime(sim.Δt))

    stats = string(", max |u|: ", maximum(abs, sim.model.velocities.u),
                   ", max |v|: ", maximum(abs, sim.model.velocities.v),
                   ", min S: ", minimum(sim.model.tracers.S),
                   ", max S: ", maximum(sim.model.tracers.S))

    msg = string(msg, stats)

    wall_clock[] = time_ns()
    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20))

simulation.output_writers[:jld2] = JLD2Writer(model, (; u, v, w, S);
                                              filename = joinpath(FILE_DIR, "instantaneous_fields.jld2"),
                                              schedule = TimeInterval(10minutes),
                                              with_halos = true,
                                              overwrite_existing = true)

function save_surface_salinity_animation()
    filepath = joinpath(FILE_DIR, "instantaneous_fields.jld2")
    S_data = FieldTimeSeries(filepath, "S")
    Nt = length(S_data.times)

    Nt == 0 && return nothing

    x_m = collect(xnodes(S_data.grid, Center()))
    y_m = collect(ynodes(S_data.grid, Center()))
    z_surface = znodes(S_data.grid, Center())[Nz]
    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    times = S_data.times

    function surface_salinity_frame(n)
        S_surface = Array(interior(S_data[n], :, :, Nz))

        for i in eachindex(x_m), j in eachindex(y_m)
            S_surface[i, j] = is_wet(x_m[i], y_m[j], z_surface) ? S_surface[i, j] : NaN
        end

        return S_surface
    end

    fig = Figure(size = (900, 760), fontsize = 20)
    ax = Axis(fig[1, 1],
              title = "Surface salinity",
              xlabel = "x (km)",
              ylabel = "y (km)",
              aspect = DataAspect())

    n = Observable(1)
    S_surface = lift(n) do nn
        surface_salinity_frame(nn)
    end

    hm = heatmap!(ax, xC, yC, S_surface; colormap = :haline, colorrange = (0, 35))
    Colorbar(fig[1, 2], hm; label = "Salinity")

    time_label = lift(n) do nn
        string("t = ", round(times[nn] / 60; digits = 1), " min")
    end
    Label(fig[0, :], time_label, fontsize = 20)

    CairoMakie.record(fig, joinpath(FILE_DIR, "surface_salinity_xy.mp4"), 1:Nt; framerate = 6) do nn
        n[] = nn
    end

    return nothing
end

run!(simulation)
save_surface_salinity_animation()
