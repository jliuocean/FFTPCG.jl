using Oceananigans
using JLD2
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using Oceananigans.Utils: launch!
using Oceananigans.Units
using KernelAbstractions: @kernel, @index
using CUDA
using ArgParse
using CairoMakie

####
#### Command line
####

function parse_commandline()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--arch"
            help = "Architecture to use: GPU or CPU"
            arg_type = String
            default = "GPU"
        "--setup-only"
            help = "Build the model and save initial-condition plots without running"
            action = :store_true
        "--simulation-only"
            help = "Run the simulation and skip animation postprocessing"
            action = :store_true
        "--plot-only"
            help = "Skip the simulation and use an existing instantaneous_fields.jld2"
            action = :store_true
        "--surface-buoyancy-only"
            help = "With --plot-only, regenerate only surface_buoyancy_xy.mp4"
            action = :store_true
        "--surface-tracers-only"
            help = "With --plot-only, regenerate only surface_tracers_xy.mp4"
            action = :store_true
        "--surface-tracer-panels-only"
            help = "With --plot-only, regenerate only surface_tracers_panels_xy.mp4"
            action = :store_true
        "--surface-density-anomaly-only"
            help = "With --plot-only, regenerate only surface_density_xy.mp4"
            action = :store_true
        "--density-3d-only"
            help = "With --plot-only, regenerate only density_3d.mp4"
            action = :store_true
        "--surface-density-only"
            help = "With --plot-only, regenerate surface buoyancy, surface tracers, surface density, and density_3d MP4s"
            action = :store_true
        "--tracers-3d-only"
            help = "With --plot-only, regenerate only tracers_3d.mp4"
            action = :store_true
        "--all-tracers-3d-only"
            help = "With --plot-only, regenerate the combined and both single-river 3D tracer MP4s"
            action = :store_true
        "--winyah-3d-only"
            help = "With --plot-only, regenerate only Winyah_tracers_3d.mp4"
            action = :store_true
        "--santee-3d-only"
            help = "With --plot-only, regenerate only Santee_tracers_3d.mp4"
            action = :store_true
        "--stop-time"
            help = "Simulation stop time in seconds; negative selects the wind-case default"
            arg_type = Float64
            default = -1.0
        "--pickup"
            help = "false, latest, recent, highest, iteration, or checkpoint path"
            arg_type = String
            default = "false"
        "--checkpoint-interval"
            help = "Checkpoint interval in seconds"
            arg_type = Float64
            default = 3600.0
        "--horizontal-resolution"
            help = "Uniform horizontal grid spacing in meters"
            arg_type = Float64
            default = 100.0
        "--vertical-levels"
            help = "Number of uniform vertical grid levels"
            arg_type = Int
            default = 20
        "--output-interval"
            help = "Instantaneous 3D field output interval in seconds"
            arg_type = Float64
            default = 3600.0
        "--output-without-halos"
            help = "Exclude halo cells from instantaneous_fields.jld2"
            action = :store_true
        "--santee-discharge"
            help = "Santee River discharge (m³/s)"
            arg_type = Float64
            default = 1000.0
        "--no-santee-river"
            help = "Remove the Santee embayment, mouth, channel, and inflow"
            action = :store_true
        "--no-winyah-bay"
            help = "Remove the Winyah embayment, mouth, channel, jetties, and inflow"
            action = :store_true
        "--winyah-discharge"
            help = "Winyah Bay discharge (m³/s)"
            arg_type = Float64
            default = 750.0
        "--wind-speed"
            help = "Wind speed after ramp-up (m/s)"
            arg_type = Float64
            default = 4.0
        "--wind-case"
            help = "Wind forcing: schedule, westerly, southerly, or northerly"
            arg_type = String
            default = "schedule"
        "--wind-start-time"
            help = "Time when wind ramp begins (s)"
            arg_type = Float64
            default = 259200.0
        "--wind-ramp-time"
            help = "Wind ramp duration (s)"
            arg_type = Float64
            default = 21600.0
        "--wind-hold-time"
            help = "Constant-wind duration after ramp-up for compass wind cases (s)"
            arg_type = Float64
            default = 259200.0
        "--nw-turn-start-time"
            help = "Time when Westerly wind starts turning toward NW wind (s)"
            arg_type = Float64
            default = 345600.0
        "--nw-turn-time"
            help = "Duration of the smooth Westerly-to-NW turn (s)"
            arg_type = Float64
            default = 21600.0
        "--sw-duration"
            help = "Duration of the final SW wind phase (s)"
            arg_type = Float64
            default = 172800.0
        "--output-root"
            help = "Root directory for simulation output"
            arg_type = String
            default = "/mnt/workdir/jliu1/FFTPCG/Data"
    end

    return parse_args(settings)
end

function parse_pickup_argument(value)
    stripped = strip(value)
    lower = lowercase(stripped)
    lower in ("false", "none", "no", "0") && return false
    lower in ("true", "latest") && return :latest
    lower in ("recent", "recent_time_stamp") && return :recent_time_stamp
    lower in ("highest", "highest_iteration") && return :highest_iteration
    iteration = tryparse(Int, stripped)
    isnothing(iteration) || return iteration
    return stripped
end

args = parse_commandline()
args["setup-only"] && args["plot-only"] && error("--setup-only and --plot-only cannot be combined")
args["simulation-only"] && args["plot-only"] && error("--simulation-only and --plot-only cannot be combined")
args["simulation-only"] && args["setup-only"] && error("--simulation-only and --setup-only cannot be combined")
plot_targets = (args["surface-buoyancy-only"], args["surface-tracers-only"],
                args["surface-tracer-panels-only"],
                args["surface-density-anomaly-only"],
                args["density-3d-only"], args["surface-density-only"],
                args["tracers-3d-only"], args["all-tracers-3d-only"],
                args["winyah-3d-only"],
                args["santee-3d-only"])
count(identity, plot_targets) > 1 && error("Choose only one plot-only target")
any(plot_targets) && !args["plot-only"] && error("Plot targets require --plot-only")
args["simulation-only"] && any(plot_targets) && error("--simulation-only cannot be combined with plot targets")

const ARCH = uppercase(args["arch"])
ARCH in ("GPU", "CPU") || error("--arch must be GPU or CPU")
arch = ARCH == "GPU" ? GPU() : CPU()
const PICKUP = parse_pickup_argument(args["pickup"])

####
#### Rotated domain and physical parameters
####

# This is the physical 90-degree counter-clockwise rotation
# (x_rot, y_rot) = (-y_original, x_original).
const Lx = 20e3
const Ly = 40e3
const Lz = 20.0
const Δx = args["horizontal-resolution"]
const Δy = args["horizontal-resolution"]
const Nz = args["vertical-levels"]
Δx > 0 || error("--horizontal-resolution must be positive")
Nz > 0 || error("--vertical-levels must be positive")
isinteger(Lx / Δx) && isinteger(Ly / Δy) ||
    error("--horizontal-resolution must divide both Lx and Ly exactly")
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
const inlet_depth = 5.0
const inlet_center_spacing = 10e3
const santee_center_y = -inlet_center_spacing / 2
const winyah_center_y =  inlet_center_spacing / 2
const NO_SANTEE_RIVER = args["no-santee-river"]
const NO_WINYAH_BAY = args["no-winyah-bay"]
NO_SANTEE_RIVER && NO_WINYAH_BAY &&
    error("--no-santee-river and --no-winyah-bay cannot be combined")
const santee_discharge = args["santee-discharge"]
const winyah_discharge = args["winyah-discharge"]
santee_discharge >= 0 || error("--santee-discharge must be nonnegative")
winyah_discharge >= 0 || error("--winyah-discharge must be nonnegative")
NO_SANTEE_RIVER && santee_discharge != 0 &&
    error("--no-santee-river requires --santee-discharge 0")
NO_WINYAH_BAY && winyah_discharge != 0 &&
    error("--no-winyah-bay requires --winyah-discharge 0")
(santee_discharge + winyah_discharge) > 0 ||
    error("At least one river discharge must be positive")
const inlet_cross_sectional_area = inlet_width * inlet_depth
const santee_inlet_speed = santee_discharge / inlet_cross_sectional_area
const winyah_inlet_speed = winyah_discharge / inlet_cross_sectional_area
const inlet_transport = santee_discharge + winyah_discharge

const shelf_N² = 1e-5
const santee_N² = 8e-3
const winyah_N² = 1.6e-2
const river_bottom_buoyancy = -shelf_N² * inlet_depth
const santee_surface_buoyancy = river_bottom_buoyancy + santee_N² * inlet_depth
const winyah_surface_buoyancy = river_bottom_buoyancy + winyah_N² * inlet_depth
const plume_surface_buoyancy = max(santee_surface_buoyancy, winyah_surface_buoyancy)
const f₀ = 8e-5
const Cd = 2e-3

const ρ₀ = 1025.0
const ρ_air = 1.225
const Cᴰ_air = 1.3e-3
const WIND_SPEED = args["wind-speed"]
const WIND_CASE = lowercase(args["wind-case"])
const COMPASS_WIND_CASES = ("s", "se", "e", "ne", "n", "nw", "w", "sw")
WIND_CASE in ("schedule", "westerly", "southerly", "northerly", COMPASS_WIND_CASES...) ||
    error("--wind-case must be schedule, westerly, southerly, northerly, S, SE, E, NE, N, NW, W, or SW")
const WIND_START_TIME = args["wind-start-time"]
const WIND_RAMP_TIME = args["wind-ramp-time"]
const COMPASS_WIND_HOLD_TIME = args["wind-hold-time"]
const NW_TURN_START_TIME = args["nw-turn-start-time"]
const NW_TURN_TIME = args["nw-turn-time"]
const SW_DURATION = args["sw-duration"]
WIND_SPEED >= 0 || error("--wind-speed must be nonnegative")
WIND_START_TIME >= 0 || error("--wind-start-time must be nonnegative")
WIND_RAMP_TIME >= 0 || error("--wind-ramp-time must be nonnegative")
COMPASS_WIND_HOLD_TIME >= 0 || error("--wind-hold-time must be nonnegative")
NW_TURN_START_TIME >= WIND_START_TIME + WIND_RAMP_TIME ||
    error("--nw-turn-start-time must not precede the end of wind ramp-up")
NW_TURN_TIME >= 0 || error("--nw-turn-time must be nonnegative")
SW_DURATION >= 0 || error("--sw-duration must be nonnegative")
const SW_START_TIME = NW_TURN_START_TIME + NW_TURN_TIME
const SCHEDULE_STOP_TIME = SW_START_TIME + SW_DURATION
const CONSTANT_WIND_STOP_TIME = 3days
const COMPASS_WIND_STOP_TIME = WIND_START_TIME + WIND_RAMP_TIME + COMPASS_WIND_HOLD_TIME
const EXPECTED_STOP_TIME = WIND_CASE == "schedule" ? SCHEDULE_STOP_TIME :
                           WIND_CASE in COMPASS_WIND_CASES ? COMPASS_WIND_STOP_TIME :
                           CONSTANT_WIND_STOP_TIME
const STOP_TIME = args["stop-time"] < 0 ? EXPECTED_STOP_TIME : args["stop-time"]

const sponge_width = 2e3
const sponge_timescale = 30minutes
const sponge_rate = 1 / sponge_timescale
const initial_buoyancy_noise = 1e-6

####
#### Rotated bathymetry
####

# Constants in the original, unrotated coordinate system. They are used only
# inside original_water_depth; the model itself uses the rotated coordinates.
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
const original_santee_channel_south_y = original_river_mouth_y - santee_channel_length
const original_santee_channel_x = (santee_center_y - inlet_width / 2,
                                    santee_center_y + inlet_width / 2)
const river_shelf_transition_length = 1e3
const channel_edge_transition_width = 200.0

@inline in_original_santee_embayment(x) = abs(x - santee_center_y) <= inlet_width / 2
@inline in_original_winyah_embayment(x) = abs(x - winyah_center_y) <= inlet_width / 2
@inline in_original_embayment(x) =
    (!NO_SANTEE_RIVER && in_original_santee_embayment(x)) ||
    (!NO_WINYAH_BAY && in_original_winyah_embayment(x))

@inline function in_original_winyah_jetty(x, y)
    NO_WINYAH_BAY && return false
    along = original_jetty_south_y <= y <= original_river_mouth_y
    on_south_side = abs(x - original_winyah_jetty_x[1]) <= jetty_width / 2
    on_north_side = abs(x - original_winyah_jetty_x[2]) <= jetty_width / 2
    return along && (on_south_side || on_north_side)
end

@inline function in_original_winyah_channel(x, y)
    return !NO_WINYAH_BAY &&
           original_winyah_jetty_x[1] < x < original_winyah_jetty_x[2] &&
           original_jetty_south_y <= y <= original_river_mouth_y
end

@inline function in_original_santee_channel(x, y)
    return !NO_SANTEE_RIVER &&
           original_santee_channel_x[1] < x < original_santee_channel_x[2] &&
           original_santee_channel_south_y <= y <= original_river_mouth_y
end

@inline function original_shelf_depth(x, y)
    if y > original_river_mouth_y
        return in_original_embayment(x) ? inlet_depth : 0.0
    end

    offshore_distance = original_river_mouth_y - y
    if offshore_distance <= nearshore_slope_length
        return nearshore_slope_depth * clamp(offshore_distance / nearshore_slope_length, 0.0, 1.0)
    end

    outer_length = original_river_mouth_y - original_y₀ - nearshore_slope_length
    outer_fraction = clamp((offshore_distance - nearshore_slope_length) / outer_length, 0.0, 1.0)
    return nearshore_slope_depth + (slope_depth - nearshore_slope_depth) * outer_fraction
end

@inline function original_water_depth(x, y)
    in_original_winyah_jetty(x, y) && return 0.0
    in_original_winyah_channel(x, y) && return inlet_depth
    in_original_santee_channel(x, y) && return inlet_depth
    return original_shelf_depth(x, y)
end

# Inverse rotation: (x_original, y_original) = (y_rotated, -x_rotated).
@inline water_depth(x, y) = original_water_depth(y, -x)
@inline bathymetry(x, y) = -water_depth(x, y)
@inline is_wet(x, y, z) = water_depth(x, y) > 0 && z >= bathymetry(x, y)

if NO_SANTEE_RIVER
    @assert water_depth(rotated_coast_x, santee_center_y) == 0.0
else
    @assert water_depth(rotated_coast_x, santee_center_y) == inlet_depth
end
if NO_WINYAH_BAY
    @assert water_depth(rotated_coast_x, winyah_center_y) == 0.0
else
    @assert water_depth(rotated_coast_x, winyah_center_y) == inlet_depth
    @assert water_depth(-original_jetty_south_y, winyah_center_y) == inlet_depth
end

####
#### Inflow, compensation, wind, and initial condition
####

@inline in_santee_inlet(y, z) = !NO_SANTEE_RIVER &&
                                abs(y - santee_center_y) <= inlet_width / 2 &&
                                z >= -inlet_depth
@inline in_winyah_inlet(y, z) = !NO_WINYAH_BAY &&
                                abs(y - winyah_center_y) <= inlet_width / 2 &&
                                z >= -inlet_depth

@inline function u_inflow_profile(y, z, t)
    in_santee_inlet(y, z) && return santee_inlet_speed
    in_winyah_inlet(y, z) && return winyah_inlet_speed
    return 0.0
end

@inline ambient_buoyancy(z) = shelf_N² * z
@inline santee_buoyancy_profile(z) = river_bottom_buoyancy + santee_N² * (z + inlet_depth)
@inline winyah_buoyancy_profile(z) = river_bottom_buoyancy + winyah_N² * (z + inlet_depth)

@inline function b_inflow_profile(y, z, t)
    z >= -inlet_depth || return ambient_buoyancy(z)
    in_santee_inlet(y, z) && return santee_buoyancy_profile(z)
    in_winyah_inlet(y, z) && return winyah_buoyancy_profile(z)
    return ambient_buoyancy(z)
end

@inline c_santee_inflow_profile(y, z, t) = in_santee_inlet(y, z) ? 1.0 : 0.0
@inline c_winyah_inflow_profile(y, z, t) = in_winyah_inlet(y, z) ? 1.0 : 0.0

# Numerically integrate the wet cross-sectional area of the rotated south edge.
x_area_nodes = range(x₀ + Δx / 2, x₁ - Δx / 2; length = Nx)
const south_outlet_area = sum(water_depth(x, y₀) for x in x_area_nodes) * Δx
const outlet_speed = inlet_transport / south_outlet_area
@inline v_south_outflow(x, z, t) = -outlet_speed

@inline function smoothstep(η)
    η = clamp(η, 0.0, 1.0)
    return η^2 * (3 - 2η)
end

# Meteorological names specify where the wind comes from. Thus Westerly wind
# blows eastward, NW wind blows southeastward, and SW wind blows northeastward.
@inline function compass_wind_unit_vector()
    component = inv(sqrt(2.0))
    WIND_CASE == "s"  && return (0.0, 1.0)
    WIND_CASE == "se" && return (-component, component)
    WIND_CASE == "e"  && return (-1.0, 0.0)
    WIND_CASE == "ne" && return (-component, -component)
    WIND_CASE == "n"  && return (0.0, -1.0)
    WIND_CASE == "nw" && return (component, -component)
    WIND_CASE == "w"  && return (1.0, 0.0)
    return (component, component) # SW wind
end

@inline function compass_wind_ramp(t)
    t <= WIND_START_TIME && return 0.0
    WIND_RAMP_TIME == 0 && return 1.0
    return smoothstep((t - WIND_START_TIME) / WIND_RAMP_TIME)
end

@inline function wind_velocity(t)
    if WIND_CASE == "westerly"
        return (WIND_SPEED, 0.0)
    elseif WIND_CASE == "southerly"
        return (0.0, WIND_SPEED)
    elseif WIND_CASE == "northerly"
        return (0.0, -WIND_SPEED)
    elseif WIND_CASE != "schedule"
        û, v̂ = compass_wind_unit_vector()
        speed = WIND_SPEED * compass_wind_ramp(t)
        return (speed * û, speed * v̂)
    elseif t <= WIND_START_TIME
        return (0.0, 0.0)
    elseif t < WIND_START_TIME + WIND_RAMP_TIME
        fraction = WIND_RAMP_TIME == 0 ? 1.0 :
                   smoothstep((t - WIND_START_TIME) / WIND_RAMP_TIME)
        return (fraction * WIND_SPEED, 0.0)
    elseif t < NW_TURN_START_TIME
        return (WIND_SPEED, 0.0)
    elseif t < SW_START_TIME
        fraction = NW_TURN_TIME == 0 ? 1.0 :
                   smoothstep((t - NW_TURN_START_TIME) / NW_TURN_TIME)
        angle = -π / 4 * fraction
        return (WIND_SPEED * cos(angle), WIND_SPEED * sin(angle))
    else
        component = WIND_SPEED / sqrt(2)
        return (component, component)
    end
end

@inline function u_wind_stress(x, y, t)
    u₁₀, v₁₀ = wind_velocity(t)
    return -ρ_air * Cᴰ_air * hypot(u₁₀, v₁₀) * u₁₀ / ρ₀
end

@inline function v_wind_stress(x, y, t)
    u₁₀, v₁₀ = wind_velocity(t)
    return -ρ_air * Cᴰ_air * hypot(u₁₀, v₁₀) * v₁₀ / ρ₀
end

@inline function river_weight(x, y, z, center_y)
    is_wet(x, y, z) || return 0.0
    z >= -inlet_depth || return 0.0

    inner_half_width = inlet_width / 2 - channel_edge_transition_width / 2
    cross_coordinate = (abs(y - center_y) - inner_half_width) / channel_edge_transition_width
    cross_weight = 1 - smoothstep(cross_coordinate)

    original_shelfward_edge = original_river_mouth_y - river_shelf_transition_length
    original_y = -x
    along_coordinate = (original_y - original_shelfward_edge) /
                       (original_y₁ - original_shelfward_edge)
    along_weight = smoothstep(along_coordinate)
    return cross_weight * along_weight
end

@inline function b_initial(x, y, z)
    background = ambient_buoyancy(z)
    santee_weight = NO_SANTEE_RIVER ? 0.0 :
                     river_weight(x, y, z, santee_center_y)
    winyah_weight = NO_WINYAH_BAY ? 0.0 :
                    river_weight(x, y, z, winyah_center_y)
    bottom_noise_weight = smoothstep((z - bathymetry(x, y)) / Δz)
    return background +
           santee_weight * (santee_buoyancy_profile(z) - background) +
           winyah_weight * (winyah_buoyancy_profile(z) - background) +
           bottom_noise_weight * initial_buoyancy_noise * rand()
end

@info "Rotated configuration" Lx Ly Nx Ny Nz rotated_coast_x NO_SANTEE_RIVER NO_WINYAH_BAY
@info "River inflow and south compensation" santee_discharge winyah_discharge inlet_transport south_outlet_area outlet_speed
@info "Wind forcing in rotated coordinates" WIND_CASE WIND_SPEED EXPECTED_STOP_TIME
WIND_CASE == "schedule" &&
    @info "Scheduled wind timing" WIND_START_TIME WIND_RAMP_TIME NW_TURN_START_TIME NW_TURN_TIME SW_START_TIME SW_DURATION SCHEDULE_STOP_TIME
WIND_CASE in COMPASS_WIND_CASES &&
    @info "Compass wind timing" WIND_START_TIME WIND_RAMP_TIME COMPASS_WIND_HOLD_TIME COMPASS_WIND_STOP_TIME
westerly_wind = (WIND_SPEED, 0.0)
northwest_wind = (WIND_SPEED / sqrt(2), -WIND_SPEED / sqrt(2))
southwest_wind = (WIND_SPEED / sqrt(2), WIND_SPEED / sqrt(2))
@info "Wind vectors (eastward, northward)" westerly_wind northwest_wind southwest_wind
@info "Initial wind vector for selected case" WIND_CASE initial_wind = wind_velocity(0.0)
STOP_TIME == EXPECTED_STOP_TIME ||
    @warn "Stop time differs from the expected duration for this wind case" STOP_TIME EXPECTED_STOP_TIME WIND_CASE

####
#### Model
####

grid = RectilinearGrid(arch, Float64,
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (x₀, x₁),
                       y = (y₀, y₁),
                       z = (z₀, z₁),
                       topology = (Bounded, Bounded, Bounded))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

u_west_bc = NormalFlowBoundaryCondition(u_inflow_profile;
                                        scheme = PerturbationAdvection(target_transport = inlet_transport))
v_south_bc = NormalFlowBoundaryCondition(v_south_outflow;
                                         scheme = PerturbationAdvection())
b_west_bc = ValueBoundaryCondition(b_inflow_profile)
c_santee_west_bc = ValueBoundaryCondition(c_santee_inflow_profile)
c_winyah_west_bc = ValueBoundaryCondition(c_winyah_inflow_profile)

quadratic_drag = BulkDrag(coefficient = Cd)
no_slip_bc = ValueBoundaryCondition(0)
u_wind_bc = FluxBoundaryCondition(u_wind_stress)
v_wind_bc = FluxBoundaryCondition(v_wind_stress)

u_bcs = FieldBoundaryConditions(immersed = quadratic_drag,
                                bottom = quadratic_drag,
                                top = u_wind_bc,
                                west = u_west_bc)
v_bcs = FieldBoundaryConditions(immersed = quadratic_drag,
                                bottom = quadratic_drag,
                                top = v_wind_bc,
                                west = no_slip_bc,
                                south = v_south_bc)
w_bcs = FieldBoundaryConditions(west = no_slip_bc)
b_bcs = FieldBoundaryConditions(west = b_west_bc)
c_santee_bcs = FieldBoundaryConditions(west = c_santee_west_bc)
c_winyah_bcs = FieldBoundaryConditions(west = c_winyah_west_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs,
                       b = b_bcs, c_santee = c_santee_bcs,
                       c_winyah = c_winyah_bcs)

@inline function open_boundary_sponge_mask(x, y)
    east = smoothstep((x - (x₁ - sponge_width)) / sponge_width)
    south = smoothstep((y₀ + sponge_width - y) / sponge_width)
    north = smoothstep((y - (y₁ - sponge_width)) / sponge_width)
    return max(east, south, north)
end

@inline function non_south_sponge_mask(x, y)
    east = smoothstep((x - (x₁ - sponge_width)) / sponge_width)
    north = smoothstep((y - (y₁ - sponge_width)) / sponge_width)
    return max(east, north)
end

u_sponge(x, y, z, t, u) = -sponge_rate * open_boundary_sponge_mask(x, y) * u
v_sponge(x, y, z, t, v) = -sponge_rate * non_south_sponge_mask(x, y) * v
b_sponge(x, y, z, t, b) = -sponge_rate * open_boundary_sponge_mask(x, y) * (b - ambient_buoyancy(z))
c_santee_sponge(x, y, z, t, c) = -sponge_rate * open_boundary_sponge_mask(x, y) * c
c_winyah_sponge(x, y, z, t, c) = -sponge_rate * open_boundary_sponge_mask(x, y) * c

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

####
#### Output directory and initial plots
####

function duration_suffix(seconds)
    isinteger(seconds / 3600) && return string(Int(seconds / 3600), "h")
    return string(Int(round(seconds)), "s")
end

function speed_label(speed)
    return replace(string(round(speed; digits = 1)), "-" => "m", "." => "p")
end

function wind_case_suffix()
    if WIND_CASE == "schedule"
        return string("_Wind", speed_label(WIND_SPEED), "ms_W_NW_SW",
                      "_start", duration_suffix(WIND_START_TIME),
                      "_ramp", duration_suffix(WIND_RAMP_TIME),
                      "_NWstart", duration_suffix(NW_TURN_START_TIME),
                      "_NWturn", duration_suffix(NW_TURN_TIME),
                      "_SW", duration_suffix(SW_DURATION))
    elseif WIND_CASE in COMPASS_WIND_CASES
        return string("_Wind", speed_label(WIND_SPEED), "ms_",
                      uppercase(WIND_CASE),
                      "_calm", duration_suffix(WIND_START_TIME),
                      "_ramp", duration_suffix(WIND_RAMP_TIME),
                      "_hold", duration_suffix(COMPASS_WIND_HOLD_TIME))
    end
    return string("_Wind", speed_label(WIND_SPEED), "ms_",
                  uppercasefirst(WIND_CASE), "_constant")
end

const river_geometry_suffix = NO_SANTEE_RIVER ? "_NoSanteeRiver" :
                              NO_WINYAH_BAY ? "_NoWinyahBay" : ""
filename = string("MAMD_RotatedCCW_Lx20km_Ly40km_Winyah", Int(winyah_discharge),
                  "_Santee", Int(santee_discharge), river_geometry_suffix,
                  "_SouthOutflow",
                  "_Nx", Nx, "_Ny", Ny, "_Nz", Nz,
                  wind_case_suffix(),
                  "_total", duration_suffix(STOP_TIME))
FILE_DIR = joinpath(args["output-root"], filename)
mkpath(FILE_DIR)

nearest_index(nodes, value) = argmin(abs.(nodes .- value))

function save_initial_plots!(model)
    x_m = collect(xnodes(model.grid, Center()))
    y_m = collect(ynodes(model.grid, Center()))
    z_m = collect(znodes(model.grid, Center()))
    x_km = x_m ./ 1e3
    y_km = y_m ./ 1e3
    k_surface = length(z_m)
    i_santee = nearest_index(y_m, santee_center_y)
    i_winyah = nearest_index(y_m, winyah_center_y)

    depth = [water_depth(x, y) for x in x_m, y in y_m]
    b_surface = Array(interior(model.tracers.b, :, :, k_surface))
    b_santee = Array(interior(model.tracers.b, :, i_santee, :))
    b_winyah = Array(interior(model.tracers.b, :, i_winyah, :))

    for j in eachindex(y_m), i in eachindex(x_m)
        is_wet(x_m[i], y_m[j], z_m[k_surface]) || (b_surface[i, j] = NaN)
    end
    for k in eachindex(z_m), i in eachindex(x_m)
        is_wet(x_m[i], y_m[i_santee], z_m[k]) || (b_santee[i, k] = NaN)
        is_wet(x_m[i], y_m[i_winyah], z_m[k]) || (b_winyah[i, k] = NaN)
    end

    colorrange = (-1e-4, winyah_buoyancy_profile(z_m[k_surface]))
    fig = Figure(size = (1550, 1100), fontsize = 18)
    ax_depth = Axis(fig[1, 1], title = "Rotated bathymetry",
                    xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                    aspect = DataAspect())
    ax_surface = Axis(fig[2, 1], title = "Initial surface buoyancy",
                      xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                      aspect = DataAspect())
    ax_winyah = Axis(fig[1, 2], title = "Winyah centerline",
                     xlabel = "Eastward x (km)", ylabel = "z (m)")
    ax_santee = Axis(fig[2, 2], title = "Santee centerline",
                     xlabel = "Eastward x (km)", ylabel = "z (m)")

    heatmap!(ax_depth, x_km, y_km, depth; colormap = :deep, colorrange = (0, slope_depth))
    hm = heatmap!(ax_surface, x_km, y_km, b_surface;
                  colormap = :thermal, colorrange, nan_color = :lightgray)
    heatmap!(ax_winyah, x_km, z_m, b_winyah;
             colormap = :thermal, colorrange, nan_color = :lightgray)
    heatmap!(ax_santee, x_km, z_m, b_santee;
             colormap = :thermal, colorrange, nan_color = :lightgray)
    Colorbar(fig[:, 3], hm; label = "Buoyancy b (m s⁻²)")
    save(joinpath(FILE_DIR, "initial_rotated_bathymetry_and_buoyancy.png"), fig)
    return nothing
end

args["plot-only"] || save_initial_plots!(model)
if args["setup-only"]
    @info "Rotated setup complete" FILE_DIR
    exit()
end

####
#### Simulation and diagnostics
####

simulation = Simulation(model; Δt = 1.0, stop_time = STOP_TIME)
simulation.callbacks[:wizard] = Callback(TimeStepWizard(cfl = 0.3, max_change = 1.05, max_Δt = 10.0),
                                         IterationInterval(1))

u, v, w = model.velocities
b = model.tracers.b
c_santee = model.tracers.c_santee
c_winyah = model.tracers.c_winyah
divergence = CenterField(grid)

@kernel function _divergence!(target, u, v, w, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds target[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function compute_divergence!(target, model)
    launch!(architecture(model.grid), model.grid, :xyz, _divergence!,
            target, model.velocities.u, model.velocities.v,
            model.velocities.w, model.grid)
    return nothing
end

wall_clock = Ref(time_ns())
function progress(sim)
    now = time_ns()
    elapsed = 1e-9 * (now - wall_clock[])
    compute_divergence!(divergence, sim.model)
    println("i: ", iteration(sim),
            ", t: ", prettytime(sim), " / ", prettytime(sim.stop_time),
            ", wall Δt: ", prettytime(elapsed),
            ", model Δt: ", prettytime(sim.Δt),
            ", wind (u10, v10): ", wind_velocity(sim.model.clock.time),
            ", max |u|: ", maximum(abs, sim.model.velocities.u),
            ", max |v|: ", maximum(abs, sim.model.velocities.v),
            ", max |w|: ", maximum(abs, sim.model.velocities.w),
            ", max div: ", maximum(abs, divergence))
    wall_clock[] = now
    return nothing
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(5))

if !args["plot-only"]
    simulation.output_writers[:jld2] = JLD2Writer(model, (; b, c_santee, c_winyah);
                                                  filename = joinpath(FILE_DIR, "instantaneous_fields.jld2"),
                                                  schedule = TimeInterval(args["output-interval"]),
                                                  with_halos = !args["output-without-halos"],
                                                  overwrite_existing = PICKUP === false)
    simulation.output_writers[:checkpointer] = Checkpointer(model;
                                                            dir = FILE_DIR,
                                                            prefix = "checkpoint",
                                                            schedule = TimeInterval(args["checkpoint-interval"]),
                                                            overwrite_existing = true)
end

####
#### Rotated-coordinate animations
####

function save_surface_buoyancy_animation()
    b_data = FieldTimeSeries(joinpath(FILE_DIR, "instantaneous_fields.jld2"), "b")
    Nt = length(b_data.times)
    Nt == 0 && return nothing

    x_m = collect(xnodes(b_data.grid, Center()))
    y_m = collect(ynodes(b_data.grid, Center()))
    z_m = collect(znodes(b_data.grid, Center()))
    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    k_surface = length(z_m)
    i_coast = nearest_index(x_m, rotated_coast_x)
    i_nearshore = nearest_index(x_m, -5e3)
    i_offshore = nearest_index(x_m, -original_y₀)
    j_santee = nearest_index(y_m, santee_center_y)
    j_winyah = nearest_index(y_m, winyah_center_y)
    colorrange = (-plume_surface_buoyancy, plume_surface_buoyancy)

    fig = Figure(size = (1700, 1500), fontsize = 18)
    ax_xy = Axis(fig[1, 1], title = "Surface buoyancy",
                 xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                 aspect = DataAspect())
    ax_coast = Axis(fig[1, 2],
                    title = "y-z buoyancy at shoreline, x = $(round(xC[i_coast]; digits = 2)) km",
                    xlabel = "Northward y (km)", ylabel = "z (m)")
    ax_nearshore = Axis(fig[2, 1],
                        title = "y-z buoyancy at x = $(round(xC[i_nearshore]; digits = 2)) km",
                        xlabel = "Northward y (km)", ylabel = "z (m)")
    ax_offshore = Axis(fig[2, 2],
                       title = "y-z buoyancy at x = $(round(xC[i_offshore]; digits = 2)) km",
                       xlabel = "Northward y (km)", ylabel = "z (m)")
    ax_santee = Axis(fig[3, 1],
                     title = "x-z buoyancy, Santee channel y = $(round(yC[j_santee]; digits = 2)) km",
                     xlabel = "Eastward x (km)", ylabel = "z (m)")
    ax_winyah = Axis(fig[3, 2],
                     title = "x-z buoyancy, Winyah channel y = $(round(yC[j_winyah]; digits = 2)) km",
                     xlabel = "Eastward x (km)", ylabel = "z (m)")

    n = Observable(1)
    b_surface = lift(n) do nn
        Array(interior(b_data[nn], :, :, k_surface))
    end
    b_yz_coast = lift(n) do nn
        Array(interior(b_data[nn], i_coast, :, :))
    end
    b_yz_nearshore = lift(n) do nn
        Array(interior(b_data[nn], i_nearshore, :, :))
    end
    b_yz_offshore = lift(n) do nn
        Array(interior(b_data[nn], i_offshore, :, :))
    end
    b_xz_santee = lift(n) do nn
        Array(interior(b_data[nn], :, j_santee, :))
    end
    b_xz_winyah = lift(n) do nn
        Array(interior(b_data[nn], :, j_winyah, :))
    end

    hm = heatmap!(ax_xy, xC, yC, b_surface; colormap = :balance, colorrange)
    heatmap!(ax_coast, yC, z_m, b_yz_coast; colormap = :balance, colorrange)
    heatmap!(ax_nearshore, yC, z_m, b_yz_nearshore; colormap = :balance, colorrange)
    heatmap!(ax_offshore, yC, z_m, b_yz_offshore; colormap = :balance, colorrange)
    heatmap!(ax_santee, xC, z_m, b_xz_santee; colormap = :balance, colorrange)
    heatmap!(ax_winyah, xC, z_m, b_xz_winyah; colormap = :balance, colorrange)

    lines!(ax_santee, xC, [bathymetry(x, y_m[j_santee]) for x in x_m];
           color = :black, linewidth = 3)
    lines!(ax_winyah, xC, [bathymetry(x, y_m[j_winyah]) for x in x_m];
           color = :black, linewidth = 3)
    vlines!(ax_santee, [rotated_coast_x / 1e3]; color = :black,
            linewidth = 2, linestyle = :dot)
    vlines!(ax_winyah, [rotated_coast_x / 1e3]; color = :black,
            linewidth = 2, linestyle = :dot)

    Colorbar(fig[:, 3], hm; label = "Buoyancy b (m s⁻²)")
    time_label = lift(n) do nn
        "t = $(round(b_data.times[nn] / 3600; digits = 1)) hour"
    end
    Label(fig[0, :], time_label, fontsize = 22)
    CairoMakie.record(fig, joinpath(FILE_DIR, "surface_buoyancy_xy.mp4"), 1:Nt; framerate = 6) do nn
        n[] = nn
    end
    return nothing
end

function mask_tracer(snapshot; cutoff = 1e-6)
    values = Array(snapshot)
    values[values .<= cutoff] .= NaN
    return values
end

function save_surface_tracer_animation()
    santee = FieldTimeSeries(joinpath(FILE_DIR, "instantaneous_fields.jld2"), "c_santee")
    winyah = FieldTimeSeries(joinpath(FILE_DIR, "instantaneous_fields.jld2"), "c_winyah")
    Nt = min(length(santee.times), length(winyah.times))
    Nt == 0 && return nothing

    x_m = collect(xnodes(santee.grid, Center()))
    y_m = collect(ynodes(santee.grid, Center()))
    z_m = collect(znodes(santee.grid, Center()))
    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    k_surface = length(z_m)
    i_coast = nearest_index(x_m, rotated_coast_x)
    i_nearshore = nearest_index(x_m, -5e3)
    i_offshore = nearest_index(x_m, -original_y₀)
    j_santee = nearest_index(y_m, santee_center_y)
    j_winyah = nearest_index(y_m, winyah_center_y)
    colorrange = (0.0, 1.0)

    fig = Figure(size = (1850, 1500), fontsize = 18, backgroundcolor = :white)
    ax_xy = Axis(fig[1, 1], title = "Surface tracer concentration",
                 xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                 backgroundcolor = :white, aspect = DataAspect())
    ax_coast = Axis(fig[1, 2],
                    title = "y-z tracers at shoreline, x = $(round(xC[i_coast]; digits = 2)) km",
                    xlabel = "Northward y (km)", ylabel = "z (m)", backgroundcolor = :white)
    ax_nearshore = Axis(fig[2, 1],
                        title = "y-z tracers at x = $(round(xC[i_nearshore]; digits = 2)) km",
                        xlabel = "Northward y (km)", ylabel = "z (m)", backgroundcolor = :white)
    ax_offshore = Axis(fig[2, 2],
                       title = "y-z tracers at x = $(round(xC[i_offshore]; digits = 2)) km",
                       xlabel = "Northward y (km)", ylabel = "z (m)", backgroundcolor = :white)
    ax_santee = Axis(fig[3, 1],
                     title = "x-z tracers, Santee channel y = $(round(yC[j_santee]; digits = 2)) km",
                     xlabel = "Eastward x (km)", ylabel = "z (m)", backgroundcolor = :white)
    ax_winyah = Axis(fig[3, 2],
                     title = "x-z tracers, Winyah channel y = $(round(yC[j_winyah]; digits = 2)) km",
                     xlabel = "Eastward x (km)", ylabel = "z (m)", backgroundcolor = :white)

    n = Observable(1)
    s_surface = lift(n) do nn
        mask_tracer(interior(santee[nn], :, :, k_surface))
    end
    w_surface = lift(n) do nn
        mask_tracer(interior(winyah[nn], :, :, k_surface))
    end
    s_yz_coast = lift(n) do nn
        mask_tracer(interior(santee[nn], i_coast, :, :))
    end
    w_yz_coast = lift(n) do nn
        mask_tracer(interior(winyah[nn], i_coast, :, :))
    end
    s_yz_nearshore = lift(n) do nn
        mask_tracer(interior(santee[nn], i_nearshore, :, :))
    end
    w_yz_nearshore = lift(n) do nn
        mask_tracer(interior(winyah[nn], i_nearshore, :, :))
    end
    s_yz_offshore = lift(n) do nn
        mask_tracer(interior(santee[nn], i_offshore, :, :))
    end
    w_yz_offshore = lift(n) do nn
        mask_tracer(interior(winyah[nn], i_offshore, :, :))
    end
    s_xz_santee = lift(n) do nn
        mask_tracer(interior(santee[nn], :, j_santee, :))
    end
    w_xz_santee = lift(n) do nn
        mask_tracer(interior(winyah[nn], :, j_santee, :))
    end
    s_xz_winyah = lift(n) do nn
        mask_tracer(interior(santee[nn], :, j_winyah, :))
    end
    w_xz_winyah = lift(n) do nn
        mask_tracer(interior(winyah[nn], :, j_winyah, :))
    end

    hs = heatmap!(ax_xy, xC, yC, s_surface; colormap = :viridis, colorrange,
                  nan_color = :transparent, alpha = 0.72)
    hw = heatmap!(ax_xy, xC, yC, w_surface; colormap = :magma, colorrange,
                  nan_color = :transparent, alpha = 0.62)
    for (ax, s_values, w_values, horizontal) in
        ((ax_coast, s_yz_coast, w_yz_coast, yC),
         (ax_nearshore, s_yz_nearshore, w_yz_nearshore, yC),
         (ax_offshore, s_yz_offshore, w_yz_offshore, yC),
         (ax_santee, s_xz_santee, w_xz_santee, xC),
         (ax_winyah, s_xz_winyah, w_xz_winyah, xC))
        heatmap!(ax, horizontal, z_m, s_values; colormap = :viridis, colorrange,
                 nan_color = :transparent, alpha = 0.72)
        heatmap!(ax, horizontal, z_m, w_values; colormap = :magma, colorrange,
                 nan_color = :transparent, alpha = 0.62)
    end

    lines!(ax_santee, xC, [bathymetry(x, y_m[j_santee]) for x in x_m];
           color = :black, linewidth = 3)
    lines!(ax_winyah, xC, [bathymetry(x, y_m[j_winyah]) for x in x_m];
           color = :black, linewidth = 3)
    vlines!(ax_santee, [rotated_coast_x / 1e3]; color = :black,
            linewidth = 2, linestyle = :dot)
    vlines!(ax_winyah, [rotated_coast_x / 1e3]; color = :black,
            linewidth = 2, linestyle = :dot)

    Colorbar(fig[:, 3], hs; label = "Santee tracer")
    Colorbar(fig[:, 4], hw; label = "Winyah tracer")
    time_label = lift(n) do nn
        "t = $(round(santee.times[nn] / 3600; digits = 1)) hour"
    end
    Label(fig[0, :], time_label, fontsize = 22)
    CairoMakie.record(fig, joinpath(FILE_DIR, "surface_tracers_xy.mp4"), 1:Nt; framerate = 6) do nn
        n[] = nn
    end
    return nothing
end

function save_surface_tracer_panels_animation()
    filepath = joinpath(FILE_DIR, "instantaneous_fields.jld2")
    santee = FieldTimeSeries(filepath, "c_santee")
    winyah = FieldTimeSeries(filepath, "c_winyah")
    Nt = min(length(santee.times), length(winyah.times))
    Nt == 0 && return nothing

    x_m = collect(xnodes(santee.grid, Center()))
    y_m = collect(ynodes(santee.grid, Center()))
    z_m = collect(znodes(santee.grid, Center()))
    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    k_surface = length(z_m)
    colorrange = (0.0, 1.0)

    n = Observable(1)
    santee_surface = lift(n) do nn
        mask_tracer(interior(santee[nn], :, :, k_surface))
    end
    winyah_surface = lift(n) do nn
        mask_tracer(interior(winyah[nn], :, :, k_surface))
    end

    fig = Figure(size = (1900, 1050), fontsize = 18, backgroundcolor = :white)
    ax_both = Axis(fig[1, 1], title = "Surface Santee + Winyah tracers",
                   xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                   backgroundcolor = :white, aspect = DataAspect())
    ax_winyah = Axis(fig[1, 2], title = "Surface Winyah tracer",
                     xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                     backgroundcolor = :white, aspect = DataAspect())
    ax_santee = Axis(fig[1, 3], title = "Surface Santee tracer",
                     xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                     backgroundcolor = :white, aspect = DataAspect())

    heatmap!(ax_both, xC, yC, santee_surface; colormap = :viridis, colorrange,
             nan_color = :transparent, alpha = 0.72)
    heatmap!(ax_both, xC, yC, winyah_surface; colormap = :magma, colorrange,
             nan_color = :transparent, alpha = 0.62)
    winyah_plot = heatmap!(ax_winyah, xC, yC, winyah_surface;
                           colormap = :magma, colorrange,
                           nan_color = :transparent, alpha = 0.85)
    santee_plot = heatmap!(ax_santee, xC, yC, santee_surface;
                           colormap = :viridis, colorrange,
                           nan_color = :transparent, alpha = 0.85)

    for ax in (ax_both, ax_winyah, ax_santee)
        xlims!(ax, extrema(xC)...)
        ylims!(ax, extrema(yC)...)
    end
    Colorbar(fig[1, 4], winyah_plot; label = "Winyah tracer")
    Colorbar(fig[1, 5], santee_plot; label = "Santee tracer")

    time_label = lift(n) do nn
        "t = $(round(santee.times[nn] / 3600; digits = 1)) hour"
    end
    Label(fig[0, :], time_label, fontsize = 22)

    output_file = joinpath(FILE_DIR, "surface_tracers_panels_xy.mp4")
    CairoMakie.record(fig, output_file, 1:Nt; framerate = 6) do nn
        n[] = nn
    end
    @info "Saved three-panel surface tracer animation" output_file
    return nothing
end

function save_surface_density_animation()
    b_data = FieldTimeSeries(joinpath(FILE_DIR, "instantaneous_fields.jld2"), "b")
    Nt = length(b_data.times)
    Nt == 0 && return nothing

    x_m = collect(xnodes(b_data.grid, Center()))
    y_m = collect(ynodes(b_data.grid, Center()))
    z_m = collect(znodes(b_data.grid, Center()))
    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    k_surface = length(z_m)
    z_surface = z_m[k_surface]

    function surface_density(snapshot)
        buoyancy = Array(interior(snapshot, :, :, k_surface))
        density = -ρ₀ .* (buoyancy .- ambient_buoyancy(z_surface)) ./ 9.81
        for j in eachindex(y_m), i in eachindex(x_m)
            is_wet(x_m[i], y_m[j], z_surface) || (density[i, j] = NaN)
        end
        return density
    end

    n = Observable(1)
    density_surface = lift(n) do nn
        surface_density(b_data[nn])
    end

    fig = Figure(size = (1000, 1050), fontsize = 18, backgroundcolor = :white)
    ax = Axis(fig[1, 1], title = "Surface density anomaly",
              xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
              backgroundcolor = :white, aspect = DataAspect())
    hm = heatmap!(ax, xC, yC, density_surface; colormap = :balance,
                  colorrange = (-12.0, 12.0), nan_color = :transparent)
    Colorbar(fig[1, 2], hm; label = "Density anomaly (kg m⁻³)")
    time_label = lift(n) do nn
        "t = $(round(b_data.times[nn] / 3600; digits = 1)) hour"
    end
    Label(fig[0, :], time_label, fontsize = 22)

    output_file = joinpath(FILE_DIR, "surface_density_xy.mp4")
    CairoMakie.record(fig, output_file, 1:Nt; framerate = 6) do nn
        n[] = nn
    end
    @info "Saved surface density anomaly animation" output_file
    return nothing
end

function density_points(b_snapshot, x_m, y_m, z_m; threshold = 0.05, stride = 2)
    B = Array(interior(b_snapshot, :, :, :))
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    ρs = Float64[]

    for k in firstindex(z_m):stride:lastindex(z_m),
        j in firstindex(y_m):stride:lastindex(y_m),
        i in firstindex(x_m):stride:lastindex(x_m)

        is_wet(x_m[i], y_m[j], z_m[k]) || continue
        ρ_anomaly = -ρ₀ * (B[i, j, k] - ambient_buoyancy(z_m[k])) / 9.81
        if abs(ρ_anomaly) >= threshold
            push!(xs, x_m[i] / 1e3)
            push!(ys, y_m[j] / 1e3)
            push!(zs, z_m[k])
            push!(ρs, ρ_anomaly)
        end
    end

    return xs, ys, zs, ρs
end

function save_3d_density_animation()
    b_data = FieldTimeSeries(joinpath(FILE_DIR, "instantaneous_fields.jld2"), "b")
    Nt = length(b_data.times)
    Nt == 0 && return nothing

    x_m = collect(xnodes(b_data.grid, Center()))
    y_m = collect(ynodes(b_data.grid, Center()))
    z_m = collect(znodes(b_data.grid, Center()))
    xC = x_m ./ 1e3
    yC = y_m ./ 1e3
    bottom = [bathymetry(x, y) for x in x_m, y in y_m]

    xs0, ys0, zs0, ρs0 = density_points(b_data[1], x_m, y_m, z_m)
    xs = Observable(xs0)
    ys = Observable(ys0)
    zs = Observable(zs0)
    ρs = Observable(ρs0)

    fig = Figure(size = (1300, 900), fontsize = 18)
    ax = Axis3(fig[1, 1], title = "3D density anomaly plume",
               xlabel = "Eastward x (km)", ylabel = "Northward y (km)", zlabel = "z (m)",
               azimuth = 1.2pi, elevation = 0.18pi, aspect = (1, 1, 0.45))
    surface!(ax, xC, yC, bottom; colormap = :deep,
             colorrange = (-slope_depth, 0), alpha = 0.55, transparency = true)
    plume = scatter!(ax, xs, ys, zs; color = ρs, colormap = :balance,
                     colorrange = (-12.0, 12.0), markersize = 9,
                     alpha = 0.7, transparency = true)
    Colorbar(fig[1, 2], plume; label = "Density anomaly (kg m⁻³)")
    time_label = Observable("t = $(round(b_data.times[1] / 3600; digits = 1)) hour")
    Label(fig[0, :], time_label, fontsize = 22)
    xlims!(ax, extrema(xC)...)
    ylims!(ax, extrema(yC)...)
    zlims!(ax, -slope_depth, 5)

    output_file = joinpath(FILE_DIR, "density_3d.mp4")
    CairoMakie.record(fig, output_file, 1:Nt; framerate = 6) do nn
        new_xs, new_ys, new_zs, new_ρs = density_points(b_data[nn], x_m, y_m, z_m)
        xs[] = new_xs
        ys[] = new_ys
        zs[] = new_zs
        ρs[] = new_ρs
        time_label[] = "t = $(round(b_data.times[nn] / 3600; digits = 1)) hour"
    end
    @info "Saved 3D density anomaly animation" output_file
    return nothing
end

function tracer_points(snapshot, x, y, z; threshold = 0.02)
    values = Array(interior(snapshot, :, :, :))
    xs = Float64[]; ys = Float64[]; zs = Float64[]; cs = Float64[]
    for k in eachindex(z), j in eachindex(y), i in eachindex(x)
        c = values[i, j, k]
        if c >= threshold
            push!(xs, x[i]); push!(ys, y[j]); push!(zs, z[k]); push!(cs, c)
        end
    end
    return xs, ys, zs, cs
end

function save_3d_tracer_animation()
    filepath = joinpath(FILE_DIR, "instantaneous_fields.jld2")
    santee = FieldTimeSeries(filepath, "c_santee")
    winyah = FieldTimeSeries(filepath, "c_winyah")
    Nt = min(length(santee.times), length(winyah.times))
    Nt == 0 && return nothing

    x_m = collect(xnodes(santee.grid, Center()))
    y_m = collect(ynodes(santee.grid, Center()))
    z_m = collect(znodes(santee.grid, Center()))
    x = x_m ./ 1e3
    y = y_m ./ 1e3
    bottom = [bathymetry(xi, yi) for xi in x_m, yi in y_m]

    xs_s0, ys_s0, zs_s0, cs_s0 = tracer_points(santee[1], x, y, z_m)
    xs_w0, ys_w0, zs_w0, cs_w0 = tracer_points(winyah[1], x, y, z_m)
    xs_s = Observable(xs_s0); ys_s = Observable(ys_s0)
    zs_s = Observable(zs_s0); cs_s = Observable(cs_s0)
    xs_w = Observable(xs_w0); ys_w = Observable(ys_w0)
    zs_w = Observable(zs_w0); cs_w = Observable(cs_w0)

    fig = Figure(size = (1300, 900), fontsize = 18)
    ax = Axis3(fig[1, 1], title = "Rotated 3D river tracer plumes",
               xlabel = "Eastward x (km)", ylabel = "Northward y (km)", zlabel = "z (m)",
               azimuth = 1.2pi, elevation = 0.18pi, aspect = (1, 1, 0.45))
    surface!(ax, x, y, bottom; colormap = :deep, colorrange = (-slope_depth, 0),
             alpha = 0.55, transparency = true)
    santee_plume = scatter!(ax, xs_s, ys_s, zs_s; color = cs_s,
                            colormap = :viridis, colorrange = (0, 1),
                            markersize = 9, alpha = 0.75, transparency = true)
    winyah_plume = scatter!(ax, xs_w, ys_w, zs_w; color = cs_w,
                            colormap = :magma, colorrange = (0, 1),
                            markersize = 9, alpha = 0.65, transparency = true)
    Colorbar(fig[1, 2], santee_plume; label = "Santee tracer")
    Colorbar(fig[1, 3], winyah_plume; label = "Winyah tracer")
    time_label = Observable("t = 0.0 hour")
    Label(fig[0, :], time_label, fontsize = 22)
    xlims!(ax, extrema(x)...); ylims!(ax, extrema(y)...); zlims!(ax, -slope_depth, 5)

    output_file = joinpath(FILE_DIR, "tracers_3d.mp4")
    CairoMakie.record(fig, output_file, 1:Nt; framerate = 6) do n
        xn, yn, zn, cn = tracer_points(santee[n], x, y, z_m)
        xs_s[] = xn; ys_s[] = yn; zs_s[] = zn; cs_s[] = cn
        xn, yn, zn, cn = tracer_points(winyah[n], x, y, z_m)
        xs_w[] = xn; ys_w[] = yn; zs_w[] = zn; cs_w[] = cn
        time_label[] = "t = $(round(santee.times[n] / 3600; digits = 1)) hour"
    end
    @info "Saved combined 3D tracer animation" output_file
    return nothing
end

function save_3d_single_tracer_animation(field_name, river_name, colormap,
                                         output_filename; alpha = 0.75)
    data = FieldTimeSeries(joinpath(FILE_DIR, "instantaneous_fields.jld2"), field_name)
    Nt = length(data.times)
    Nt == 0 && return nothing

    x_m = collect(xnodes(data.grid, Center()))
    y_m = collect(ynodes(data.grid, Center()))
    z_m = collect(znodes(data.grid, Center()))
    x = x_m ./ 1e3
    y = y_m ./ 1e3
    bottom = [bathymetry(xi, yi) for xi in x_m, yi in y_m]
    x0, y0, z0, c0 = tracer_points(data[1], x, y, z_m)
    xs = Observable(x0); ys = Observable(y0); zs = Observable(z0); cs = Observable(c0)

    fig = Figure(size = (1200, 900), fontsize = 18, backgroundcolor = :white)
    ax = Axis3(fig[1, 1], title = "Rotated 3D $(river_name) tracer plume",
               xlabel = "Eastward x (km)", ylabel = "Northward y (km)", zlabel = "z (m)",
               azimuth = 1.2pi, elevation = 0.18pi, aspect = (1, 1, 0.45))
    surface!(ax, x, y, bottom; colormap = :deep, colorrange = (-slope_depth, 0),
             alpha = 0.55, transparency = true)
    plume = scatter!(ax, xs, ys, zs; color = cs, colormap,
                     colorrange = (0, 1), markersize = 9, alpha,
                     transparency = true)
    Colorbar(fig[1, 2], plume; label = "$(river_name) tracer")
    time_label = Observable("t = 0.0 hour")
    Label(fig[0, :], time_label, fontsize = 22)
    xlims!(ax, extrema(x)...); ylims!(ax, extrema(y)...); zlims!(ax, -slope_depth, 5)

    output_file = joinpath(FILE_DIR, output_filename)
    CairoMakie.record(fig, output_file, 1:Nt; framerate = 6) do n
        xn, yn, zn, cn = tracer_points(data[n], x, y, z_m)
        xs[] = xn; ys[] = yn; zs[] = zn; cs[] = cn
        time_label[] = "t = $(round(data.times[n] / 3600; digits = 1)) hour"
    end
    @info "Saved single-river 3D tracer animation" river_name output_file
    return nothing
end

save_3d_winyah_tracer_animation() =
    save_3d_single_tracer_animation("c_winyah", "Winyah", :magma,
                                    "Winyah_tracers_3d.mp4"; alpha = 0.65)

save_3d_santee_tracer_animation() =
    save_3d_single_tracer_animation("c_santee", "Santee", :viridis,
                                    "Santee_tracers_3d.mp4"; alpha = 0.75)

if args["plot-only"]
    data_file = joinpath(FILE_DIR, "instantaneous_fields.jld2")
    isfile(data_file) || error("Cannot plot; file does not exist: $data_file")
    @info "Plot-only mode" data_file
else
    run!(simulation; pickup = PICKUP, checkpoint_at_end = true)
end

if args["simulation-only"]
    @info "Simulation complete; skipping animation postprocessing" FILE_DIR
    exit()
end

if args["surface-buoyancy-only"]
    save_surface_buoyancy_animation()
elseif args["surface-tracers-only"]
    save_surface_tracer_animation()
elseif args["surface-tracer-panels-only"]
    save_surface_tracer_panels_animation()
elseif args["surface-density-anomaly-only"]
    save_surface_density_animation()
elseif args["density-3d-only"]
    save_3d_density_animation()
elseif args["surface-density-only"]
    save_surface_buoyancy_animation()
    save_surface_tracer_animation()
    save_surface_density_animation()
    save_3d_density_animation()
elseif args["tracers-3d-only"]
    save_3d_tracer_animation()
elseif args["all-tracers-3d-only"]
    save_3d_tracer_animation()
    save_3d_winyah_tracer_animation()
    save_3d_santee_tracer_animation()
elseif args["winyah-3d-only"]
    save_3d_winyah_tracer_animation()
elseif args["santee-3d-only"]
    save_3d_santee_tracer_animation()
else
    save_surface_buoyancy_animation()
    save_surface_tracer_animation()
    save_surface_tracer_panels_animation()
    save_surface_density_animation()
    save_3d_density_animation()
    save_3d_tracer_animation()
    save_3d_winyah_tracer_animation()
    save_3d_santee_tracer_animation()
end

