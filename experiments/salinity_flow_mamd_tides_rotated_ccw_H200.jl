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
            help = "Architecture: GPU or CPU"
            arg_type = String
            default = "GPU"
        "--setup-only"
            help = "Build the model and save the initial-condition summary without running"
            action = :store_true
        "--skip-setup-plot"
            help = "Skip the initial-condition summary plot (useful for smoke tests and batches)"
            action = :store_true
        "--stop-time"
            help = "Stop time in seconds; negative selects six M2 cycles (74.4 h)"
            arg_type = Float64
            default = -1.0
        "--pickup"
            help = "false, latest, recent, highest, iteration, or checkpoint path"
            arg_type = String
            default = "false"
        "--horizontal-resolution"
            help = "Uniform horizontal grid spacing in meters"
            arg_type = Float64
            default = 100.0
        "--vertical-levels"
            help = "Number of uniform vertical levels; 20 gives dz = 1 m"
            arg_type = Int
            default = 20
        "--eastern-boundary-x"
            help = "Eastern boundary coordinate in meters; west remains at -10 km"
            arg_type = Float64
            default = 30e3
        "--east-sponge-width"
            help = "Width of the eastern velocity sponge in meters"
            arg_type = Float64
            default = 3e3
        "--east-sponge-timescale"
            help = "Maximum eastern velocity-sponge relaxation timescale in seconds"
            arg_type = Float64
            default = 30minutes
        "--east-gravity-wave-speed"
            help = "Additional eastern momentum-radiation phase speed in m/s"
            arg_type = Float64
            default = 0.5
        "--east-tangential-sponge-factor"
            help = "Eastern v-sponge rate relative to the normal-velocity sponge rate"
            arg_type = Float64
            default = 1.0
        "--santee-freshwater-discharge"
            help = "Time-mean freshwater-equivalent Santee discharge (m3/s)"
            arg_type = Float64
            default = 3200 / 7
        "--winyah-freshwater-discharge"
            help = "Time-mean freshwater-equivalent Winyah discharge (m3/s)"
            arg_type = Float64
            default = 2400 / 7
        "--m2-period"
            help = "M2 discharge-modulation period in seconds"
            arg_type = Float64
            default = 12.4hours
        "--buoyancy-delay-cycles"
            help = "Number of initial tidal cycles with ambient-salinity inflow"
            arg_type = Float64
            default = 1.0
        "--wind-stress"
            help = "Maximum physical wind-stress magnitude (Pa)"
            arg_type = Float64
            default = 0.03
        "--wind-direction"
            help = "Meteorological wind direction: CALM, S, SW, W, NW, N, NE, E, or SE"
            arg_type = String
            default = "SE"
        "--progress-interval"
            help = "Iterations between expensive progress diagnostics"
            arg_type = Int
            default = 20
        "--output-interval"
            help = "3D output interval in seconds"
            arg_type = Float64
            default = 3600.0
        "--checkpoint-interval"
            help = "Checkpoint interval in seconds"
            arg_type = Float64
            default = 12.4hours
        "--output-without-halos"
            help = "Exclude halos from instantaneous_fields.jld2"
            action = :store_true
        "--output-root"
            help = "Root output directory"
            arg_type = String
            default = "/mnt/workdir/jliu1/FFTPCG/Data"
        "--output-dir"
            help = "Exact output directory; empty uses the parameter-derived name under --output-root"
            arg_type = String
            default = ""
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
    iteration_number = tryparse(Int, stripped)
    isnothing(iteration_number) || return iteration_number
    return String(stripped)
end

args = parse_commandline()
const ARCH = uppercase(args["arch"])
ARCH in ("GPU", "CPU") || error("--arch must be GPU or CPU")
args["horizontal-resolution"] > 0 || error("--horizontal-resolution must be positive")
args["vertical-levels"] > 0 || error("--vertical-levels must be positive")
args["eastern-boundary-x"] > -10e3 ||
    error("--eastern-boundary-x must be east of the fixed -10 km western boundary")
args["east-sponge-width"] > 0 || error("--east-sponge-width must be positive")
args["east-sponge-timescale"] > 0 || error("--east-sponge-timescale must be positive")
args["east-gravity-wave-speed"] >= 0 ||
    error("--east-gravity-wave-speed must be nonnegative")
0 <= args["east-tangential-sponge-factor"] <= 1 ||
    error("--east-tangential-sponge-factor must lie between 0 and 1")
args["santee-freshwater-discharge"] >= 0 || error("Santee discharge must be nonnegative")
args["winyah-freshwater-discharge"] >= 0 || error("Winyah discharge must be nonnegative")
args["m2-period"] > 0 || error("--m2-period must be positive")
args["buoyancy-delay-cycles"] >= 0 || error("--buoyancy-delay-cycles must be nonnegative")
args["wind-stress"] >= 0 || error("--wind-stress must be nonnegative")
args["progress-interval"] > 0 || error("--progress-interval must be positive")
args["output-interval"] > 0 || error("--output-interval must be positive")
args["checkpoint-interval"] > 0 || error("--checkpoint-interval must be positive")
const WIND_DIRECTION = uppercase(args["wind-direction"])
const VALID_WIND_DIRECTIONS = ("CALM", "S", "SW", "W", "NW", "N", "NE", "E", "SE")
WIND_DIRECTION in VALID_WIND_DIRECTIONS ||
    error("--wind-direction must be one of $(join(VALID_WIND_DIRECTIONS, ", "))")

arch = ARCH == "GPU" ? GPU() : CPU()
const PICKUP = parse_pickup_argument(args["pickup"])

####
#### Domain and the complete rotated Winyah-Santee bathymetry
####

# Physical 90-degree counter-clockwise rotation:
# (x_rotated, y_rotated) = (-y_original, x_original).
const x₀ = -10e3
const x₁ = args["eastern-boundary-x"]
const Lx = x₁ - x₀
const Ly = 40e3
const Lz = 20.0
const Δx = args["horizontal-resolution"]
const Δy = Δx
const Nz = args["vertical-levels"]
isinteger(Lx / Δx) && isinteger(Ly / Δy) ||
    error("--horizontal-resolution must divide both domain lengths exactly")
const Nx = Int(Lx / Δx)
const Ny = Int(Ly / Δy)
const Δz = Lz / Nz

const y₀ = -Ly / 2
const y₁ =  Ly / 2
const z₀ = -Lz
const z₁ = 0.0

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
const original_santee_channel_south_y = original_river_mouth_y - santee_channel_length
const original_santee_channel_x = (santee_center_y - inlet_width / 2,
                                    santee_center_y + inlet_width / 2)

@inline in_original_santee_embayment(x) = abs(x - santee_center_y) <= inlet_width / 2
@inline in_original_winyah_embayment(x) = abs(x - winyah_center_y) <= inlet_width / 2
@inline in_original_embayment(x) =
    in_original_santee_embayment(x) || in_original_winyah_embayment(x)

@inline function in_original_winyah_jetty(x, y)
    along = original_jetty_south_y <= y <= original_river_mouth_y
    on_south_side = abs(x - original_winyah_jetty_x[1]) <= jetty_width / 2
    on_north_side = abs(x - original_winyah_jetty_x[2]) <= jetty_width / 2
    return along && (on_south_side || on_north_side)
end

@inline function in_original_winyah_channel(x, y)
    return original_winyah_jetty_x[1] < x < original_winyah_jetty_x[2] &&
           original_jetty_south_y <= y <= original_river_mouth_y
end

@inline function in_original_santee_channel(x, y)
    return original_santee_channel_x[1] < x < original_santee_channel_x[2] &&
           original_santee_channel_south_y <= y <= original_river_mouth_y
end

@inline function original_shelf_depth(x, y)
    if y > original_river_mouth_y
        return in_original_embayment(x) ? inlet_depth : 0.0
    end

    offshore_distance = original_river_mouth_y - y
    if offshore_distance <= nearshore_slope_length
        return nearshore_slope_depth *
               clamp(offshore_distance / nearshore_slope_length, 0.0, 1.0)
    end

    outer_length = original_river_mouth_y - original_y₀ - nearshore_slope_length
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

@assert water_depth(rotated_coast_x, santee_center_y) == inlet_depth
@assert water_depth(rotated_coast_x, winyah_center_y) == inlet_depth
@assert water_depth(-original_jetty_south_y, winyah_center_y) == inlet_depth

####
#### Model-run-S salinity, M2 discharge, and wind forcing
####

const ambient_salinity = 34.0
const inflow_surface_salinity = 12.0
const inflow_bottom_salinity = 24.0
const tracer_lower_bound = inflow_surface_salinity
const tracer_upper_bound = ambient_salinity
const tracer_span = tracer_upper_bound - tracer_lower_bound
const inflow_mean_salinity = 0.5 * (inflow_surface_salinity +
                                    inflow_bottom_salinity)
const reference_density = 1025.0
const haline_contraction = 7.8e-4
const gravitational_acceleration = 9.81
const constant_temperature = 20.0
const f₀ = 8e-5
const Cd = 5e-3

const M2_PERIOD = args["m2-period"]
const M2_FREQUENCY = 2π / M2_PERIOD
const BUOYANCY_START_TIME = args["buoyancy-delay-cycles"] * M2_PERIOD
const NOMINAL_STOP_TIME = 6M2_PERIOD
const STOP_TIME = args["stop-time"] < 0 ? NOMINAL_STOP_TIME : args["stop-time"]

# Qr in Yankovsky & Yankovsky (2024), Eq. (3), is freshwater-equivalent
# discharge. Multiplication by s0 / (s0 - si) converts it to brackish-water
# volume transport at mean inlet salinity si.
const santee_freshwater_discharge = args["santee-freshwater-discharge"]
const winyah_freshwater_discharge = args["winyah-freshwater-discharge"]
const brackish_transport_factor =
    ambient_salinity / (ambient_salinity - inflow_mean_salinity)
const santee_mean_volume_transport =
    santee_freshwater_discharge * brackish_transport_factor
const winyah_mean_volume_transport =
    winyah_freshwater_discharge * brackish_transport_factor
const total_mean_volume_transport =
    santee_mean_volume_transport + winyah_mean_volume_transport
const inlet_cross_sectional_area = inlet_width * inlet_depth

@inline tidal_discharge_factor(t) = 1 + sin(M2_FREQUENCY * t)
@inline santee_volume_transport(t) =
    santee_mean_volume_transport * tidal_discharge_factor(t)
@inline winyah_volume_transport(t) =
    winyah_mean_volume_transport * tidal_discharge_factor(t)
@inline total_volume_transport(t) =
    total_mean_volume_transport * tidal_discharge_factor(t)

@inline in_santee_inlet(y, z) =
    abs(y - santee_center_y) <= inlet_width / 2 && z >= -inlet_depth
@inline in_winyah_inlet(y, z) =
    abs(y - winyah_center_y) <= inlet_width / 2 && z >= -inlet_depth

@inline function u_inflow_profile(y, z, t)
    in_santee_inlet(y, z) &&
        return santee_volume_transport(t) / inlet_cross_sectional_area
    in_winyah_inlet(y, z) &&
        return winyah_volume_transport(t) / inlet_cross_sectional_area
    return 0.0
end

@inline function river_salinity_profile(z)
    depth_fraction = clamp(-z / inlet_depth, 0.0, 1.0)
    return inflow_surface_salinity +
           (inflow_bottom_salinity - inflow_surface_salinity) * depth_fraction
end

@inline river_salinity_is_active(t) = t >= BUOYANCY_START_TIME

@inline function S_inflow_profile(y, z, t)
    in_river = in_santee_inlet(y, z) || in_winyah_inlet(y, z)
    return in_river && river_salinity_is_active(t) ?
           river_salinity_profile(z) : ambient_salinity
end

@inline c_santee_inflow_profile(y, z, t) =
    in_santee_inlet(y, z) && river_salinity_is_active(t) ?
    tracer_upper_bound : tracer_lower_bound
@inline c_winyah_inflow_profile(y, z, t) =
    in_winyah_inlet(y, z) && river_salinity_is_active(t) ?
    tracer_upper_bound : tracer_lower_bound

# Numerically integrate the wet cross-sectional area of the eastern edge.
# Positive u at the eastern boundary points out of the model domain.
y_area_nodes = range(y₀ + Δy / 2, y₁ - Δy / 2; length = Ny)
const east_outlet_area =
    sum(water_depth(x₁, y) for y in y_area_nodes) * Δy
@inline u_east_outflow(y, z, t) =
    total_volume_transport(t) / east_outlet_area

@inline function smoothstep(η)
    η = clamp(η, 0.0, 1.0)
    return η^2 * (3 - 2η)
end

# Meteorological direction denotes where the wind comes from. Wind is zero for
# one M2 cycle, ramps smoothly over the following half cycle, and then remains
# at full strength.
function meteorological_wind_unit_vector(direction)
    d = inv(sqrt(2.0))
    direction == "CALM" && return ( 0.0,  0.0)
    direction == "S"    && return ( 0.0,  1.0)
    direction == "SW"   && return ( d,     d)
    direction == "W"    && return ( 1.0,  0.0)
    direction == "NW"   && return ( d,    -d)
    direction == "N"    && return ( 0.0, -1.0)
    direction == "NE"   && return (-d,    -d)
    direction == "E"    && return (-1.0,  0.0)
    direction == "SE"   && return (-d,     d)
    error("Unsupported meteorological wind direction: $direction")
end

const WIND_STRESS_MAGNITUDE = args["wind-stress"]
const WIND_START_TIME = M2_PERIOD
const WIND_RAMP_TIME = 0.5M2_PERIOD
const WIND_FULL_TIME = WIND_START_TIME + WIND_RAMP_TIME
const WIND_UNIT_VECTOR = meteorological_wind_unit_vector(WIND_DIRECTION)
const τx_wind = WIND_STRESS_MAGNITUDE * WIND_UNIT_VECTOR[1]
const τy_wind = WIND_STRESS_MAGNITUDE * WIND_UNIT_VECTOR[2]
const Qx_wind = -τx_wind / reference_density
const Qy_wind = -τy_wind / reference_density

@inline function wind_ramp(t)
    t <= WIND_START_TIME && return 0.0
    t >= WIND_FULL_TIME && return 1.0
    return smoothstep((t - WIND_START_TIME) / WIND_RAMP_TIME)
end

@inline u_wind_stress(x, y, t) = wind_ramp(t) * Qx_wind
@inline v_wind_stress(x, y, t) = wind_ramp(t) * Qy_wind

const sponge_width = 2e3
const sponge_timescale = 30minutes
const sponge_rate = 1 / sponge_timescale
const EAST_SPONGE_WIDTH = args["east-sponge-width"]
const EAST_SPONGE_TIMESCALE = args["east-sponge-timescale"]
const EAST_SPONGE_RATE = 1 / EAST_SPONGE_TIMESCALE
const EAST_GRAVITY_WAVE_SPEED = args["east-gravity-wave-speed"]
const EAST_TANGENTIAL_SPONGE_FACTOR =
    args["east-tangential-sponge-factor"]

x₁ - EAST_SPONGE_WIDTH > rotated_coast_x ||
    error("The eastern sponge must begin offshore of the rotated coastline")

@inline function north_south_sponge_mask(x, y)
    south = smoothstep((y₀ + sponge_width - y) / sponge_width)
    north = smoothstep((y - (y₁ - sponge_width)) / sponge_width)
    return max(south, north)
end

@inline east_velocity_sponge_mask(x) =
    smoothstep((x - (x₁ - EAST_SPONGE_WIDTH)) / EAST_SPONGE_WIDTH)

####
#### Grid, boundary conditions, and model
####

@info "Salinity-tide rotated configuration" Lx Ly Lz Nx Ny Nz Δx Δz
@info "Complete two-river geometry" santee_center_y winyah_center_y rotated_coast_x
@info "Model-run-S salinity" ambient_salinity inflow_surface_salinity inflow_bottom_salinity inflow_mean_salinity
@info "M2 forcing" M2_PERIOD M2_FREQUENCY BUOYANCY_START_TIME NOMINAL_STOP_TIME STOP_TIME
@info "Freshwater-equivalent discharge" santee_freshwater_discharge winyah_freshwater_discharge
@info "Mean brackish transport" santee_mean_volume_transport winyah_mean_volume_transport total_mean_volume_transport
@info "Eastern compensation outflow" east_outlet_area mean_east_outlet_speed = total_mean_volume_transport / east_outlet_area
@info "Eastern momentum radiation and sponge" EAST_GRAVITY_WAVE_SPEED EAST_SPONGE_WIDTH EAST_SPONGE_TIMESCALE EAST_TANGENTIAL_SPONGE_FACTOR
@info "Wind stress schedule" WIND_DIRECTION WIND_STRESS_MAGNITUDE WIND_START_TIME WIND_RAMP_TIME WIND_FULL_TIME
@info "Full-strength wind stress components" τx_wind τy_wind Qx_wind Qy_wind

grid = RectilinearGrid(arch, Float64,
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (x₀, x₁),
                       y = (y₀, y₁),
                       z = (z₀, z₁),
                       topology = (Bounded, Bounded, Bounded))
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

# Both normal velocities are prescribed with the same M2 factor. We leave
# target_transport unset because the present Oceananigans API evaluates it as
# a function of grid only, not time. The global open-boundary correction then
# removes only roundoff/discretization residual from the balanced pair.
u_west_bc = NormalFlowBoundaryCondition(u_inflow_profile;
                                        scheme = PerturbationAdvection())
east_momentum_radiation =
    PerturbationAdvection(gravity_wave_speed = EAST_GRAVITY_WAVE_SPEED,
                          inflow_timescale = EAST_SPONGE_TIMESCALE,
                          outflow_timescale = Inf)
u_east_bc = NormalFlowBoundaryCondition(u_east_outflow;
                                        scheme = east_momentum_radiation)
v_east_bc = ValueBoundaryCondition(0.0;
                                   scheme = east_momentum_radiation)
w_east_bc = ValueBoundaryCondition(0.0;
                                   scheme = east_momentum_radiation)
# Permit local alongshore inflow and outflow while constraining the integrated
# transport across each alongshore boundary to zero. Thus the north and south
# boundaries radiate velocity perturbations without becoming compensation
# outlets for the river inflow.
v_south_bc = NormalFlowBoundaryCondition(0.0;
                                         scheme = PerturbationAdvection(target_transport = 0.0))
v_north_bc = NormalFlowBoundaryCondition(0.0;
                                         scheme = PerturbationAdvection(target_transport = 0.0))
S_west_bc = ValueBoundaryCondition(S_inflow_profile)
# Radiate salinity anomalies out of the eastern, southern, and northern
# boundaries while relaxing any exterior inflow toward ambient shelf salinity.
S_east_bc = ValueBoundaryCondition(ambient_salinity;
                                   scheme = PerturbationAdvection())
S_south_bc = ValueBoundaryCondition(ambient_salinity;
                                    scheme = PerturbationAdvection())
S_north_bc = ValueBoundaryCondition(ambient_salinity;
                                    scheme = PerturbationAdvection())
c_santee_west_bc = ValueBoundaryCondition(c_santee_inflow_profile)
c_winyah_west_bc = ValueBoundaryCondition(c_winyah_inflow_profile)
# The prognostic dye variables use the same numerical range as salinity so
# that one bounds-preserving WENO scheme can constrain all three tracers.
# These exterior values correspond to zero normalized river-water fraction.
dye_open_bc = ValueBoundaryCondition(tracer_lower_bound;
                                     scheme = PerturbationAdvection())

quadratic_drag = BulkDrag(coefficient = Cd)
no_slip_bc = ValueBoundaryCondition(0)
u_wind_bc = FluxBoundaryCondition(u_wind_stress)
v_wind_bc = FluxBoundaryCondition(v_wind_stress)

u_bcs = FieldBoundaryConditions(immersed = quadratic_drag,
                                bottom = quadratic_drag,
                                top = u_wind_bc,
                                west = u_west_bc,
                                east = u_east_bc)
v_bcs = FieldBoundaryConditions(immersed = quadratic_drag,
                                bottom = quadratic_drag,
                                top = v_wind_bc,
                                west = no_slip_bc,
                                east = v_east_bc,
                                south = v_south_bc,
                                north = v_north_bc)
w_bcs = FieldBoundaryConditions(west = no_slip_bc,
                                east = w_east_bc)
S_bcs = FieldBoundaryConditions(west = S_west_bc,
                                east = S_east_bc,
                                south = S_south_bc,
                                north = S_north_bc)
c_santee_bcs = FieldBoundaryConditions(west = c_santee_west_bc,
                                       east = dye_open_bc,
                                       south = dye_open_bc,
                                       north = dye_open_bc)
c_winyah_bcs = FieldBoundaryConditions(west = c_winyah_west_bc,
                                       east = dye_open_bc,
                                       south = dye_open_bc,
                                       north = dye_open_bc)

boundary_conditions = (u = u_bcs, v = v_bcs, w = w_bcs,
                       S = S_bcs, c_santee = c_santee_bcs,
                       c_winyah = c_winyah_bcs)

u_sponge(x, y, z, t, u) =
    -EAST_SPONGE_RATE * east_velocity_sponge_mask(x) *
    (u - u_east_outflow(y, z, t))
v_sponge(x, y, z, t, v) =
    -max(sponge_rate * north_south_sponge_mask(x, y),
         EAST_TANGENTIAL_SPONGE_FACTOR * EAST_SPONGE_RATE *
         east_velocity_sponge_mask(x)) * v
w_sponge(x, y, z, t, w) =
    -EAST_SPONGE_RATE * east_velocity_sponge_mask(x) * w
S_sponge(x, y, z, t, S) =
    -sponge_rate * north_south_sponge_mask(x, y) * (S - ambient_salinity)
c_santee_sponge(x, y, z, t, c) =
    -sponge_rate * north_south_sponge_mask(x, y) *
    (c - tracer_lower_bound)
c_winyah_sponge(x, y, z, t, c) =
    -sponge_rate * north_south_sponge_mask(x, y) *
    (c - tracer_lower_bound)

forcing = (u = Forcing(u_sponge, field_dependencies = :u),
           v = Forcing(v_sponge, field_dependencies = :v),
           w = Forcing(w_sponge, field_dependencies = :w),
           S = Forcing(S_sponge, field_dependencies = :S),
           c_santee = Forcing(c_santee_sponge, field_dependencies = :c_santee),
           c_winyah = Forcing(c_winyah_sponge, field_dependencies = :c_winyah))

equation_of_state =
    LinearEquationOfState(thermal_expansion = 0.0,
                          haline_contraction = haline_contraction)
seawater_buoyancy =
    SeawaterBuoyancy(; gravitational_acceleration,
                     equation_of_state,
                     constant_temperature)

closure =
    Oceananigans.TurbulenceClosures.ModifiedAnisotropicMinimumDissipation()
pressure_solver = ConjugateGradientPoissonSolver(grid)
coriolis = FPlane(f = f₀)

model = NonhydrostaticModel(grid;
                            pressure_solver,
                            advection = WENO(order = 5,
                                             bounds = (tracer_lower_bound,
                                                       tracer_upper_bound)),
                            tracers = (:S, :c_santee, :c_winyah),
                            coriolis,
                            closure,
                            forcing,
                            buoyancy = seawater_buoyancy,
                            boundary_conditions)
set!(model,
     S = ambient_salinity,
     c_santee = tracer_lower_bound,
     c_winyah = tracer_lower_bound)

####
#### Output directory and setup summary
####

function duration_suffix(seconds)
    seconds >= 3600 &&
        return string(number_label(seconds / 3600), "h")
    return string(Int(round(seconds)), "s")
end

function number_label(value)
    return replace(string(round(value; digits = 2)),
                   "-" => "m", "." => "p")
end

filename = string("SalinityTides",
                  "_Qs", Int(round(santee_freshwater_discharge)),
                  "_Qw", Int(round(winyah_freshwater_discharge)),
                  "_dx", number_label(Δx),
                  "_dz", number_label(Δz),
                  "_xe", number_label(x₁ / 1e3), "km",
                  "_es", number_label(EAST_SPONGE_WIDTH / 1e3), "km",
                  "_cg", number_label(EAST_GRAVITY_WAVE_SPEED),
                  "_et", number_label(EAST_TANGENTIAL_SPONGE_FACTOR),
                  "_tau", WIND_DIRECTION, number_label(WIND_STRESS_MAGNITUDE),
                  "_delay", number_label(BUOYANCY_START_TIME / M2_PERIOD), "M2",
                  "_T", duration_suffix(STOP_TIME))
FILE_DIR = isempty(args["output-dir"]) ?
           joinpath(args["output-root"], filename) :
           abspath(args["output-dir"])
mkpath(FILE_DIR)

function save_setup_summary()
    x_m = collect(xnodes(model.grid, Center()))
    y_m = collect(ynodes(model.grid, Center()))
    x_km = x_m ./ 1e3
    y_km = y_m ./ 1e3
    depth = [water_depth(x, y) for x in x_m, y in y_m]

    times = range(0, 2M2_PERIOD; length = 400)
    q_santee = [santee_volume_transport(t) for t in times]
    q_winyah = [winyah_volume_transport(t) for t in times]
    z_profile = range(-inlet_depth, 0; length = 200)
    S_profile = river_salinity_profile.(z_profile)

    fig = Figure(size = (1500, 1050), fontsize = 18)
    ax_depth = Axis(fig[1, 1], title = "Complete rotated bathymetry",
                    xlabel = "Eastward x (km)", ylabel = "Northward y (km)",
                    aspect = DataAspect())
    hm = heatmap!(ax_depth, x_km, y_km, depth;
                  colormap = :deep, colorrange = (0, slope_depth))
    Colorbar(fig[1, 2], hm; label = "Water depth (m)")

    ax_profile = Axis(fig[1, 3],
                      title = "River salinity after first M2 cycle",
                      xlabel = "Salinity", ylabel = "z (m)")
    lines!(ax_profile, S_profile, z_profile; linewidth = 3)
    vlines!(ax_profile, [ambient_salinity]; color = :black,
            linestyle = :dash, label = "Ambient")
    axislegend(ax_profile; position = :lb)

    ax_tide = Axis(fig[2, 1:3],
                   title = "M2-modulated brackish-water transport",
                   xlabel = "Time (hour)", ylabel = "Q (m³ s⁻¹)")
    lines!(ax_tide, times ./ 3600, q_santee;
           linewidth = 3, label = "Santee")
    lines!(ax_tide, times ./ 3600, q_winyah;
           linewidth = 3, label = "Winyah")
    vlines!(ax_tide, [BUOYANCY_START_TIME / 3600];
            color = :black, linestyle = :dash,
            label = "River salinity activated")
    axislegend(ax_tide; position = :rt)

    output_file = joinpath(FILE_DIR, "initial_salinity_tide_setup.png")
    save(output_file, fig)
    @info "Saved salinity-tide setup summary" output_file
    return nothing
end

args["skip-setup-plot"] || save_setup_summary()
if args["setup-only"]
    @info "Salinity-tide setup complete" FILE_DIR
    exit()
end

####
#### Simulation, diagnostics, and output
####

simulation = Simulation(model; Δt = 0.5, stop_time = STOP_TIME)
simulation.callbacks[:wizard] =
    # Bounds-preserving WENO5 requires an advective Courant number no larger
    # than 5/18 ≈ 0.278. Use 0.25 so that the salinity maximum principle is
    # respected without applying a non-conservative post-step clamp.
    Callback(TimeStepWizard(cfl = 0.25,
                            diffusive_cfl = 0.25,
                            max_change = 1.05,
                            max_Δt = 10.0),
             IterationInterval(1))

u, v, w = model.velocities
S = model.tracers.S
c_santee = model.tracers.c_santee
c_winyah = model.tracers.c_winyah
c_santee_output = (c_santee - tracer_lower_bound) / tracer_span
c_winyah_output = (c_winyah - tracer_lower_bound) / tracer_span
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
    t = sim.model.clock.time
    S_extrema = extrema(sim.model.tracers.S)
    c_santee_extrema = extrema(sim.model.tracers.c_santee)
    c_winyah_extrema = extrema(sim.model.tracers.c_winyah)
    c_santee_fraction_extrema =
        ((c_santee_extrema[1] - tracer_lower_bound) / tracer_span,
         (c_santee_extrema[2] - tracer_lower_bound) / tracer_span)
    c_winyah_fraction_extrema =
        ((c_winyah_extrema[1] - tracer_lower_bound) / tracer_span,
         (c_winyah_extrema[2] - tracer_lower_bound) / tracer_span)
    # Bounds-preserving WENO constrains advection, while MAMD diffusion,
    # sponge forcing, and the small discrete divergence residual can introduce
    # roundoff-scale departures. Warn only for materially significant
    # violations; do not clamp, so tracer conservation remains intact.
    bounds_tolerance = 1e-4
    within_bounds(extrema) =
        extrema[1] >= tracer_lower_bound - bounds_tolerance &&
        extrema[2] <= tracer_upper_bound + bounds_tolerance

    if !(within_bounds(S_extrema) &&
         within_bounds(c_santee_extrema) &&
         within_bounds(c_winyah_extrema))
        @warn "Tracer boundedness violation" t S_extrema c_santee_extrema c_winyah_extrema
    end

    println("i: ", iteration(sim),
            ", t: ", prettytime(sim), " / ", prettytime(sim.stop_time),
            ", wall Δt: ", prettytime(elapsed),
            ", model Δt: ", prettytime(sim.Δt),
            ", M2 factor: ", round(tidal_discharge_factor(t); digits = 3),
            ", Q total: ", round(total_volume_transport(t); digits = 2),
            ", wind ramp: ", round(wind_ramp(t); digits = 3),
            ", river salt active: ", river_salinity_is_active(t),
            ", min/max S: ", S_extrema,
            ", min/max c_santee: ", c_santee_fraction_extrema,
            ", min/max c_winyah: ", c_winyah_fraction_extrema,
            ", max |u|: ", maximum(abs, sim.model.velocities.u),
            ", max |v|: ", maximum(abs, sim.model.velocities.v),
            ", max |w|: ", maximum(abs, sim.model.velocities.w),
            ", max div: ", maximum(abs, divergence))
    wall_clock[] = now
    return nothing
end

simulation.callbacks[:progress] =
    Callback(progress, IterationInterval(args["progress-interval"]))

simulation.output_writers[:jld2] =
    JLD2Writer(model, (; u, v, w, S,
                       c_santee = c_santee_output,
                       c_winyah = c_winyah_output);
               filename = joinpath(FILE_DIR, "instantaneous_fields.jld2"),
               schedule = TimeInterval(args["output-interval"]),
               with_halos = !args["output-without-halos"],
               overwrite_existing = PICKUP === false)

simulation.output_writers[:checkpointer] =
    Checkpointer(model;
                 dir = FILE_DIR,
                 prefix = "checkpoint",
                 schedule = TimeInterval(args["checkpoint-interval"]),
                 overwrite_existing = true)

run!(simulation; pickup = PICKUP, checkpoint_at_end = true)
@info "Salinity-tide simulation complete" FILE_DIR
