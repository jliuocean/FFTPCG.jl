using Oceananigans
using Oceanostics
using CairoMakie
using ArgParse
using Statistics
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Grids: inactive_cell

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
        "--output-dir"
            help = "Animation output directory; empty writes beside the input data"
            arg_type = String
            default = ""
        "--frame-stride"
            help = "Use every Nth saved simulation snapshot"
            arg_type = Int
            default = 1
        "--max-frames"
            help = "Maximum number of animation frames; zero uses all selected snapshots"
            arg_type = Int
            default = 0
        "--epv-limit"
            help = "Symmetric EPV color limit (s^-3); zero estimates it from representative frames"
            arg_type = Float64
            default = 0.0
        "--tke-limit"
            help = "Upper resolved-TKE color limit (m^2 s^-2); zero estimates it automatically"
            arg_type = Float64
            default = 0.0
        "--w-limit"
            help = "Symmetric vertical-velocity color limit (m s^-1); zero estimates it automatically"
            arg_type = Float64
            default = 0.0
    end
    return parse_args(settings)
end

args = parse_commandline()
const FILE_DIR = abspath(args["data-dir"])
const DATA_FILE = joinpath(FILE_DIR, "instantaneous_fields.jld2")
const OUTPUT_DIR = isempty(args["output-dir"]) ? FILE_DIR :
                   abspath(args["output-dir"])
const FRAMERATE = args["framerate"]
const FRAME_STRIDE = args["frame-stride"]
const MAX_FRAMES = args["max-frames"]

isfile(DATA_FILE) || error("Data file does not exist: $DATA_FILE")
mkpath(OUTPUT_DIR)
FRAMERATE > 0 || error("--framerate must be positive")
FRAME_STRIDE > 0 || error("--frame-stride must be positive")
MAX_FRAMES >= 0 || error("--max-frames must be nonnegative")
args["epv-limit"] >= 0 || error("--epv-limit must be nonnegative")
args["tke-limit"] >= 0 || error("--tke-limit must be nonnegative")
args["w-limit"] >= 0 || error("--w-limit must be nonnegative")

####
#### Primitive model output and working fields
####

u_data = FieldTimeSeries(DATA_FILE, "u")
v_data = FieldTimeSeries(DATA_FILE, "v")
w_data = FieldTimeSeries(DATA_FILE, "w")
S_data = FieldTimeSeries(DATA_FILE, "S")

Nt = minimum((length(u_data.times), length(v_data.times),
              length(w_data.times), length(S_data.times)))
Nt > 0 || error("No snapshots were found in $DATA_FILE")

grid = S_data.grid
Nx, Ny, Nz = size(grid)

u = XFaceField(grid)
v = YFaceField(grid)
w = ZFaceField(grid)
b = CenterField(grid)

# Oceanostics defines TKE with respect to user-supplied mean velocities.
# Here the mean is the temporal mean over the complete saved record, so the
# result is resolved TKE: 1/2 [(u-Ubar)^2 + (v-Vbar)^2 + (w-Wbar)^2].
U = XFaceField(grid)
V = YFaceField(grid)
W = ZFaceField(grid)

fill!(interior(U), 0)
fill!(interior(V), 0)
fill!(interior(W), 0)

@info "Computing record-mean velocity for resolved TKE" Nt
for n in 1:Nt
    interior(U) .+= Array(interior(u_data[n])) ./ Nt
    interior(V) .+= Array(interior(v_data[n])) ./ Nt
    interior(W) .+= Array(interior(w_data[n])) ./ Nt
end
fill_halo_regions!((U, V, W))

const gravitational_acceleration = 9.81
const haline_contraction = 7.8e-4
const reference_salinity = 34.0
const f₀ = 8e-5

diagnostic_model = (grid = grid,
                    velocities = (u = u, v = v, w = w),
                    tracers = (b = b,),
                    coriolis = FPlane(f = f₀))

# Oceanostics returns EPV at (Face, Face, Face). Interpolate it to cell centers
# so that EPV, resolved TKE, and vertical velocity share the same plotting grid.
epv = Field(@at (Center, Center, Center) ErtelPotentialVorticity(diagnostic_model))
tke = Field(TurbulentKineticEnergy(diagnostic_model; U, V, W))
vertical_velocity = Field(@at (Center, Center, Center) w)

function load_diagnostics!(n)
    interior(u) .= Array(interior(u_data[n]))
    interior(v) .= Array(interior(v_data[n]))
    interior(w) .= Array(interior(w_data[n]))

    # For the linear equation of state used by the simulation,
    # b = -g β (S - Sref), so ∇b = -g β ∇S.
    salinity = Array(interior(S_data[n]))
    interior(b) .= -gravitational_acceleration * haline_contraction .*
                   (salinity .- reference_salinity)

    fill_halo_regions!((u, v, w, b))
    compute!(epv)
    compute!(tke)
    compute!(vertical_velocity)
    return nothing
end

####
#### Masks and diagnostic slices
####

x = collect(xnodes(grid, Center()))
y = collect(ynodes(grid, Center()))
z = collect(znodes(grid, Center()))
x_km, y_km = x ./ 1e3, y ./ 1e3

active = falses(Nx, Ny, Nz)
for k in 1:Nz, j in 1:Ny, i in 1:Nx
    active[i, j, k] = !inactive_cell(i, j, k, grid)
end

# EPV contains spatial derivatives. Exclude physical-domain edges and cells
# adjacent to immersed topography, because no-halo output cannot reconstruct
# the original open-boundary and immersed-boundary derivative stencils there.
epv_valid = falses(Nx, Ny, Nz)
for k in 2:Nz-1, j in 2:Ny-1, i in 2:Nx-1
    epv_valid[i, j, k] =
        active[i, j, k] &&
        active[i-1, j, k] && active[i+1, j, k] &&
        active[i, j-1, k] && active[i, j+1, k] &&
        active[i, j, k-1] && active[i, j, k+1]
end

nearest_index(nodes, value) = argmin(abs.(nodes .- value))
const santee_center_y = -5e3
const winyah_center_y = 5e3
j_santee = nearest_index(y, santee_center_y)
j_winyah = nearest_index(y, winyah_center_y)
k_surface = Nz
k_epv_surface = max(2, Nz - 1)

function masked_surface(field, k, mask)
    values = Array(interior(field, :, :, k))
    values[.!view(mask, :, :, k)] .= NaN
    return values
end

function masked_xz(field, j, mask)
    values = Array(interior(field, :, j, :))
    values[.!view(mask, :, j, :)] .= NaN
    return values
end

function diagnostic_slices(n)
    load_diagnostics!(n)
    return (
        epv_surface = masked_surface(epv, k_epv_surface, epv_valid),
        epv_winyah = masked_xz(epv, j_winyah, epv_valid),
        epv_santee = masked_xz(epv, j_santee, epv_valid),
        tke_surface = masked_surface(tke, k_surface, active),
        tke_winyah = masked_xz(tke, j_winyah, active),
        tke_santee = masked_xz(tke, j_santee, active),
        w_surface = masked_surface(vertical_velocity, k_surface, active),
        w_winyah = masked_xz(vertical_velocity, j_winyah, active),
        w_santee = masked_xz(vertical_velocity, j_santee, active),
    )
end

####
#### Robust color limits
####

function finite_values(values)
    vector = vec(values)
    return vector[isfinite.(vector)]
end

sample_indices = unique(round.(Int, range(1, Nt; length = min(Nt, 7))))
epv_samples = Float64[]
tke_samples = Float64[]
w_samples = Float64[]

@info "Estimating diagnostic color ranges" sample_indices
for n in sample_indices
    slices = diagnostic_slices(n)
    append!(epv_samples, abs.(finite_values(slices.epv_surface)))
    append!(epv_samples, abs.(finite_values(slices.epv_winyah)))
    append!(epv_samples, abs.(finite_values(slices.epv_santee)))
    append!(tke_samples, finite_values(slices.tke_surface))
    append!(tke_samples, finite_values(slices.tke_winyah))
    append!(tke_samples, finite_values(slices.tke_santee))
    append!(w_samples, abs.(finite_values(slices.w_surface)))
    append!(w_samples, abs.(finite_values(slices.w_winyah)))
    append!(w_samples, abs.(finite_values(slices.w_santee)))
end

robust_upper(values, fallback) =
    isempty(values) ? fallback : max(quantile(values, 0.995), eps(Float64))

epv_limit = args["epv-limit"] > 0 ? args["epv-limit"] :
            robust_upper(epv_samples, 1e-7)
tke_limit = args["tke-limit"] > 0 ? args["tke-limit"] :
            robust_upper(tke_samples, 1e-3)
w_limit = args["w-limit"] > 0 ? args["w-limit"] :
          robust_upper(w_samples, 1e-3)

@info "Diagnostic color limits" epv_limit tke_limit w_limit

####
#### EPV, resolved-TKE, and vertical-velocity animation
####

frames = collect(1:FRAME_STRIDE:Nt)
MAX_FRAMES > 0 && (frames = frames[1:min(MAX_FRAMES, length(frames))])
first_slices = diagnostic_slices(first(frames))

epv_surface = Observable(first_slices.epv_surface)
epv_winyah = Observable(first_slices.epv_winyah)
epv_santee = Observable(first_slices.epv_santee)
tke_surface = Observable(first_slices.tke_surface)
tke_winyah = Observable(first_slices.tke_winyah)
tke_santee = Observable(first_slices.tke_santee)
w_surface = Observable(first_slices.w_surface)
w_winyah = Observable(first_slices.w_winyah)
w_santee = Observable(first_slices.w_santee)

fig = Figure(size = (1900, 1550), fontsize = 17)

function surface_axis(row, title)
    return Axis(fig[row, 1], title = title,
                xlabel = "Eastward x (km)",
                ylabel = "Northward y (km)",
                aspect = DataAspect())
end

function transect_axis(row, column, title)
    return Axis(fig[row, column], title = title,
                xlabel = "Eastward x (km)", ylabel = "z (m)")
end

ax_epv_surface = surface_axis(1, "Near-surface Ertel PV, z = $(round(z[k_epv_surface]; digits=1)) m")
ax_epv_winyah = transect_axis(1, 2, "Ertel PV: Winyah centerline")
ax_epv_santee = transect_axis(1, 3, "Ertel PV: Santee centerline")

ax_tke_surface = surface_axis(2, "Surface resolved TKE")
ax_tke_winyah = transect_axis(2, 2, "Resolved TKE: Winyah centerline")
ax_tke_santee = transect_axis(2, 3, "Resolved TKE: Santee centerline")

ax_w_surface = surface_axis(3, "Near-surface vertical velocity")
ax_w_winyah = transect_axis(3, 2, "Vertical velocity: Winyah centerline")
ax_w_santee = transect_axis(3, 3, "Vertical velocity: Santee centerline")

hm_epv = heatmap!(ax_epv_surface, x_km, y_km, epv_surface;
                  colormap = :balance,
                  colorrange = (-epv_limit, epv_limit),
                  nan_color = :lightgray)
heatmap!(ax_epv_winyah, x_km, z, epv_winyah;
         colormap = :balance, colorrange = (-epv_limit, epv_limit),
         nan_color = :lightgray)
heatmap!(ax_epv_santee, x_km, z, epv_santee;
         colormap = :balance, colorrange = (-epv_limit, epv_limit),
         nan_color = :lightgray)

hm_tke = heatmap!(ax_tke_surface, x_km, y_km, tke_surface;
                  colormap = :thermal, colorrange = (0, tke_limit),
                  nan_color = :lightgray)
heatmap!(ax_tke_winyah, x_km, z, tke_winyah;
         colormap = :thermal, colorrange = (0, tke_limit),
         nan_color = :lightgray)
heatmap!(ax_tke_santee, x_km, z, tke_santee;
         colormap = :thermal, colorrange = (0, tke_limit),
         nan_color = :lightgray)

hm_w = heatmap!(ax_w_surface, x_km, y_km, w_surface;
                colormap = :balance, colorrange = (-w_limit, w_limit),
                nan_color = :lightgray)
heatmap!(ax_w_winyah, x_km, z, w_winyah;
         colormap = :balance, colorrange = (-w_limit, w_limit),
         nan_color = :lightgray)
heatmap!(ax_w_santee, x_km, z, w_santee;
         colormap = :balance, colorrange = (-w_limit, w_limit),
         nan_color = :lightgray)

Colorbar(fig[1, 4], hm_epv; label = "Ertel PV (s⁻³)")
Colorbar(fig[2, 4], hm_tke; label = "Resolved TKE (m² s⁻²)")
Colorbar(fig[3, 4], hm_w; label = "w (m s⁻¹)")

time_label = Observable("t = $(round(S_data.times[first(frames)] / 3600; digits = 1)) hour")
Label(fig[0, :], time_label; fontsize = 22)

output_file = joinpath(OUTPUT_DIR, "ertel_pv_tke_vertical_velocity.mp4")
CairoMakie.record(fig, output_file, frames; framerate = FRAMERATE) do n
    slices = diagnostic_slices(n)
    epv_surface[] = slices.epv_surface
    epv_winyah[] = slices.epv_winyah
    epv_santee[] = slices.epv_santee
    tke_surface[] = slices.tke_surface
    tke_winyah[] = slices.tke_winyah
    tke_santee[] = slices.tke_santee
    w_surface[] = slices.w_surface
    w_winyah[] = slices.w_winyah
    w_santee[] = slices.w_santee
    time_label[] = "t = $(round(S_data.times[n] / 3600; digits = 1)) hour"
end

@info "Saved Oceanostics diagnostics animation" output_file
