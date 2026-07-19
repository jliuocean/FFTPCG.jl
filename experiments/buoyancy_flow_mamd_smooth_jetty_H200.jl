# Run the complete buoyancy-flow experiment with the 40 km jetty/channel
# bathymetry defined in winyah_jetty_bathymetry_H200.jl.
"--jetty-geometry" in ARGS || push!(ARGS, "--jetty-geometry")
include(joinpath(@__DIR__, "buoyancy_flow_mamd_smooth_H200.jl"))
