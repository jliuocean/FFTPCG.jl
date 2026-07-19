# Run the complete 40 km jetty experiment with river transport compensated
# through the south boundary rather than the west boundary.
"--jetty-geometry" in ARGS || push!(ARGS, "--jetty-geometry")
"--south-outflow" in ARGS || push!(ARGS, "--south-outflow")
include(joinpath(@__DIR__, "buoyancy_flow_mamd_smooth_H200.jl"))
