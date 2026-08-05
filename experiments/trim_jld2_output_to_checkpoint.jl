using Oceananigans
using JLD2

length(ARGS) == 2 ||
    error("Usage: julia trim_jld2_output_to_checkpoint.jl OUTPUT_FILE CHECKPOINT_FILE")

output_file = abspath(ARGS[1])
checkpoint_file = abspath(ARGS[2])

isfile(output_file) || error("Output file does not exist: $output_file")
isfile(checkpoint_file) || error("Checkpoint file does not exist: $checkpoint_file")

checkpoint_time = jldopen(checkpoint_file, "r") do file
    file["simulation/model/clock"].time
end

removed_iterations = String[]

jldopen(output_file, "r+") do file
    timeseries = file["timeseries"]
    time_group = timeseries["t"]

    for iteration in keys(time_group)
        time_group[iteration] > checkpoint_time &&
            push!(removed_iterations, String(iteration))
    end

    sort!(removed_iterations; by = iteration -> parse(Int, iteration))

    for field_name in keys(timeseries)
        group = timeseries[field_name]
        group isa JLD2.Group || continue

        for iteration in removed_iterations
            haskey(group, iteration) && delete!(group, iteration)
        end
    end
end

@info "Trimmed output to checkpoint" output_file checkpoint_file checkpoint_time removed_iterations
