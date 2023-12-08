# AA228 Final Project Policy Visualizer
using Plots
using CSV
using DataFrames
using Statistics
# Load Policy from CSV File
function load_policy(file_path::AbstractString)
    df = CSV.read(joinpath(@__DIR__, file_path), DataFrame)
    # Extract full columns with header ":x" into vectors
    policy = df[:, :P]
    return policy
end
# Load Data from CSV File
function load_data(file_path::AbstractString)
    df = CSV.read(joinpath(@__DIR__, file_path), DataFrame)
    # Extract full columns with header ":x" into vectors
    #s = df[:, :s]
    #a = df[:, :a]
    r = df[:, :r]
    sp = df[:, :sp]
    return r, sp
end
# Generate arrow directions for policy plotting
function arrow_direction(value)
    if value == 1
        return (1, 0)  # Arrow pointing right
    elseif value == 2
        return (0, 1) # Arrow pointing up
    elseif value == 3
        return (-1, 0) # Arrow pointing left
    elseif value == 4
        return (0, -1)  # Arrow pointing down
    else
        return (0, 0)  # No arrow for other values
    end
end
# Plot 
function visualize(policy_matrix, reward_matrix, areaSize)
    world_rows, world_cols = size(reward_matrix)
    plot(aspect_ratio=:equal, ylimits=(0,world_rows+0.5), xlimits=(0,world_cols+0.5), title="Rewards and Policy Visualization")
    
    # Rewards plotting
    heatmap!(transpose(reward_matrix))
    # Policy plotting
    rows, cols = size(policy_matrix)
    x = 1:cols
    y = repeat(1:cols, inner = cols)
    u = zeros(0)
    v = zeros(0)
    for i in 1:rows
        for j in 1:cols
            val = policy_matrix[i,j]
            dir = arrow_direction(val)
            println(i, ",", j, ':', dir)
            append!(u, dir[1]/2)
            append!(v, dir[2]/2)
        end
    end
    quiver!(x, y, quiver=(u,v), color=:green)
    # Policy plotting averages
    # rows = Int(world_rows/areaSize)
    # cols = Int(world_cols/areaSize)
    # println("Average Policy Matrix Size")
    # println(rows)
    # println(cols)
    # avg_policy_matrix = zeros(rows, cols)
    # # build an averaged policy matrix
    # for k in 1:areaSize:(world_rows-areaSize+1)
    #     for m in 1:areaSize:(world_cols-areaSize+1)
    #         section = policy_matrix[k:k+(areaSize-1), m:m+(areaSize-1)]
    #         avg_val = mean(section)
    #         avg_policy_matrix[Int((k-1)/areaSize) + 1, Int((m-1)/areaSize)+1] = round(avg_val)
    #     end
    # end
    # x = 1:areaSize:world_cols
    # y = repeat(x, inner = length(x))
    # u = zeros(0)
    # v = zeros(0)
    # for i in 1:rows
    #     for j in 1:cols
    #         val = avg_policy_matrix[i,j]
    #         dir = arrow_direction(val)
    #         println(j, ",", i, ": ", dir)
    #         append!(u, dir[1]/2)
    #         append!(v, dir[2]/2)
    #     end
    # end
    # quiver!(x, y, quiver=(u,v), color=:green)
    savefig("test.png")
end
# Driving function
function compute(policyfile, datafile, world_height, world_width, areaSize)
    policy = load_policy(policyfile)
    rewards, nextstates = load_data(datafile)
    println(policy)
    policy_matrix = transpose(reshape(policy, (world_height, world_width)))
    println(policy_matrix)
    reward_array = zeros(world_height*world_width)
    for i in 1:length(nextstates)
        reward_array[nextstates[i]] = rewards[i]
    end
    reward_matrix = reshape(reward_array, (world_height, world_width))
    visualize(policy_matrix, reward_matrix, areaSize)
end
# Run
if length(ARGS) != 5
    error("usage: julia CODE.jl <policyfile>.csv <datafile>.csv <world_height> <world_width>")
end
policyfilename = ARGS[1]
datafilename = ARGS[2]
world_height = ARGS[3]
world_width = ARGS[4]
areaSize = parse(Int, ARGS[5]) # size/number of states in the square area that the policy is averaged over
height = parse(Int, world_height)
width = parse(Int, world_width)
compute(policyfilename, datafilename, height, width, areaSize)
