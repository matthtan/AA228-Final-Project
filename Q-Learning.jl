using LinearAlgebra
using CSV
using DataFrames
using Printf

# Structures
mutable struct QLearning
    S #state space
    A #action space
    γ # discount
    Q #action value function
    α # learning rate
end

# WRITE DATA FUNCTION

function write_policy(policy, filename)
    open(filename, "w") do data
        for data = 1:length(policy)
            @printf(data, "i\n", policy[state])
        end
    end
end

# Functions
function data_upload(infile)
    # Read the CSV file into a DataFrame
    df = CSV.read(infile, DataFrame)
    
    # Extract columns into separate variables
    s = df[:, :s]
    a = df[:, :a]
    r = df[:, :r]
    sp = df[:, :sp]
    # Return variables
    return s, a, r, sp
end

function update!(model::QLearning, s, a, r, sp) 
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α * (r + γ*maximum(Q[sp,:]) - Q[s,a])
    return model
end
println("Fcn End")
# Output Functions
println("Output Start")
function write_policy(policy, filename)
    # creates a <filename>.policy file with the input policy
    open(filename, "w") do f
        for i in 1:length(policy)
            println(f, Int(policy[i]))
        end
    end
end

function compute(infile, outfile, num_states)
    @time begin
        # Discount factor
        γ = 0.95
        # Learning rate
        α = 0.1

        # Read the data
        s, a, r, sp = data_upload(infile)
        # Initialize Q-learning model
        S = Set(s)
        A = Set(a)
        Q = zeros(length(1:num_states), length(A))
        model = QLearning(S, A, γ, Q, α)
        q = copy(model.Q)
        # Iterate Through and Update Q until it converges
        q_iter = 0
        while true
            for i = 1:length(s)
                update!(model, s[i], a[i], r[i], sp[i])
            end
            q_iter += 1
            if norm(model.Q - q) < 1e-15
                break
            end
            q = copy(model.Q)
        end
        println("Q iter loop count:", q_iter)
        # Initialize policy with zero actions for unexplored states        
        policy = zeros(length(model.S))
        # Update Q-values and policy
        for i = 1:length(policy)
            policy[i] = argmax(a -> model.Q[i,a], model.A)
        end
        # Write policy to the output file
        write_policy(policy, outfile)
    end
end

# Final Compute
if length(ARGS) != 3
    error("usage: julia CODE.jl <infile>.csv <outfile>.policy <num_states>")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]
num_states = parse(Int, ARGS[3])
println("Compute Start")
compute(inputfilename, outputfilename, num_states)
println("Compute End")