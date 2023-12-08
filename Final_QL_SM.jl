using LinearAlgebra
using CSV
using DataFrames
using Printf
using Distributions

# Structures
mutable struct QLearning
    S # state space
    A # action space
    γ # discount
    Q # action value function
    α # learning rate
end

mutable struct SoftmaxExploration
    𝒜 # action space
    τ # temperature parameter
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

function lookahead(model, s, a)
    γ, Q = model.γ, model.Q
    return Q[s, a]
end
function replace_nan_with_zero(probabilities)
    prob_sum = 0
    for i in eachindex(probabilities)
        if isnan(probabilities[i])
            probabilities[i] = 0.0
        else
            prob_sum += probabilities[i]
        end
    end
    return probabilities, prob_sum
end

function no_inf(probabilities)
    has_inf = false
    for i in eachindex(probabilities)
        if isinf(probabilities[i])
            has_inf = true
            probabilities .= 0
            probabilities[i] = 1
            break
        end
    end
    return probabilities
end

function softmax_policy(Q, state, action_space, τ)
    #println(Q[state,:])
    probabilities = exp.(Q[state, :] / τ)
    #println(probabilities)
    #probabilities, prob_sum = replace_nan_with_zero(probabilities)
    #probabilities /= prob_sum
    probabilities = no_inf(probabilities)
    probabilities /=sum(probabilities)
    
    #println(probabilities)
    # Use Categorical distribution to sample an action
    action_distribution = Categorical(probabilities)
    return rand(action_distribution, 1)[1]
end

function (π::SoftmaxExploration)(model, s)
    𝒜, τ = π.𝒜, π.τ
    return softmax_policy(model.Q, s, 𝒜, τ)
end

function update!(model::QLearning, s, a, r, sp, π)
    γ, Q, α = model.γ, model.Q, model.α
    current_action = π(model, s)
    Q[s, current_action] += α * (r + γ * maximum(Q[sp, :]) - Q[s, current_action])
    return model
end

function write_policy(policy, filename)
    open(filename, "w") do f
        for i in 1:length(policy)
            println(f, Int(policy[i]))
        end
    end
end

function compute(infile, outfile, num_states, π, max_q_iter)
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
        while   q_iter < max_q_iter
            for i = 1:length(s)
                update!(model, s[i], a[i], r[i], sp[i], π)
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
            policy[i] = π(model, i)
        end
        # Write policy to the output file
        write_policy(policy, outfile)
    end
end

# Final Compute
if length(ARGS) != 5
    error("usage: julia CODE.jl <infile>.csv <outfile>.policy <num_states> <temperature>, <max_q_iter>")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]
num_states = parse(Int, ARGS[3])
temperature = parse(Float64, ARGS[4])
max_q_iter = parse(Int, ARGS[5])
exploration_policy = SoftmaxExploration(Set(1:num_states), temperature)

println("Compute Start")
compute(inputfilename, outputfilename, num_states, exploration_policy, max_q_iter)
println("Compute End")
