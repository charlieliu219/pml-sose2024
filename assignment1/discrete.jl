# A distribution over the discrete values 1...n
#
# 2024 by Ralf Herbrich
# Hasso-Plattner Institute

module DiscreteDistribution

export Discrete, ℙ, *, /, logsumexp

"""
    The struct containing the parameters of the distributions.

    Should contain exactly one variable:
    logP, which stores a vector of LOG PROBABILITIES as floats
"""
struct Discrete{T} 
    logP::Vector{Float64}
end

"""
    Discrete(logP::Vector{Float64})

Create a discrete distribution from a vector of log probabilities. Note that this function
stronlgy types the discrete distribution to the number of elements. (This means the type assignment in the constructor {} should be based on the property length of logP)

# Examples

```jldoctest
julia> Discrete([0.0, 0.0])
Discrete{2}([0.0, 0.0])

julia> Discrete([0.0, 0.0, 1.0])
Discrete{3}([0.0, 0.0, 1.0])
```
""" 
Discrete(logP::Vector{Float64}) = Discrete{length(logP)}(logP)

"""
    Discrete(n::Int64)

Initialize a uniform discrete distribution for n elements (all probabilities being equal for each of the n elements).

Please not that this and the following examples use the show function at the end of the file to display,
which in turn, uses the helper functions to convert the logprobs to real probabilities.
# Examples

```jldoctest
julia> Discrete(2)
 P = [0.5, 0.5]

julia> Discrete(3)
 P = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
```
""" 
function Discrete(n::Int64)
    probs = zeros(n) 
    return Discrete(probs)
end

"""
    *(p::Discrete{T}, q::Float64) -> Discrete{T}

Multiplies two discrete distributions and returns the new discrete distribution.

# Examples

```jldoctest
julia> Discrete([0.0, 2.0]) * Discrete([1.0, 0.0])
 P = [0.26894142136999505, 0.7310585786300048]

julia> Discrete([0.0, 2.0, -1.0]) * Discrete([1.0, 0.0, 1.0])
 P = [0.2447284710547977, 0.665240955774822, 0.09003057317038048]
```
""" 
function Base.:*(p::Discrete{T}, q::Discrete{T})::Discrete{T} where {T}
    length(p.logP) == length(q.logP) || throw(ArgumentError("Lengths of distributions do not match"))
    
    sum_logP = Vector{Float64}()

    for i in eachindex(p.logP)
        logP_p = p.logP[i]
        logP_q = q.logP[i]
        push!(sum_logP, logP_p + logP_q)
    end

    return Discrete{T}(sum_logP)
end

"""
    /(p::Discrete{T}, q::Discrete{T}) -> Discrete{T}

Divides two discrete distributions and returns the new discrete distribution.

# Examples

```jldoctest
julia> Discrete([1.0, 2.0]) / Discrete([1.0, 0.0])
 P = [0.11920292202211753, 0.8807970779778823]

julia> Discrete([2.0, 0.0, -1.0]) / Discrete([1.0, 0.0, 1.0])
 P = [0.7053845126982412, 0.25949646034241913, 0.035119026959339716]
```
""" 
function Base.:/(p::Discrete{T}, q::Discrete{T})::Discrete{T} where {T}
    length(p.logP) == length(q.logP) || throw(ArgumentError("Lengths of distributions do not match"))
    
    diff_logP = Vector{Float64}()

    for i in eachindex(p.logP)
        logP_p = p.logP[i]
        logP_q = q.logP[i]
        push!(diff_logP, logP_p - logP_q)
    end

    return Discrete{T}(diff_logP)
end

"""
    logsumexp(a::Vector{Float64})

Should compute the logsumexp function for the respective vector:
    max(a) + log(exp(a_1-max(a)) + exp(a_2-max(a)) + exp(a_3-max(a)) + ... + exp(a_n-max(a)))

Remember: You can use the . operator to apply functions like exp element wise to each element of an iterable.
"""
function logsumexp(a::Vector{Float64})
    max_a = maximum(a)
    sum_exp_diff = sum(exp.(a .- max_a))
    result = max_a + log(sum_exp_diff)
    
    return result
end

"""
    ℙ(p::Discrete{T}) -> Vector{Float64}

Computes the actual probabilities from the log probabilities.

# Examples

```jldoctest
julia> ℙ(Discrete([0.0, 0.0]))
2-element Vector{Float64}:
 0.5
 0.5
 
 julia> ℙ(Discrete([1000.0, 1000.0]))
 2-element Vector{Float64}:
  0.5
  0.5
 
 julia> ℙ(Discrete([1.0, -1.0]))
 2-element Vector{Float64}:
 0.8807970779778824
 0.11920292202211759
```
""" 
function ℙ(p::Discrete)
    unnormalized_probs = exp.(p.logP)
    normalized_probs = unnormalized_probs / sum(unnormalized_probs)
    
    return normalized_probs
end

"""
    show(io, p::Discrete)

Pretty-prints a discrete distribution
"""
function Base.show(io::IO, p::Discrete)
    probs = ℙ(p)
    print(io, " P = [")
    for i in eachindex(probs)
        print(io, round(probs[i], digits=4))
        if (i < length(probs))
            print(io, ", ")
        end
    end
    print(io, "]")
end

end