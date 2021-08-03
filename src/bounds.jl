"""
    Dependent Bounds

Specify lower and upper bounds that are not independent.
###
It can take as input a function, `f(pomdp, belief)`, that returns both lower and upper bounds.
It can take as input any object for which a `bounds` function is implemented

"""

init_bounds(bds, pomdp, sol, rng) = bds

# Used to initialize lower and upper bounds with a single function
bounds(f::Function, pomdp::P, s::S, max_depth::Int) where {S,P<:POMDP{S}} = f(pomdp, s)

# Used to initialize both the lower and upper bound with an object for which a `bounds` function is implemented
function bounds!(L::Vector{Float64}, U::Vector{Float64}, bd::B, pomdp::P, particles::Vector{S}, max_depth::Int) where {S,P<:POMDP{S},B}
    @inbounds for (i, s) in enumerate(particles)
        if isterminal(pomdp, s)
            L[i], U[i] = 0.0, 0.0
        else
            L[i], U[i] = bounds(bd, pomdp, s, max_depth)
        end
    end
    return L, U
end


"""
    IndependentBounds(lower, upper)

Specify lower and upper bounds that are independent of each other (the most common case).
A lower or upper bound can be a Number, a Function, `f(pomdp, belief)`, that returns a bound, an object for which a `bound` function 
is implemented. Specifically, for FOValue, POValue, FORollout, SemiPORollout, and PORollout, a `bound` function is already implemented.
You can also implement a `bound!` function to initialize sibling beliefs simultaneously, if it could provide a further performace gain.
"""

mutable struct IndependentBounds{L, U}
    lower::L
    upper::U
end

function IndependentBounds(l, u)
    return IndependentBounds(l, u)
end

function init_bounds(bds::IndependentBounds, pomdp::POMDP, sol::DESPOTAlphaSolver, rng::AbstractRNG)
    return IndependentBounds(convert_estimator(bds.lower, sol, pomdp),
                             convert_estimator(bds.upper, sol, pomdp)
                             )
end

function bounds!(L::Vector{Float64}, U::Vector{Float64}, bds::IndependentBounds{LB,UB}, pomdp::P, particles::Vector{S}, max_depth::Int) where {LB,UB,S,P<:POMDP{S}}
    bound!(L, bds.lower, pomdp, particles, max_depth)
    bound!(U, bds.upper, pomdp, particles, max_depth)
    for (i, s) in enumerate(particles)
        if isterminal(pomdp, s)
            L[i], U[i] = 0.0, 0.0
        end
    end
    return L, U
end

# Used when the lower or upper bound is a fixed number
bound!(V::Vector{Float64}, n::Float64, pomdp::P, particles::Vector{S}, max_depth::Int) where {S,P<:POMDP{S}} = fill!(V, n)
bound!(V::Vector{Float64}, n::Int, pomdp::P, particles::Vector{S}, max_depth::Int) where {S,P<:POMDP{S}} = fill!(V, convert(Float64, n))

# Used when the lower or upper bound is an object for which a `bound` function is implemented
function bound!(V::Vector{Float64}, bd::B, pomdp::P, particles::Vector{S}, max_depth::Int) where {S,P<:POMDP{S},B}
    @inbounds for s in particles
        V[i] = bound(bd, pomdp, s, max_depth)
    end
    return V
end

function bound!(V::Vector{Float64}, bd::SolvedFORollout, pomdp::M, particles::Vector{S}, max_depth::Int) where {S,M<:POMDP{S}}
    sim = RolloutSimulator(bd.rng, max_depth)
    broadcast!((s)->simulate(sim, pomdp, bd.policy, s), V, particles)
end

function bound!(V::Vector{Float64}, bd::SolvedFOValue{P}, pomdp::M, particles::Vector{S}, max_depth::Int) where {P,S,M<:POMDP{S}}
    broadcast!((s)->value(bd.policy, s), V, particles)
end