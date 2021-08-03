module DESPOTAlpha

using POMDPs
using BeliefUpdaters
using Parameters
using CPUTime
using ParticleFilters
using D3Trees
using Random
using Printf
using POMDPModelTools
using POMDPSimulators
using LinearAlgebra
using Statistics
using Distributions
using Plots
using Plots.PlotMeasures

using MCTS
using BasicPOMCP
import BasicPOMCP: SolvedFORollout, SolvedFOValue, convert_estimator

export
    DESPOTAlphaSolver,
    DESPOTAlphaPlanner,
    DESPOTAlphaTree,

    default_action,

    IndependentBounds,
    bounds,
    init_bounds,
    bound,
    init_bound,


    FOValue,
    FORollout,

    info_analysis,
    hist_analysis

"""
    DESPOTAlphaSolver(<keyword arguments>)

Each field may be set via keyword argument. The fields that correspond to algorithm
parameters match the definitions in the paper exactly.

# Fields
- `epsilon_0`
- `xi`
- `K`
- `D`
- `lambda`
- `T_max`
- `max_trials`
- `bounds`
- `default_action`
- `rng`
- `random_source`
- `tree_in_info`

Further information can be found in the field docstrings (e.g.
`?DESPOTAlphaSolver.xi`)
"""
@with_kw struct DESPOTAlphaSolver{R<:AbstractRNG} <: Solver
    "The target gap between the upper and the lower bound at the root of the DESPOTAlpha tree."
    epsilon_0::Float64                      = 1e-3

    "The minimum relative gap required for a branch to be expanded."
    xi::Float64                             = 0.95

    "The number of particles used for approximating beliefs"
    K::Int                                  = 100

    "Maximum number of observations per action sequences"
    C::Int                                  = 20

    "If true, pp_alpha_upper will be updated during backup, which is not necessary when the MDP upper bound is used."
    pp_update::Bool                         = false

    "The maximum depth of the belief tree."
    max_depth::Int                          = 90

    "The maximum online planning time per step."
    T_max::Float64                          = 1.0

    "The maximum number of trials of the planner."
    max_trials::Int                         = typemax(Int)

    "A representation for the upper and lower bound on the discounted value (e.g. `IndependentBounds`)."
    bounds::Any                             = IndependentBounds(-1e6, 1e6)

    """A default action to use if algorithm fails to provide an action because of an error.
   
    This can either be an action object, i.e. `default_action=1` if `actiontype(pomdp)==Int` or a function `f(pomdp, b, ex)` where b is the belief and ex is the exception that caused the planner to fail.
    """
    default_action::Any                     = ExceptionRethrow()

    "A random number generator for the internal sampling processes."
    rng::R                                  = MersenneTwister(rand(UInt32))

    "If true, a reprenstation of the constructed DESPOT is returned by POMDPModelTools.action_info."
    tree_in_info::Bool                      = false

    "Issue an warning when the planning time surpass the time limit by `timeout_warning_threshold` times"
    timeout_warning_threshold::Float64     = T_max * 2.0

    "Number of pre-allocated belief nodes"
    num_b::Int                              = 50_000
end

mutable struct DESPOTAlphaTree{S,A,O}
    # belief nodes
    weights::Vector{Vector{Float64}} # stores weights for *belief node*
    alpha::Vector{Vector{Float64}} # stores alpha vectors for each belief node
    children::Vector{UnitRange{Int}} # to children *ba nodes*
    parent::Vector{Int} # maps to the parent *ba node*
    Delta::Vector{Int}
    u::Vector{Float64}
    l::Vector{Float64}
    obs_prob::Vector{Float64}

    # action nodes
    ba_children::Vector{UnitRange{Int}}
    ba_parent::Vector{Int} # maps to parent *belief node*
    ba_u::Vector{Float64}
    ba_l::Vector{Float64}
    ba_action::Vector{A}

    # action sequences
    as_map::Dict{Vector{A}, Int}
    particles::Vector{Vector{S}}
    obs::Vector{Vector{O}}
    r::Vector{Vector{Float64}}
    L::Vector{Matrix{Float64}}
    def_alpha::Vector{Vector{Float64}}
    def_alpha_upper::Vector{Vector{Float64}}
    pp_alpha_upper::Vector{Vector{Float64}}

    root_particles::Vector{S}
    b::Int
    ba::Int
    as::Int
end

mutable struct DESPOTAlphaPlanner{S, A, O, P<:POMDP{S,A,O}, B, OD, RNG<:AbstractRNG} <: Policy
    sol::DESPOTAlphaSolver{RNG}
    pomdp::P
    bounds::B
    discounts::Vector{Float64}
    rng::RNG
    # The following attributes are used to avoid reallocating memory
    obs_ind_dict::Dict{O, Int}
    obs_dists::Vector{OD}
    tree::Union{Nothing, DESPOTAlphaTree{S,A,O}}
end

function DESPOTAlphaPlanner(sol::DESPOTAlphaSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    rng = deepcopy(sol.rng)
    bounds = init_bounds(sol.bounds, pomdp, sol, rng)
    discounts = discount(pomdp) .^[0:(sol.max_depth+1);]

    obs_dists = Vector{Union{typeof(observation(pomdp, first(actions(pomdp)), rand(initialstate(pomdp)))), Nothing}}(undef, sol.K)
    return DESPOTAlphaPlanner(deepcopy(sol), pomdp, bounds, discounts, rng, 
                            Dict{O, Int}(), obs_dists, nothing)
end

solver(p::DESPOTAlphaPlanner) = p.sol

include("bounds.jl")
include("tree.jl")
include("planner.jl")
include("pomdps_glue.jl")
include("visualization.jl")
include("analysis.jl")

end # module
