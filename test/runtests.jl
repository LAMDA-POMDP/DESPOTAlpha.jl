using DESPOTAlpha
using Test

using POMDPs
using POMDPModels
using POMDPSimulators
using Random
using POMDPModelTools
using ParticleFilters
using BeliefUpdaters
using POMDPPolicies
using Plots
theme(:mute)

# include("baby_sanity_check.jl")

pomdp = BabyPOMDP()
pomdp.discount = 1.0
p = solve(DESPOTAlphaSolver(tree_in_info=true), pomdp)
Random.seed!(p, 1)
b0 = initialstate(pomdp)
a, info = action_info(p, b0)
info_analysis(info)

# Light Dark Test
# pomdp = LightDark1D()
# random = solve(RandomSolver(), pomdp)
# bds = IndependentBounds(FORollout(random), pomdp.correct_r)
# solver = DESPOTAlphaSolver(bounds=bds,
#                     K=100,
#                     C=20,
#                     tree_in_info=true
#                     )
# planner = solve(solver, pomdp)
# a, info = action_info(planner, initialstate(pomdp))
# info_analysis(info)
# hr = HistoryRecorder(max_steps=50)
# @time hist = simulate(hr, pomdp, planner)
# # hist_analysis(hist)
# println("Discounted reward is $(discounted_reward(hist))")

# BabyPOMDP Test
# Type stability
pomdp = BabyPOMDP()
bds = (pomdp, b)->(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = DESPOTAlphaSolver(bounds=bds,
                      rng=MersenneTwister(4),
                      K=100,
                      tree_in_info=true
                     )
p = solve(solver, pomdp)

b0 = initialstate(pomdp)
D, Depth = @inferred DESPOTAlpha.build_tree(p, b0)
@inferred action_info(p, b0)
a, info = action_info(p, b0)
info_analysis(info)
@inferred DESPOTAlpha.explore!(D, 1, Bool[], p)
@inferred DESPOTAlpha.expand!(D, D.b-1, DESPOTAlpha.retrieve_as(D, D.b-1), p)
@inferred DESPOTAlpha.backup!(D, D.b-1, DESPOTAlpha.retrieve_as(D, D.b-1), p)
@inferred DESPOTAlpha.next_best(D, 1, p)
@inferred DESPOTAlpha.excess_uncertainty(D, 1, DESPOTAlpha.solver(p).xi, p)
@inferred action(p, b0)

pomdp = BabyPOMDP()

# constant bounds
bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)
solver = DESPOTAlphaSolver(bounds=bds, K=200, tree_in_info=true, num_b=10_000)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=20)
@time hist = simulate(hr, pomdp, planner)
# hist_analysis(hist)
println("Discounted reward is $(discounted_reward(hist))")

# FO policy lower bound
bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), FORollout(FeedWhenCrying()))
solver = DESPOTAlphaSolver(bounds=bds, K=200, tree_in_info=true, num_b=10_000)
planner = solve(solver, pomdp)
hr = HistoryRecorder(max_steps=20)
@time hist = simulate(hr, pomdp, planner)
# hist_analysis(hist)
println("Discounted reward is $(discounted_reward(hist))")

# from README:
using POMDPs, POMDPModels, POMDPSimulators, DESPOTAlpha

pomdp = TigerPOMDP()

solver = DESPOTAlphaSolver(bounds=IndependentBounds(-20.0, 0.0))
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
