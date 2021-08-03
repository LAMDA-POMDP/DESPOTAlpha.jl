using ProgressMeter

T = 50
N = 50

pomdp = BabyPOMDP()

bds = IndependentBounds(reward(pomdp, true, false)/(1-discount(pomdp)), 0.0)

solver = DESPOTAlphaSolver(epsilon_0=0.1,
                      K=300,
                      max_depth=50,
                      bounds=bds,
                      T_max=Inf,
                      max_trials=5000,
                      rng=MersenneTwister(4)
                     )

rsum = 0.0
fwc_rsum = 0.0
@showprogress for i in 1:N
    planner = solve(solver, pomdp)
    sim = RolloutSimulator(max_steps=T, rng=MersenneTwister(i))
    fwc_sim = deepcopy(sim)
    global rsum += simulate(sim, pomdp, planner)
    global fwc_rsum += simulate(fwc_sim, pomdp, FeedWhenCrying())
end

@show rsum/N
@show fwc_rsum/N
