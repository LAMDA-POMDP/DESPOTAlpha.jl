function DESPOTAlphaTree(p::DESPOTAlphaPlanner{S,A,O}, b0::RB) where {S,A,O,RB}
    sol = solver(p)
    b0, all_terminal = strip_terminals(b0, p.pomdp)
    if all_terminal
        error("All states in the current belief are terminal.")
    end
    resampled_P = resample(sol.K, b0, p.pomdp, p.rng)
    L = Vector{Float64}(undef, sol.K)
    U = Vector{Float64}(undef, sol.K)
    bounds!(L, U, p.bounds, p.pomdp, resampled_P, sol.max_depth)
    if sol.tree_in_info || p.tree === nothing
        p.tree = DESPOTAlphaTree([fill(1.0/sol.K, sol.K)],
                        [Float64[]],
                        [1:0],
                        [0],
                        [0],
                        [mean(U)],
                        [mean(L)],
                        [1.0],

                        UnitRange{Int64}[],
                        Int[],
                        Float64[],
                        Float64[],
                        A[],

                        Dict{Vector{A}, Int}(),
                        Vector{S}[],
                        Vector{O}[],
                        Vector{Float64}[],
                        Matrix{Float64}[],
                        Vector{Float64}[],
                        Vector{Float64}[],
                        Vector{Float64}[],

                        resampled_P,
                        1,
                        0,
                        0
                    )
        resize_b!(p.tree, sol.num_b, sol.K)
        resize_ba!(p.tree, sol.num_b)
    else
        reset!(p.tree, resampled_P, mean(U), mean(L))
    end
    return p.tree::DESPOTAlphaTree{S,A,O}
end

function reset!(tree::DESPOTAlphaTree{S}, P::Vector{S}, u::Float64, l::Float64) where S
    fill!(tree.children, 1:0)
    fill!(view(tree.obs_prob, 2:length(tree.obs_prob)), 0.0)
    empty!.(tree.obs)
    empty!(tree.as_map)
    tree.u[1] = u
    tree.l[1] = l
    tree.b = 1
    tree.ba = 0
    tree.as = 0
    tree.root_particles = P
    return nothing
end

Base.zero(::Type{AbstractVector{T}}) where T = T[]

function expand!(D::DESPOTAlphaTree{T,A,Z}, b::Int, as::Vector{A}, p::DESPOTAlphaPlanner{T,A,Z}) where {T,A,Z}
    if b == 1
        P = D.root_particles
    else
        P = D.particles[D.as_map[as]]
    end

    sol = solver(p)
    acts = actions(p.pomdp, WeightedParticleBelief(P, D.weights[b]))
    num_a = length(acts)
    K = sol.K
    C = sol.C
    resize_ba!(D, D.ba + num_a)
    resize_b!(D, D.b + (C + 1) * num_a, sol.K)
    D.children[b] = (D.ba + 1):(D.ba + num_a)

    # Initialize action sequences
    asa = push!(copy(as), acts[1])
    if !haskey(D.as_map, asa)
        resize_as!(D, D.as + num_a, K)
        for a in acts
            D.as += 1
            D.as_map[push!(copy(as), a)] = D.as
            S, O, R = propagate_particles(D, P, a, p)
            L = Matrix{Float64}(undef, (length(O)+1, K))
            η_max = 1e-10
            for spi in eachindex(S)
                η = 0.0
                for (oi, o) in enumerate(O)
                    η += L[oi, spi] = pdf(p.obs_dists[spi], o)
                end
                if η > η_max
                    η_max = η
                end
                L[end, spi] = η
            end
            for spi in eachindex(S)
                for (oi, o) in enumerate(O)
                    L[oi, spi] /= η_max
                end
                L[end, spi] = 1.0 - L[end, spi]/η_max
            end
            push!(O, zero(Z))
            D.L[D.as] = L
            bounds!(D.def_alpha[D.as], D.pp_alpha_upper[D.as], p.bounds, p.pomdp, S, sol.max_depth - D.Delta[b])
            D.def_alpha_upper[D.as] .= D.pp_alpha_upper[D.as]
        end
    end

    # Initialize action branches
    for a in acts
        asa[length(asa)] = a
        as_ba = D.as_map[asa]
        D.ba += 1 # increase ba count
        n_obs = length(D.obs[as_ba]) # number of new obs
        fbp = D.b + 1 # first bp
        lbp = D.b + n_obs # last bp
        D.b += n_obs

        # Calculate child beliefs
        W = view(D.weights, fbp:lbp)
        obs_w = view(D.obs_prob, fbp:lbp)
        for oi in 1:n_obs
            for spi in 1:sol.K
                obs_w[oi] += W[oi][spi] = D.L[as_ba][oi, spi] * D.weights[b][spi]
            end
        end
        for oi in 1:n_obs
            W[oi] ./= obs_w[oi] + 1e-10
        end

        # Initialize the new action branch
        D.ba_children[D.ba] = fbp:lbp
        D.ba_parent[D.ba] = b
        D.ba_action[D.ba] = a

        # Initialize new obs branches
        view(D.parent, fbp:lbp) .= D.ba
        view(D.Delta, fbp:lbp) .= D.Delta[b] + 1
        view(D.obs_prob, fbp:lbp) .= obs_w
        broadcast!(x->dot(x, D.def_alpha[as_ba]), view(D.l, fbp:lbp), W)
        broadcast!(x->dot(x, D.def_alpha_upper[as_ba]), view(D.u, fbp:lbp), W)
        for i in fbp:lbp
            D.alpha[i] = D.def_alpha[as_ba]
        end

        # update upper bounds for action selection
        D.ba_u[D.ba] = D.weights[b]' * D.r[as_ba] + discount(p.pomdp) * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[D.ba])
        D.ba_l[D.ba] = D.weights[b]' * D.r[as_ba] + discount(p.pomdp) * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[D.ba])
    end
    # With this update, we need not to check all action branches during backup.
    D.u[b] = maximum(D.ba_u[ba] for ba in D.children[b])
    D.l[b] = maximum(D.ba_l[ba] for ba in D.children[b])
    return nothing
end

POMDPs.pdf(dist::Nothing, o) = 0.0

function strip_terminals(b, m::POMDP)
    return b, false
end

function strip_terminals(b::AbstractParticleBelief, m::POMDP)
    w_sum = 0.0
    P = particles(b)
    W = weights(b)
    @inbounds for (i, s) in enumerate(P)
        if isterminal(m, s)
            W[i] = 0.0
        else
            w_sum += W[i]
        end
    end
    return WeightedParticleBelief(P, W / (w_sum + 1e-10), 1.0), w_sum == 0.0
end

function propagate_particles(D::DESPOTAlphaTree, particles::Vector, a, p::DESPOTAlphaPlanner)
    sol = solver(p)
    S = D.particles[D.as]
    O = D.obs[D.as]
    R = D.r[D.as]
    empty!(p.obs_ind_dict)

    for (i, s) in enumerate(particles)
        if isterminal(p.pomdp, s)
            S[i] = s
            R[i] = 0.0
            p.obs_dists[i] = nothing
        else
            sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, p.rng)
            S[i] = sp
            R[i] = r
            p.obs_dists[i] = observation(p.pomdp, a, sp)
            if !haskey(p.obs_ind_dict, o) && length(O) < sol.C
                push!(O, o)
                p.obs_ind_dict[o] = length(O)
            end
        end
    end
    return S, O, R
end

function resize_b!(D::DESPOTAlphaTree, n::Int, K::Int)
    if n > length(D.weights)
        resize!(D.weights, n)
        resize!(D.alpha, n)
        resize!(D.children, n)
        resize!(D.obs_prob, n)
        @inbounds for i in (length(D.parent)+1):n
            D.weights[i] = Vector{Float64}(undef, K)
            D.children[i] = 1:0
            D.obs_prob[i] = 0.0
        end
        resize!(D.parent, n)
        resize!(D.Delta, n)
        resize!(D.u, n)
        resize!(D.l, n)
    end
    return nothing
end

function resize_ba!(D::DESPOTAlphaTree{S}, n::Int) where S
    if n > length(D.ba_children)
        resize!(D.ba_children, n)
        resize!(D.ba_parent, n)
        resize!(D.ba_u, n)
        resize!(D.ba_l, n)
        resize!(D.ba_action, n)
    end
    return nothing
end

function resize_as!(D::DESPOTAlphaTree{S,A,O}, n::Int, K::Int) where {S,A,O}
    if n > length(D.particles)
        resize!(D.particles, n)
        resize!(D.obs, n)
        resize!(D.r, n)
        resize!(D.def_alpha, n)
        resize!(D.def_alpha_upper, n)
        resize!(D.pp_alpha_upper, n)
        @inbounds for i in (length(D.L)+1):n
            D.particles[i] = Vector{S}(undef, K)
            D.obs[i] = O[]
            D.r[i] = Vector{Float64}(undef, K)
            D.def_alpha[i] = Vector{Float64}(undef, K)
            D.def_alpha_upper[i] = Vector{Float64}(undef, K)
            D.pp_alpha_upper[i] = Vector{Float64}(undef, K)
        end
        resize!(D.L, n)
    end
    return nothing
end

function resample(m::Int, b::AbstractParticleBelief{S}, pomdp::POMDP{S}, rng::AbstractRNG) where S
    step = weight_sum(b)/m
    U = rand(rng)*step
    c = weight(b,1) # accumulate sum of weights
    i = 1
    P = particles(b)
    P_resampled = Vector{S}(undef, m)
    @inbounds for j in 1:m
        while U > c
            i += 1
            c += weight(b, i)
        end
        U += step
        P_resampled[j] = P[i]
    end
    return P_resampled
end

function resample(m::Int, b, pomdp::POMDP{S}, rng::AbstractRNG) where S
    i = 0
    P_resampled = Vector{S}(undef, m)
    @inbounds for i in 1:m
        s = rand(rng, b)
        while isterminal(pomdp, s)
            s = rand(rng, b)
        end
        P_resampled[i] = s
    end
    return P_resampled
end