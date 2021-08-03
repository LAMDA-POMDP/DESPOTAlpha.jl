function build_tree(p::DESPOTAlphaPlanner{S,A}, b0) where {S, A}
    D = DESPOTAlphaTree(p, b0)
    b = 1
    trial = 1
    start = CPUtime_us()

    Depth = sizehint!(Int[], 2000)
    sol = solver(p)
    while D.u[1]-D.l[1] > sol.epsilon_0 &&
          CPUtime_us()-start < sol.T_max*1e6 &&
          trial <= sol.max_trials
        b, as = explore!(D, 1, A[], p)
        backup!(D, b, as, p)
        push!(Depth, D.Delta[b])
        trial += 1
    end
    if (CPUtime_us()-start)*1e-6 > sol.T_max*sol.timeout_warning_threshold
        @warn(@sprintf("Surpass the time limit. The actual runtime is %3.1fs.
        Hyperparameters: K=%3d, C=%3d, bounds=%s",
        (CPUtime_us()-start)*1e-6, sol.K, sol.C, typeof(sol.bounds)))
        info_analysis(Dict(:tree=>D, :depth=>Depth))
    end
    return D, Depth
end

function explore!(D::DESPOTAlphaTree{S,A}, b::Int, as::Vector{A}, p::DESPOTAlphaPlanner{S,A}) where {S,A}
    sol = solver(p)
    while D.Delta[b] < sol.max_depth &&
        excess_uncertainty(D, b, sol.xi, p) > 0.0

        if isempty(D.children[b]) # a leaf
            expand!(D, b, as, p)
        end
        ba, b = next_best(D, b, p)
        push!(as, D.ba_action[ba])
    end
    if D.Delta[b] == sol.max_depth
        D.u[b] = D.l[b]
    end
    return b, as
end

function backup!(D::DESPOTAlphaTree, b::Int, as::Vector, p::DESPOTAlphaPlanner)
    disc = discount(p.pomdp)
    sol = solver(p)

    @assert b != 1
    ba = D.parent[b]
    as_ba = D.as_map[as]
    for sib in D.ba_children[ba]
        UpdateSiblings(D, b, as_ba, sib)
    end
    b = D.ba_parent[ba]
    while true
        r = D.weights[b]' * D.r[as_ba]
        D.ba_l[ba] = r + disc * sum(D.l[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])
        D.ba_u[ba] = r + disc * sum(D.u[bp] * D.obs_prob[bp] for bp in D.ba_children[ba])

        if D.ba_l[ba] > D.l[b]
            D.l[b] = D.ba_l[ba]
            new_alpha = zeros(sol.K)
            for spi in 1:sol.K
                for (oi, bp) in enumerate(D.ba_children[ba])
                    new_alpha[spi] += D.L[as_ba][oi, spi] * D.alpha[bp][spi]
                end
                new_alpha[spi] *= disc
            end
            new_alpha .+= D.r[as_ba]
            D.alpha[b] = new_alpha
        end
        
        if D.ba_u[ba] < D.u[b]
            D.u[b] = maximum(D.ba_u[ba] for ba in D.children[b])
        end

        if sol.pp_update
            len_as = length(as)
            as_b = D.as_map[view(as, 1:(len_as-1))]
            new_pp = fill(-Inf, sol.K)
            for ba in D.children[b]
                as[len_as] = D.ba_action[ba]
                new_pp .= max.(new_pp, D.r[as_b] .+ disc .* D.pp_alpha_upper[D.as_map[as]])
            end
            D.pp_alpha_upper[as_b] .= min.(D.pp_alpha_upper[as_b], new_pp)
        end

        if b == 1
            break
        end
        as = view(as, 1:length(as)-1)
        ba = D.parent[b]
        as_ba = D.as_map[as]
        for sib in D.ba_children[ba]
            UpdateSiblings(D, b, as_ba, sib)
        end
        b = D.ba_parent[ba]
    end
    return nothing
end

function UpdateSiblings(D::DESPOTAlphaTree, trial::Int, as::Int, sib::Int)
    new_l = D.alpha[trial]' * D.weights[sib]
    if new_l > D.l[sib]
        D.l[sib] = new_l
        D.alpha[sib] = D.alpha[trial]
    end
    # @assert D.u[trial] >= D.weights[trial]' * D.pp_alpha_upper[as]
    theta = Inf
    for si in eachindex(D.weights[trial])
        new_theta = D.weights[sib][si] / (D.weights[trial][si] + 1e-10)
        if new_theta < theta
            theta = new_theta
        end
    end
    new_u = theta * D.u[trial]
    for si in eachindex(D.weights[trial])
        new_u += (D.weights[sib][si] - theta * D.weights[trial][si]) * D.pp_alpha_upper[as][si]
    end
    if new_u < D.u[sib]
        D.u[sib] = new_u
    end
end

function next_best(D::DESPOTAlphaTree, b::Int, p::DESPOTAlphaPlanner)
    max_u = -Inf
    best_ba = first(D.children[b])
    @inbounds for ba in D.children[b]
        if D.ba_u[ba] > max_u
            max_u = D.ba_u[ba]
            best_ba = ba
        end
    end

    max_eu = -Inf
    best_bp = first(D.ba_children[best_ba])
    tolerated_gap = solver(p).xi * max(D.u[1]-D.l[1], 0.0) / p.discounts[D.Delta[best_bp]+1]
    @inbounds for bp in D.ba_children[best_ba]
        eu = D.obs_prob[bp] * (D.u[bp]-D.l[bp] - tolerated_gap)
        if eu > max_eu
            max_eu = eu
            best_bp = bp
        end
    end

    return best_ba, best_bp
end

function excess_uncertainty(D::DESPOTAlphaTree, b::Int, ξ::Float64, p::DESPOTAlphaPlanner)
    return D.obs_prob[b] * (D.u[b]-D.l[b] - ξ * max(D.u[1]-D.l[1], 0.0) / p.discounts[D.Delta[b]+1])
end