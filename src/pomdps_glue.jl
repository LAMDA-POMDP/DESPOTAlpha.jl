POMDPs.solve(sol::DESPOTAlphaSolver, p::POMDP) = DESPOTAlphaPlanner(sol, p)

function POMDPModelTools.action_info(p::DESPOTAlphaPlanner{S,A}, b) where {S,A}
    info = Dict{Symbol, Any}()
    sol = solver(p)
    try
        D, Depth = build_tree(p, b)

        info[:depth] = Depth
        if sol.tree_in_info
            info[:tree] = D
        else
            p.tree = D
        end

        best_l = -Inf
        best_a = A[]
        for ba in D.children[1]
            l = D.ba_l[ba]
            if l > best_l
                best_l = l
                best_a = [D.ba_action[ba]]
            elseif l == best_l
                push!(best_a, D.ba_action[ba])
            end
        end

        return rand(p.rng, best_a)::A, info # best_a will usually only have one entry, but we want to break the tie randomly
    catch ex
        return default_action(sol.default_action, p.pomdp, b, ex)::A, info
    end
end

POMDPs.action(p::DESPOTAlphaPlanner{S,A}, b) where {S,A} = first(action_info(p, b))::A
function POMDPs.updater(p::DESPOTAlphaPlanner)
    sol = solver(p)
    SIRParticleFilter(p.pomdp, sol.K*50, rng=p.rng)
end

function Random.seed!(p::DESPOTAlphaPlanner, seed)
    Random.seed!(p.rng, seed)
    return p
end