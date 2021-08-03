function D3Trees.D3Tree(D::DESPOTAlphaTree; title="DESPOTAlpha Tree", kwargs...)
    lenb = D.b
    lenba = D.ba
    len = lenb + lenba
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tt = fill("", len)
    link_style = fill("", len)
    for b in 1:lenb
        as = retrieve_as(D, b)
        obs = b == 1 ? "<root>" : string(D.obs[D.as_map[as]][b-D.ba_children[D.parent[b]][1]+1])
        children[b] = D.children[b] .+ lenb
        text[b] = @sprintf("""
                           o:%s prob:%6.2f
                           l:%6.2f, u:%6.2f""",
                           obs,
                           D.obs_prob[b],
                           D.l[b],
                           D.u[b],
                          )
        tt[b] = """
                o: $(obs)
                prob: $(D.obs_prob[b])
                l: $(D.l[b])
                u: $(D.u[b])
                $(length(D.children[b])) children
                """
        link_width = 2.0
        link_style[b] = "stroke-width:$link_width"
        for ba in D.children[b]
            link_style[ba+lenb] = "stroke-width:$link_width"
        end

        push!(as, zero(eltype(as)))
        for ba in D.children[b]
            as[length(as)] = D.ba_action[ba]
            as_ba = D.as_map[as]
            children[ba+lenb] = D.ba_children[ba]
            text[ba+lenb] = @sprintf("""
                                     a:%s r:%6.2f |ϕ|:%2d
                                     l:%6.2f, u:%6.2f""",
                                     D.ba_action[ba], D.weights[b]' * D.r[as_ba], length(D.particles[as_ba]),
                                     D.ba_l[ba], D.ba_u[ba],
                                     )
            tt[ba+lenb] = """
                          a: $(D.ba_action[ba])
                          r: $(D.weights[b]' * D.r[as_ba])
                          |ϕ|:$(length(D.particles[as_ba]))
                          l: $(D.ba_l[ba])
                          u: $(D.ba_u[ba])
                          $(length(D.ba_children[ba])) children
                          """
        end

    end
    return D3Tree(children;
                  text=text,
                  tooltip=tt,
                  link_style=link_style,
                  title=title,
                  kwargs...
                 )
end

Base.show(io::IO, mime::MIME"text/html", D::DESPOTAlphaTree) = show(io, mime, D3Tree(D))
Base.show(io::IO, mime::MIME"text/plain", D::DESPOTAlphaTree) = show(io, mime, D3Tree(D))

function retrieve_as(D::DESPOTAlphaTree{S,A}, b::Int) where {S,A}
    as = Vector{A}(undef, D.Delta[b])
    while b != 1
        bp = b
        ba = D.parent[bp]
        b = D.ba_parent[ba]
        as[D.Delta[bp]] = D.ba_action[ba]
    end
    return as
end
