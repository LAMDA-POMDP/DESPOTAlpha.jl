function info_analysis(info::Dict)
    D = get(info, :tree, nothing)
    if D !== nothing
        show(stdout, MIME("text/plain"), D)
        println("Number of action node expanded:", D.ba)
        println("Number of belief node expanded:", D.b)
    end
    depth = info[:depth]
    println("Times of exploration: ", length(depth))
    println(@sprintf("Depth of exploration: mean±std = %5.2f±%4.2f", mean(depth), std(depth)))
    println("90% Confidence interval = ", quantile(depth, (0.05, 0.95)))
    return nothing
end

function hist_analysis(hist::H; display_mean_and_std::Bool = false, layout=(1,2), font_size=12, margin=40px, figure_size=(900,400)) where H<:AbstractSimHistory
    infos = ainfo_hist(hist)

    median_d = Float64[]
    lower_d = Float64[]
    upper_d = Float64[]
    mean_d = Float64[]
    std_d = Float64[]

    for info in infos
        depth = info[:depth]
        l_d, m_d, u_d = quantile(depth, (0.05, 0.5, 0.95))
        push!(median_d, m_d)
        push!(lower_d, m_d-l_d)
        push!(upper_d, u_d-m_d)
        push!(mean_d, mean(depth))
        push!(std_d, std(depth))
    end
    if display_mean_and_std
        p1 = plot(median_d, ribbon=(lower_d, upper_d), xaxis="Steps", yaxis="Depth of Exploration", label="quantile", xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size)
        plot!(p1, mean_d, ribbon=std_d, label="mean", xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size)
    else
        p1 = plot(median_d, ribbon=(lower_d, upper_d), xaxis="Steps", yaxis="Depth of Exploration", legend=false, xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size)
    end

    D = get(first(infos), :tree, nothing)
    if D === nothing
        display(p1)
    else
        num_anode = Int[]
        for info in infos
            D = info[:tree]
            push!(num_anode, D.ba)
        end
        base = 10^floor(log(10, mean(num_anode)))
        p2 = plot(num_anode, legend=false, xaxis="Steps", yaxis="Action Nodes Expanded", yformatter=y->y/base, xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size)
        annotate!(p2, [(0.9, maximum(num_anode) + (maximum(num_anode)-minimum(num_anode)) * 0.07, Plots.text(@sprintf("\$\\times10^{%d}\$", round(Int, log(10,base))), font_size, :black, :center, "courier"))])
        display(plot(p1, p2, layout=layout, size=figure_size, margin=margin))
    end
    return nothing
end