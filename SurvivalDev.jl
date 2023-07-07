### Mainframe #####
### Kaplan-Meier

function surv_curve(survt, surve; default_end_of_study = 365 * 5, color="black")
    events = findall(surve .== 1)
    if length(events) != 0
        ordered_failure_times = sort(survt[events])
        end_of_study = ordered_failure_times[end]
    else
        ordered_failure_times = []
        end_of_study = default_end_of_study
    end
    
    ordered_failure_times = vcat([0], ordered_failure_times, [end_of_study])
    qfs = [] # nb of censored
    nfs = [length(survt)] # risk set
    Stf_hat = [1.] # estimator of survival proba 
    for (f, tf) in enumerate(ordered_failure_times[1:end - 1])
        # push nb of censored during interval
        push!(qfs, sum(surve[findall(survt .>= tf .&& survt .< ordered_failure_times[f + 1])] .== 0) )
        if f > 1
            de = Int(f > 2) # 0/1 indicator, first death is omitted 
            push!(nfs, nfs[f - 1] - qfs[f - 1] - de) # Risk set nb indiv. at risk before time tf
            push!(Stf_hat, Stf_hat[f - 1] * (nfs[f] - 1) / nfs[f]) # compute Stf hat (estimator of surv. proba)
        end 
        #println([f - 1, tf, qfs[f], nfs[f], Stf_hat[f]])
    end
    censored_tf = survt[surve .== 0]
    censored_Shat = []
    for tf in censored_tf
        Shat_position = Stf_hat[min(argmax(ordered_failure_times[ordered_failure_times .<= tf]),length(Stf_hat))]
        push!(censored_Shat, Shat_position)
    end
    surv_curv_1_1 = DataFrame(:tf=>ordered_failure_times[1:end-1], :Stf_hat=>Stf_hat)
    surv_curv_1_2 = DataFrame(:tf=>vcat(ordered_failure_times[2:end], [sort(censored_tf)[end]]), :Stf_hat=>vcat(Stf_hat, [Stf_hat[end]]))
    surv_curv_1 = append!(surv_curv_1_2, surv_curv_1_1)
    survt_data = surv_curv_1[sortperm(surv_curv_1.tf),:]
    censored_data = DataFrame(:tf => censored_tf, :Stf_hat=>censored_Shat)
    p1 = AlgebraOfGraphics.data(survt_data) * mapping(:tf, :Stf_hat) * visual(Lines, color = color, linewidth = 3)
    p2 = AlgebraOfGraphics.data(censored_data) * mapping(:tf, :Stf_hat) * visual(marker = [:vline for i in 1:size(censored_data)[1]])
    censored_data[:,"e"] .= zeros(size(censored_data)[1])
    surv_curv_1[:,"e"] .= ones(size(surv_curv_1)[1])
    sc1  = append!(surv_curv_1, censored_data)
    sc1 = sc1[sortperm(sc1.tf),:]
    sc2 = DataFrame(:i => collect(1:length(Stf_hat)), :tf => ordered_failure_times[1:end-1], :nf=>nfs,:qf=>qfs, :Stf_hat=>Stf_hat)
    return p1,p2, sc1, sc2
end 

function extract_surv_curves(survt, surve, subgroups)
    end_of_study = max(survt...)
    curves = []
    colors = ["red","blue","green","purple","grey","black", "yellow","orange"]
    for (i,group) in enumerate(unique(subgroups))
        cohort = findall(subgroups .== group)

        p, x, sc1, sc2 = surv_curve(survt[cohort],surve[cohort];color=colors[i])
        Stf_hat_labels = [round(Float32(sc1[findall(sc1.tf .>= i)[1],"Stf_hat"][1]);digits=3) for i in 0:1000:max(sc1.tf...)]
        fig = draw(p + x ,
            axis = (;xlabel = "Elpased Time (days)", ylabel = "Survival (fraction alive)", title =  "$group, (n=$(size(survt[cohort])[1]), c=$(sum(surve[cohort] .== 0)))", 
                limits = (0,end_of_study,0,1), yminorticksvisible = true, yminorgridvisible = true, yminorticks = IntervalsBetween(2),
                yticks = collect(0:10:100) ./ 100,
                xticks = (collect(0:1000:max(sc1.tf...)), ["$i\n$(label)" for (i,label) in zip(0:1000:max(sc1.tf...),Stf_hat_labels)]),
                )
            )
        
        push!(curves, (fig, sc1, sc2))
    end
    return curves
end 


function plot_surv_curves_combined(survt, surve, subgroups)
    end_of_study = max(survt...)
    curves = []
    colors = ["red","blue","green","purple","grey","black", "yellow","orange"]
    fig = Figure()
    Axis(fig[1,1])
    for (i,group) in enumerate(unique(subgroups))
        cohort = findall(subgroups .== group)
        # plot lines, add label
        # plot censored
        p, x, sc1, sc2 = surv_curve(survt[cohort], surve[cohort]; color = colors[i])
        main_ax[1] = lines() 
        # = surv_curve_makie(survt[cohort],surve[cohort];color=colors[i])
        # Stf_hat_labels = [round(Float32(sc1[findall(sc1.tf .>= i)[1],"Stf_hat"][1]);digits=3) for i in 0:1000:max(sc1.tf...)]
    end
    return fig
end 


### Metrics

function log_rank_test(survt, surve, subgroups, groups; end_of_study = 365 * 5)
    # LOG RANK test
    surve[survt .>= end_of_study] .= 0;
    survt[survt .>= end_of_study] .= end_of_study; 
    ordered_failure_times = sort(survt[findall(surve .== 1)])
    end_of_study = ordered_failure_times[end]
    ordered_failure_times = vcat([0],ordered_failure_times, [end_of_study])
    m1 = findall(subgroups .== groups[1])
    m2 = findall(subgroups .== groups[2])
    qfs1 = [] # nb of censored
    nfs1 = [length(survt[m1])] # risk set
    qfs2 = [] # nb of censored
    nfs2 = [length(survt[m2])] # risk set
    m1fs = []
    m2fs = []
    O_E = []
    for (f, tf) in enumerate(ordered_failure_times[1:end - 1])
    # push nb of censored during interval
    push!(qfs1, sum(surve[m1][findall(survt[m1] .>= tf .&& survt[m1] .< ordered_failure_times[f + 1])] .== 0) )
    push!(qfs2, sum(surve[m2][findall(survt[m2] .>= tf .&& survt[m2] .< ordered_failure_times[f + 1])] .== 0) )
    if f > 1
        m1f = Int(subgroups[findall(survt .== tf)[1]] == groups[1])
        m2f = Int(subgroups[findall(survt .== tf)[1]] == groups[2])
        
        push!(nfs1, nfs1[f - 1] - qfs1[f - 1] - m1f) # Risk set nb indiv. at risk before time tf
        push!(nfs2, nfs2[f - 1] - qfs2[f - 1] - m2f) # Risk set nb indiv. at risk before time tf
        push!(m1fs, m1f)
        push!(m2fs, m2f)
        e1f = nfs1[f] / (nfs1[f] + nfs2[f])
    
        push!(O_E, Int(subgroups[findall(survt .== tf)[1]] == groups[1]) - e1f)
    end
    
    end 
    table = DataFrame(:tf => ordered_failure_times[2:end-1], :m1f =>m1fs, :m2f =>m2fs, :nfs1 => nfs1[2:end], :nfs2 => nfs2[2:end], :O_E => O_E)
    V = sum(table.nfs1 .* table.nfs2 ./ (table.nfs1 .+ table.nfs2).^2)
    X = sum(O_E) ^ 2 / V
    
    logrank_pval = 1 - cdf(Chisq(1), X)
    return logrank_pval
end 
function plot_brca_subgroups(brca_data, groups, outpath; 
    end_of_study = 365 * 5, conf_interval = true, ticks = collect(0:250:end_of_study), 
    ylow = 0.5) 
    figs = []
    
    colors = ["red","blue","green","purple", "magenta","orange","yellow","grey", "black"]
    for group_of_interest in 1:size(groups)[1]
        fig = Figure(resolution =  (1000,1000));
        grid = fig[1,1] = GridLayout()
        axes = [grid[row,col] for row in 1:3 for col in 1:2]

        for i in 1:size(groups)[1]
            comp_groups = [groups[group_of_interest], groups[i]]
            comp_cohort = findall(brca_data.subgroups .== comp_groups[1] .|| brca_data.subgroups .== comp_groups[2])
            comp_survt= brca_data.survt[comp_cohort]
            comp_surve= brca_data.surve[comp_cohort]
            comp_subgroups = brca_data.subgroups[comp_cohort]
            lrt_pval = round(log_rank_test(comp_survt, comp_surve, comp_subgroups, comp_groups; end_of_study = end_of_study); digits = 5)

            group = groups[group_of_interest]
            cohort = findall(brca_data.subgroups .== group)
            # first subgroup
            p, x, sc1_ctrl, sc2_ctrl = surv_curve(brca_data.survt[cohort], brca_data.surve[cohort]; color = colors[i])
            p, x, sc1, sc2 = surv_curve(brca_data.survt[findall(brca_data.subgroups .== groups[i])], brca_data.surve[findall(brca_data.subgroups .== groups[i])]; color = colors[i])
            sc1_ctrl = sc1_ctrl[sortperm(sc1_ctrl.tf),:]
            
            Stf_hat_labels = ["$i\n$(label)" for (i,label) in zip(ticks, get_Stf_hat_surv_rates(ticks, sc1_ctrl))]    
            caption = "$group (n=$(length(cohort)),c=$(sum(brca_data.surve[cohort].==0))) vs $(groups[i]) (n=$(length(findall(brca_data.subgroups .== groups[i]))),c=$(sum(brca_data.surve[findall(brca_data.subgroups .== groups[i])].==0)))"
            if i != group_of_interest
                caption = "$caption\nLog-Rank Test pval: $lrt_pval"
                Stf_hat_labels = ["$tick\n$(label)" for (tick,label) in zip(Stf_hat_labels, get_Stf_hat_surv_rates(ticks, sc1))]       
            end
            ax = Axis(axes[i], limits = (0,end_of_study, ylow, 1.05), 
                yminorticksvisible = true, yminorgridvisible = true, yminorticks = IntervalsBetween(2),
                yticks = collect(0:10:100) ./ 100,
                xticks = (ticks, Stf_hat_labels),
                xlabel = "Elapsed time (days)",
                ylabel = "Survival (fraction still alive)",
                titlesize = 11, 
                xticklabelsize =11, 
                yticklabelsize =11, 
                ylabelsize = 11, 
                xlabelsize = 11,
                title = caption)
            # plot lines
            lines!(ax, sc1_ctrl[sc1_ctrl.e .== 1,:tf], sc1_ctrl[sc1_ctrl.e .== 1, :Stf_hat], color = "grey", label = groups[group_of_interest]) 
            # plot censored
            scatter!(ax, sc1_ctrl[sc1_ctrl.e .== 0,:tf], sc1_ctrl[sc1_ctrl.e .== 0, :Stf_hat], marker = [:vline for i in 1:sum(sc1_ctrl.e .== 0)], color = "black")
            # plot conf interval 
            if conf_interval
                conf_tf, lower_95, upper_95 = get_95_conf_interval(sc2_ctrl.tf, sc2_ctrl.nf, sc2_ctrl.Stf_hat, end_of_study)
                lines!(ax, conf_tf, upper_95, linestyle = :dot, color = "black")
                lines!(ax, conf_tf, lower_95, linestyle = :dot, color = "black")
                fill_between!(ax, conf_tf, lower_95, upper_95, color = ("black", 0.1))
            end 
            if i != group_of_interest
                # second subgroup
                sc1 = sc1[sortperm(sc1.tf),:] 
                # plot lines
                lines!(ax, sc1[sc1.e .== 1,:tf], sc1[sc1.e .== 1, :Stf_hat], color = colors[i], label = groups[i]) 
                # plot censored
                scatter!(ax, sc1[sc1.e .== 0,:tf], sc1[sc1.e .== 0, :Stf_hat], marker = [:vline for i in 1:sum(sc1.e .== 0)], color = "black")
                axislegend(ax, position = :rb, labelsize = 11, framewidth = 0)
                #plot conf interval
                if conf_interval
                    conf_tf, lower_95, upper_95 = get_95_conf_interval(sc2.tf, sc2.nf, sc2.Stf_hat, end_of_study)
                    lines!(ax, conf_tf, upper_95, linestyle = :dot, color = colors[i])
                    lines!(ax, conf_tf, lower_95, linestyle = :dot, color = colors[i])
                    fill_between!(ax, conf_tf, lower_95, upper_95, color = (colors[i], 0.1))
                end 
            end
        end
        commitnb = getcurrentcommitnb()
        title_ax = Axis(fig[1,1]; title = "Survival curves $(groups[group_of_interest]) vs other subgroups\n in TCGA Breast Cancer Data (n=$(length(brca_data.survt)),c=$(sum(brca_data.surve .== 0)))\n$commitnb ", spinewidth = 0, titlegap = 50, titlesize = 14)
        hidedecorations!(title_ax)
        push!(figs,fig)
        CairoMakie.save("$outpath/surv_curves_$(groups[group_of_interest])_1v1.pdf", fig)
    end
    return figs
end


function get_95_conf_interval(tf, nf, Stf_hat, end_of_study)
    cond_risk = 1 ./ nf ./ (nf .- 1)
    cond_risk[1] = 0
    varStf_hat = Stf_hat .^ 2 .* cumsum(cond_risk)
    upper_95 = round.(min.(Stf_hat .+ 1.96 .* sqrt.(varStf_hat), ones(length(varStf_hat))), digits = 3)
    lower_95 = round.(max.(Stf_hat .- 1.96 .* sqrt.(varStf_hat), zeros(length(varStf_hat))), digits = 3)
    conf_tf = sort(vcat(tf, vcat(tf[2:end], end_of_study)))
    lower_95 = reverse(sort(vcat(lower_95, lower_95)))
    upper_95 = reverse(sort(vcat(upper_95, upper_95)))
    return conf_tf, lower_95, upper_95
end 


### Cox Proportional Hazards
### Deep Neural Network CPH
