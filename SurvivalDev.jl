### Mainframe #####
### Kaplan-Meier

function surv_curve(survt, surve; color="black")
    ordered_failure_times = sort(survt[findall(surve .== 1)])
    end_of_study = ordered_failure_times[end]
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
### Cox Proportional Hazards
### Deep Neural Network CPH