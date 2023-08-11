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
get_Stf_hat_surv_rates(ticks, sc1)  = [round(sc1[findall(sc1.tf .<= i)[end],"Stf_hat"][1];digits=2) for i in ticks]
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
### TRAIN COXPH 
function cox_nll(t, e, out)
    ### data already sorted
    # sorted_ids = sortperm(t)
    # E = e[sorted_ids]
    # OUT = out[sorted_ids]
    uncensored_likelihood = 0
    for (x_i, e_i) in enumerate(E)
        if e_i == 1
            log_risk = log(sum(ℯ .^ OUT[1:x_i]))
            uncensored_likelihood += OUT[x_i] - log_risk    
        end 
    end 
    loss = - uncensored_likelihood / sum(E .== 1)
    return loss
end 

function cox_nll(OUT, E)
    ### data already sorted
    # sorted_ids = sortperm(t)
    # E = e[sorted_ids]
    # OUT = out[sorted_ids]
    uncensored_likelihood = 0
    for (x_i, e_i) in enumerate(E)
        if e_i == 1
            log_risk = log(sum(ℯ .^ OUT[1:x_i]))
            uncensored_likelihood += OUT[x_i] - log_risk    
        end 
    end 
    loss = - uncensored_likelihood / sum(E .== 1)
    return loss
end 

function build_cphdnn(params;device =gpu)
    mdl = Flux.Chain(Flux.Dense(params["insize"], params["hl1_size"], relu), 
    Flux.Dense(params["hl1_size"], params["hl2_size"], relu), 
    Flux.Dense(params["hl2_size"], 1, params["acto"]));
    return device(mdl) 
end 

function cox_nll_vec(mdl::Flux.Chain, X_, Y_e_, NE_frac)
    outs = vec(mdl(gpu(X_)))
    hazard_ratios = exp.(outs)
    log_risk = log.(cumsum(hazard_ratios))
    uncensored_likelihood = outs .- log_risk
    censored_likelihood = uncensored_likelihood .* gpu(Y_e_')
    #neg_likelihood = - sum(censored_likelihood) / sum(e .== 1)
    neg_likelihood = - sum(censored_likelihood) * NE_frac
    return neg_likelihood
end 


function concordance_index(T, E, S)
    concordant_pairs = S .> S'
    admissable_pairs = T .< T'
    discordant_pairs = S .< S'
    concordant = sum(E .* (admissable_pairs .* concordant_pairs))
    discordant = sum(E .* (admissable_pairs .* discordant_pairs) )
    C_index = concordant / (concordant + discordant)
    return C_index, concordant, discordant
end

function c_index_dev(T,E,S)
    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0
    denominator = 0

    n = length(S)
    tied_tol = 1e-8
    for i in 1:n
        if E[i] == 1
            for j in i+1:n 
            if T[i] < T[j]
                # concordant
                if S[j] < S[i]
                    concordant += 1
                    numerator += 1
                    denominator += 1
                elseif S[j] > S[i]
                    discordant += 1
                    denominator += 1
                else
                    numerator += 0.5
                    denominator += 1
                end 
                # discordant 
                # tied 
                end
            end  
        end 
    end 
    return numerator / denominator, concordant, discordant
end

function split_train_test(X::Matrix, Y_t::Vector,Y_e::Vector, case_ids::Vector; nfolds = 10)
    folds = Array{Dict, 1}(undef, nfolds)
    nsamples = size(X)[1]
    fold_size = Int(floor(nsamples / nfolds))
    ids = collect(1:nsamples)
    shuffled_ids = shuffle(ids)
    for i in 1:nfolds
        tst_ids = shuffled_ids[collect((i-1) * fold_size +1: min(nsamples, i * fold_size))]
        tr_ids = setdiff(ids, tst_ids)
        X_train = X[tr_ids,:]
        Y_t_train = Y_t[tr_ids]
        Y_e_train = Y_e[tr_ids]
        X_test = X[tst_ids,:]
        Y_t_test = Y_t[tst_ids]
        Y_e_test = Y_e[tst_ids]

        folds[i] = Dict("foldn"=> i, "train_ids"=>tr_ids, "test_ids"=>case_ids[tst_ids],
                        "train_case_ids"=>case_ids[tr_ids], "train_x"=>X_train,"Y_t_train"=>Y_t_train, "Y_e_train"=>Y_e_train,
                        "tst_case_ids"=>case_ids[tst_ids], "test_x"=>X_test, "Y_t_test"=>Y_t_test, "Y_e_test"=>Y_e_test)
    end
    return folds 
end 
lossf(mdl, X, Y_e, NE, wd) = cox_nll_vec(mdl, X, Y_e, NE) + wd * l2_penalty(mdl)
    
function train_cphdnn!(mdl,X_train, Y_t_train, Y_e_train, X_test, Y_t_test, Y_e_test;nsteps=20_000,wd=1e-3)
    mdl_opt = Flux.ADAM(1e-3)
    loss_tr = []
    loss_tst = []
    c_ind_tr = []
    c_ind_tst = []
    for step in 1:nsteps
        NE_frac_tr = sum(Y_e_train .== 1) != 0 ? 1 / sum(Y_e_train .== 1) : 0 
        NE_frac_tst = sum(Y_e_test .== 1) != 0 ? 1 / sum(Y_e_test .== 1) : 0 
        lossval_tr = lossf(mdl, X_train, Y_e_train, NE_frac_tr, wd)
        lossval_tst = lossf(mdl, X_test, Y_e_test, NE_frac_tst, wd)
        push!(loss_tr, lossval_tr)
        push!(loss_tst, lossval_tst)
        if step % 1000==0 || step == 1 
            push!(c_ind_tr, concordance_index(vec(mdl(X_train)), Y_t_train, Y_e_train))
            push!(c_ind_tst, concordance_index(vec(mdl(X_test)), Y_t_test, Y_e_test))
            println("$step TRAIN c_ind: $(round(c_ind_tr[end], digits = 3)) loss: $(round(loss_tr[end], digits =3)), TEST c_ind: $(round(c_ind_tst[end],digits =3)) loss: $(round(loss_tst[end], digits = 3))")
        end
        ps = Flux.params(mdl)
        gs = gradient(ps) do 
            lossf(mdl, X_train, Y_e_train, NE_frac_tr, wd)
        end 
        Flux.update!(mdl_opt, ps, gs)
    end 
    return loss_tr, loss_tst, c_ind_tr, c_ind_tst
end 

function validate_cphdnn(params, folds;)
    nfolds = size(folds)[1]
    outs = []
    survts = []
    surves = []
    for foldn in 1:nfolds 
        fold = folds[foldn]

        sorted_ids = reverse(sortperm(fold["Y_t_train"]))
        X_train = gpu(Matrix(fold["X_train"][sorted_ids,:]'))
        Y_t_train = gpu(fold["Y_t_train"][sorted_ids])
        Y_e_train = gpu(fold["Y_e_train"][sorted_ids])


        sorted_ids = reverse(gpu(sortperm(fold["Y_t_test"])))
        X_test = gpu(Matrix(fold["X_test"][sorted_ids,:]'))
        Y_t_test = gpu(fold["Y_t_test"][sorted_ids])
        Y_e_test = gpu(fold["Y_e_test"][sorted_ids])

        mdl_opt = Flux.ADAM(1e-4)
        
        mdl = build_cphdnn(params)    
        loss_tr, loss_vld, c_ind_tr, c_ind_vld = train_cphdnn!(mdl, X_train, Y_t_train, Y_e_train, X_test, Y_t_test, Y_e_test;nsteps =params["nbsteps"],wd=parmas["wd"])
        push!(outs, vec(cpu(mdl(X_test))))
        push!(survts, cpu(Y_t_test))
        push!(surves, cpu(Y_e_test))
        
        fig = Figure();
        ax = Axis(fig[1,1],xlabel = "Nb. of gradient steps", ylabel ="Cox Negative-Likelihood", title = "FOLD: $foldn, sample size: $(size(X_train)[1])")
        lines!(ax, collect(1:length(loss_tr)), Vector{Float32}(loss_tr),color = "blue",label = "training")
        lines!(ax, collect(1:length(loss_vld)), Vector{Float32}(loss_vld),color = "orange",label = "test")
        axislegend(ax, position = :rb)
        fig
        CairoMakie.save("$outpath/training_curve_loss_fold_$foldn.pdf",fig)

        fig = Figure();
        ax = Axis(fig[1,1],xlabel = "Nb. of gradient steps", ylabel ="Concordance index", limits = (0,nbsteps,0.5,1))
        lines!(ax, vcat([1], collect(1000:1000:nbsteps)), Vector{Float32}(c_ind_tr),color = "blue",label = "training")
        lines!(ax, vcat([1], collect(1000:1000:nbsteps)), Vector{Float32}(c_ind_vld),color = "orange",label = "test")
        axislegend(ax, position = :rb)
        fig
        CairoMakie.save("$outpath/training_curve_c_index_fold_$foldn.pdf",fig)
    end
    return outs, survts, surves
end 