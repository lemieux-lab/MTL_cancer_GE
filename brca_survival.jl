# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")

outpath, session_id = set_dirs() 
#brca_prediction = GDC_data("Data/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);

clin_data = CSV.read("Data/GDC_processed/TCGA_BRCA_clinicial_raw.csv", DataFrame, header = 2)
names(clin_data)
CLIN_FULL = CSV.read("Data/GDC_processed/GDC_clinical_raw.tsv", DataFrame)
features = ["case_id", "case_submitter_id", "project_id", "gender", "age_at_index","age_at_diagnosis", "days_to_death", "days_to_last_follow_up", "primary_diagnosis", "treatment_type"]
CLIN = CLIN_FULL[:, features]

names(clin_data)
clin_data = clin_data[:,["Complete TCGA ID", "Days to date of Death", "Vital Status", "Days to Date of Last Contact", "PAM50 mRNA"]]
clin_data[:,"case_submitter_id"] .= clin_data[:,"Complete TCGA ID"]
clin_data = clin_data[:,2:end]
counter(feature, clin_data) = Dict([(x, sum(clin_data[:, feature] .== x)) for x in unique(clin_data[:,feature])])
counter("Integrated Clusters (with PAM50)", clin_data)
counter("PAM50 mRNA", clin_data)
counter("Integrated Clusters (unsup exp)", clin_data)
counter("Integrated Clusters (no exp)", clin_data)
counter("CN Clusters", clin_data)
counter("SigClust Unsupervised mRNA", clin_data)
counter("SigClust Intrinsic mRNA", clin_data)

intersect(CLIN[:,"case_submitter_id"], clin_data[:,"Complete TCGA ID"])
BRCA_CLIN = innerjoin(clin_data,CLIN, on = :case_submitter_id)
CSV.write("Data/GDC_processed/tmp.csv", BRCA_CLIN)
BRCA_CLIN = BRCA_CLIN[findall(nonunique(BRCA_CLIN[:,1:end-1])),:]
CSV.write("Data/GDC_processed/TCGA_BRCA_clinical_survival.csv", BRCA_CLIN)
BRCA_CLIN[findall(BRCA_CLIN[:,"Days to date of Death"] .== "[Not Available]"), "Days to date of Death"] .= "NA"
BRCA_CLIN[findall(BRCA_CLIN[:,"Days to Date of Last Contact"] .== "[Not Available]"), "Days to Date of Last Contact"] .= "NA"
BRCA_CLIN = BRCA_CLIN[findall(BRCA_CLIN[:,"Days to date of Death"] .!= "NA" .|| BRCA_CLIN[:,"Days to Date of Last Contact"] .!= "NA"),:]

CSV.write("Data/GDC_processed/TCGA_BRCA_clinical_survival.csv", BRCA_CLIN)
BRCA_CLIN.case_id
BRCA_CLIN = CSV.read("Data/GDC_processed/TCGA_BRCA_clinical_survival.csv", DataFrame)
CSV.write("RES/SURV/TCGA_BRCA_clinical_survival.csv", BRCA_CLIN)

max(BRCA_CLIN.survt...)
#TCGA_all = GDC_data("Data/GDC_processed/TCGA_TPM_lab.h5")
TCGA_all = GDC_data("Data/GDC_processed/GDC_STAR_TPM.h5")

BRCA_gdc_ids = findall([ in(x, BRCA_CLIN.case_id) for x in  TCGA_all.rows])
BRCA_ids = TCGA_all.rows[BRCA_gdc_ids]
sort!(BRCA_CLIN, :case_id)

survt = Array{String}(BRCA_CLIN[:,"Days to date of Death"])
survt[survt .== "NA"] .= BRCA_CLIN[survt .== "NA","days_to_last_follow_up"]
BRCA_CLIN[:,"days_to_last_follow_up"]
survt = [Int(parse(Float32, x)) for x in survt]
surve = [Int(x == "DECEASED") for x in BRCA_CLIN[:,"Vital Status"]]
BRCA_CLIN = BRCA_CLIN[:,1:14]
BRCA_CLIN[:,"survt"] .= survt 
BRCA_CLIN[:,"surve"] .= surve 
BRCA_CLIN
survt
TCGA_all.data[BRCA_gdc_ids[sortperm(BRCA_ids)],:]
TCGA_BRCA_TPM_surv = GDC_data_surv(TCGA_all.data[BRCA_gdc_ids[sortperm(BRCA_ids)],:], sort(BRCA_ids), TCGA_all.cols, Array{String}(BRCA_CLIN[:,"PAM50 mRNA"]), survt, surve) 
write_h5(TCGA_BRCA_TPM_surv, "Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5")

brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);

##### DEV
subgroups = brca_prediction.subgroups
end_of_study = 365 * 10 # 10 years 
survt = brca_prediction.survt
surve = brca_prediction.surve
surve[survt .>= end_of_study] .= 0 
survt[survt .>= end_of_study] .= end_of_study 

figures = plot_brca_subgroups(brca_prediction, unique(brca_prediction.subgroups), outpath; end_of_study = end_of_study);

### ALL samples ###
### ANALYSIS using PCA Gene Expression COXPH
sum(BRCA_CLIN.survt .== brca_prediction.survt)
sum(BRCA_CLIN[:,"PAM50 mRNA"] .== brca_prediction.subgroups)
names(BRCA_CLIN)
BRCA_CLIN = BRCA_CLIN[:,1:14]
BRCA_CLIN = innerjoin(BRCA_CLIN, tmp, on = :case_id)
brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
highv = reverse(sortperm([var(x) for x in 1:size(brca_prediction.data)[2]]))
highv_25 = highv[1:Int(floor(length(highv)*0.25))]
brca_pred_subset = GDC_data_surv(brca_prediction.data[:,highv_25], brca_prediction.rows, brca_prediction.cols[highv_25], brca_prediction.subgroups, survt, surve)

# CSV tcga brca pca 25
using MultivariateStats
pca = fit(PCA, brca_pred_subset.data', maxoutdim=25, method = :cov);
brca_prediction_pca = predict(pca, brca_pred_subset.data')';
brca_pca_25_df = DataFrame(Dict([("pc_$i",brca_prediction_pca[:,i]) for i in 1:size(brca_prediction_pca)[2]]))
brca_pca_25_df[:,"case_id"] .= case_ids 
CSV.write("RES/SURV/TCGA_BRCA_pca_25_case_ids.csv", brca_pca_25_df)
CSV.write("RES/SURV/TCGA_BRCA_clinical_survival.csv", BRCA_CLIN)
# CSV tcga brca surv + clin data 
brca_pca = CSV.read("RES/SURV/TCGA_brca_pca_25_case_ids.csv", DataFrame)
### TRAIN COXPH 
function cox_nll(t, e, out)
    ### data already sorted
    # sorted_ids = sortperm(t)
    # E = e[sorted_ids]
    # OUT = out[sorted_ids]
    uncensored_likelihood = 0
    for (x_i, e_i) in enumerate(E)
        if e_i == 1
            log_risk = log(sum(ℯ .^ OUT[1:x_i+1]))
            uncensored_likelihood += OUT[x_i] - log_risk    
        end 
    end 
    loss = - uncensored_likelihood / sum(E .== 1)
    return loss
end 

function cox_nll_vec(t, e, out, observed_events)
    ### working 
    ### weights all over the place (uncomputable loss)
    hazard_ratios = ℯ .^ out
    log_risk = log.(cumsum(hazard_ratios))
    uncensored_likelihood = out .- log_risk
    censored_likelihood = uncensored_likelihood .* e
    #neg_likelihood = - sum(censored_likelihood) / sum(e .== 1)
    neg_likelihood = - sum(censored_likelihood) / observed_events
    return neg_likelihood
end 
function l2_penalty(model)
    l2_sum = 0
    for wm in model
        l2_sum += sum(abs2, wm.weight)
    end 
    return l2_sum
end
function concordance_index(scores, survt, surve)
    function helper(S,T,E)
        #vector computation of c_index 
        n = length(S)
        concordant_pairs = S .> S'
        tied_pairs = sum(S .== S', dims = 1)' - ones(length(S))
        admissable_pairs = T .< T'
        c_ind = sum(E' .* admissable_pairs .* concordant_pairs)+ 0.5 * sum(tied_pairs)
        c_ind = c_ind / (sum(E .* sum(admissable_pairs, dims = 1)') + sum(tied_pairs))
        return c_ind 
    end 
    c = helper(scores,survt,surve)
    if c < 0.5 
        c = helper(-scores, survt,surve)
    end
    return c 
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

        folds[i] = Dict("tr_ids"=>case_ids[tr_ids], "X_train"=>X_train,"Y_t_train"=>Y_t_train, "Y_e_train"=>Y_e_train,
                        "tst_ids"=>case_ids[tst_ids], "X_test"=>X_test, "Y_t_test"=>Y_t_test, "Y_e_test"=>Y_e_test)
    end
    return folds 
end 
function train_cphdnn!(mdl,X_train, Y_t_train, Y_e_train, X_test, Y_t_test, Y_e_test;nsteps=20_000 )
    loss_tr = []
    loss_tst = []
    c_ind_tr = []
    c_ind_tst = []
    for step in 1:nsteps
               
        push!(loss_tr, lossf(mdl, X_train, Y_t_train, Y_e_train, sum(Y_e_train .== 1),wd))
        push!(loss_tst, lossf(mdl, X_test, Y_t_test, Y_e_test, sum(Y_e_test .== 1),wd))
        if step % 1000==0 || step == 1 
            push!(c_ind_tr, concordance_index(vec(mdl(X_train)), Y_t_train, Y_e_train))
            push!(c_ind_tst, concordance_index(vec(mdl(X_test)), Y_t_test, Y_e_test))
            println("TRAIN c_ind: $(c_ind_tr[end]) loss: $(round(loss_tr[end], digits =3)), TEST c_ind: $(c_ind_tst[end]) loss: $(round(loss_tst[end], digits = 3))")
        end
        ps = Flux.params(mdl)
        gs = gradient(ps) do 
            lossf(mdl, X_train, Y_t_train, Y_e_train, sum(Y_e_train .== 1),wd)
        end 
        Flux.update!(mdl_opt, ps, gs)
    end 
    return loss_tr, loss_tst, c_ind_tr, c_ind_tst
end 
folds[1]["tst_ids"]
cox_nll(brca_prediction.survt, brca_prediction.surve, outs)
mdl_opt = Flux.ADAM(1e-3)
wd = 1e-3
mdl = gpu(Flux.Chain(Flux.Dense(25, 10, relu), Flux.Dense(10, 1, identity)));
lossf(model, X, t, e, nE, wd) = cox_nll_vec(t, e, vec(model(X)), nE) + wd * l2_penalty(model)
sorted_ids = sortperm(brca_prediction.survt)
X = gpu(Matrix(brca_pca[:,1:25][sorted_ids,:])')
Y_t = gpu(brca_prediction.survt[sorted_ids])
Y_e = gpu(brca_prediction.surve[sorted_ids])
ps = Flux.params(mdl)
lossval = lossf(mdl, X, Y_t, Y_e, sum(Y_e .== 1), wd)
gs = gradient(ps) do 
    lossf(mdl, X, Y_t,Y_e, sum(Y_e .== 1), wd)
end
Flux.update!(mdl_opt, ps, gs)

lossf(model, X, t, e, nE, wd) = cox_nll_vec(t, e, vec(model(X)), nE) + wd * l2_penalty(model)
insize = 29
nfolds = 5
brca_pca[:,"is_her2"] .= subgroups .== "HER2-enriched"
brca_pca[:,"is_lumA"] .= subgroups .== "Luminal A"
brca_pca[:,"is_lumB"] .= subgroups .== "Luminal B"
brca_pca[:,"is_bsllk"] .= subgroups .== "Basal-like"

folds = split_train_test(Matrix(brca_pca[:,vcat(collect(1:insize), collect(27:30))]), brca_prediction.survt, brca_prediction.surve, vec(brca_pca[:,26]);nfolds = nfolds)
outs = []
survts = []
surves = []
fold = folds[1]
sorted_ids = sortperm(fold["Y_t_train"])
X = fold["X_train"][sorted_ids,:]'
Y_t = fold["Y_t_train"][sorted_ids]
Y_e = fold["Y_e_train"][sorted_ids]

nbsteps = 50_000
wd = 1e-3
mdl_opt = Flux.ADAM(1e-3)
mdl = Flux.Chain(Flux.Dense(17, 10, relu), Flux.Dense(10, 1, sigmoid));
ps = Flux.params(mdl)
lossval = lossf(mdl, X, Y_t, Y_e, sum(Y_e .== 1), wd)
gs = gradient(ps) do 
    lossf(mdl, X, Y_t,Y_e, sum(Y_e .== 1), wd)
end
Flux.update!(mdl_opt, ps, gs)

for foldn in 1:nfolds 
    fold = folds[foldn]

    sorted_ids = sortperm(fold["Y_t_train"])
    X_train = fold["X_train"][sorted_ids,:]'
    Y_t_train = fold["Y_t_train"][sorted_ids]
    Y_e_train = fold["Y_e_train"][sorted_ids]


    sorted_ids = sortperm(fold["Y_t_test"])
    X_test = fold["X_test"][sorted_ids,:]'
    Y_t_test = fold["Y_t_test"][sorted_ids]
    Y_e_test = fold["Y_e_test"][sorted_ids]

    mdl_opt = Flux.ADAM(1e-4)
    
    hl1_size = 20
    hl2_size = 20
    mdl = Flux.Chain(Flux.Dense(insize, hl1_size, relu), Flux.Dense(hl1_size, hl2_size, relu), Flux.Dense(hl2_size, 1, sigmoid));
    loss_tr, loss_vld, c_ind_tr, c_ind_vld = train_cphdnn!(mdl, X_train, Y_t_train, Y_e_train, X_test, Y_t_test, Y_e_test;nsteps =nbsteps)
    push!(outs, vec(mdl(X_test)))
    push!(survts, Y_t_test)
    push!(surves, Y_e_test)
    
    fig = Figure();
    ax = Axis(fig[1,1],xlabel = "Nb. of gradient steps", ylabel ="Cox Negative-Likelihood", title = "FOLD: $foldn, sample size: $(size(X_train)[1])")
    lines!(ax, collect(1:length(loss_tr)), Vector{Float32}(loss_tr),color = "blue",label = "training")
    lines!(ax, collect(1:length(loss_vld)), Vector{Float32}(loss_vld),color = "orange",label = "test")
    axislegend(ax, position = :rb)
    fig
    CairoMakie.save("$outpath/training_curve_loss_fold_$foldn.pdf",fig)

    fig = Figure();
    ax = Axis(fig[1,1],xlabel = "Nb. of gradient steps", ylabel ="Concordance index")
    lines!(ax, vcat([1], collect(1000:1000:nbsteps)), Vector{Float32}(c_ind_tr),color = "blue",label = "training")
    lines!(ax, vcat([1], collect(1000:1000:nbsteps)), Vector{Float32}(c_ind_vld),color = "orange",label = "test")
    axislegend(ax, position = :rb)
    fig
    CairoMakie.save("$outpath/training_curve_c_index_fold_$foldn.pdf",fig)
end 
concat_outs = vcat(outs...)
survts = vcat(survts...)
surves = vcat(surves...)
nsamples = length(survts)
cis = []
bootstrapn = 1000
for i in 1:bootstrapn
sampling = rand(1:nsamples, nsamples)
push!(cis, concordance_index(concat_outs[sampling], survts[sampling], surves[sampling]))
end 
sorted_accs = sort(cis)
low_ci, med, upp_ci = sorted_accs[Int(round(bootstrapn * 0.025))], median(sorted_accs), sorted_accs[Int(round(bootstrapn * 0.975))]

concordance_index(concat_outs, survts, surves)

scores = concat_outs
median(scores)
groups = ["low_risk" for i in 1:length(scores)]
high_risk = scores .> median(scores)
low_risk = scores .<= median(scores)
groups[high_risk] .= "high_risk"
p_high, x_high, sc1_high, sc2_high = surv_curve(survts[high_risk], surves[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(survts[low_risk], surves[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low)    
fig = Figure();
ticks = collect(0:250:end_of_study)
Stf_hat_labels = ["$i\n$(label)" for (i,label) in zip(ticks, get_Stf_hat_surv_rates(ticks, sc1_high))] 
ylow = 0.5
lrt_pval = round(log_rank_test(survts, surves, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)

caption = "High risk vs Low risk based on median of risk scores by CPHDNN \n on TCGA BRCA data using 25 PCA from Gene Expression\n Log-rank-test pval = $lrt_pval"
            
ax = Axis(fig[1,1], limits = (0,end_of_study, ylow, 1.05), 
                yminorticksvisible = true, yminorgridvisible = true, yminorticks = IntervalsBetween(2),
                yticks = collect(0:10:100) ./ 100,
                xticks = (ticks, Stf_hat_labels),
                xlabel = "Elapsed time (days)",
                ylabel = "Survival (fraction still alive)",
                titlesize = 14, 
                xticklabelsize =11, 
                yticklabelsize =11, 
                ylabelsize = 14, 
                xlabelsize = 14,
                title = caption)
# plot lines
lines!(ax, sc1_low[sc1_low.e .== 1,:tf], sc1_low[sc1_low.e .== 1, :Stf_hat], color = "blue", label = "low risk (scores < median)") 
conf_tf, lower_95, upper_95 = get_95_conf_interval(sc2_low.tf, sc2_low.nf, sc2_low.Stf_hat, end_of_study)
lines!(ax, conf_tf, upper_95, linestyle = :dot, color = "blue")
lines!(ax, conf_tf, lower_95, linestyle = :dot, color = "blue")
fill_between!(ax, conf_tf, lower_95, upper_95, color = ("blue", 0.1))
# plot censored
# scatter!(ax, sc1_low[sc1_low.e .== 0,:tf], sc1_low[sc1_low.e .== 0, :Stf_hat], marker = [:vline for i in 1:sum(sc1_low.e .== 0)], color = "black")
lines!(ax, sc1_high[sc1_high.e .== 1,:tf], sc1_high[sc1_high.e .== 1, :Stf_hat], color = "red", label = "high risk (scores > median)") 
conf_tf, lower_95, upper_95 = get_95_conf_interval(sc2_high.tf, sc2_high.nf, sc2_high.Stf_hat, end_of_study)
lines!(ax, conf_tf, upper_95, linestyle = :dot, color = "red")
lines!(ax, conf_tf, lower_95, linestyle = :dot, color = "red")
fill_between!(ax, conf_tf, lower_95, upper_95, color = ("red", 0.1))
axislegend(ax, position = :rb, labelsize = 11, framewidth = 0)
                
fig
CairoMakie.save("$outpath/aggregated_scores_high_vs_low_CPHDNN_TCGA_BRCA.pdf",fig)
fig = Figure();
scatter(fig[1,1], ones(sum(high_risk)), log10.(scores[high_risk]), color = "red")
scatter!(fig[1,1], ones(sum(low_risk)), log10.(scores[low_risk]), color = "blue")
fig

fig = Figure();
median(scores)
hist(fig[1,1], log10.(scores[high_risk]), color = "red", bins = collect(-30:0.2:0))
hist!(fig[1,1], log10.(scores[low_risk]), color = "blue", bins = collect(-30:0.2:0))
fig
mean(scores[low_risk])
mean(scores[high_risk])
scores = vec(cpu(mdl(X)))
median(scores)
high_risk = scores .> median(scores)
low_risk = scores .<= median(scores)
cpu(Y_e)[high_risk]
cpu(Y_t)[high_risk]
cpu(Y_e)[low_risk]
p_high, x_high, sc1_high, sc2_high = surv_curve(cpu(Y_t)[high_risk], cpu(Y_e)[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(cpu(Y_t)[low_risk], cpu(Y_e)[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low)            
