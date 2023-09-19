# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("cross_validation.jl")
outpath, session_id = set_dirs() 
infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
brca_prediction = BRCA_data(infile,minmax_norm = true)

##### PCA
using MultivariateStats
maximum(brca_prediction.data')
pca = fit(PCA, brca_prediction.data', maxoutdim=25, method = :cov);
brca_prediction.stage
brca_prediction_pca = predict(pca, brca_pred_subset.data')';
brca_pca_25_df = DataFrame(Dict([("pc_$i",brca_prediction_pca[:,i]) for i in 1:size(brca_prediction_pca)[2]]))
brca_pca_25_df[:,"case_id"] .= case_ids 
CSV.write("RES/SURV/TCGA_BRCA_pca_25_case_ids.csv", brca_pca_25_df)
CSV.write("RES/SURV/TCGA_BRCA_clinical_survival.csv", BRCA_CLIN)
# CSV tcga brca surv + clin data 
brca_pca = CSV.read("RES/SURV/TCGA_brca_pca_25_case_ids.csv", DataFrame)

brca_pca[:,"is_her2"] .= subgroups .== "HER2-enriched"
brca_pca[:,"is_lumA"] .= subgroups .== "Luminal A"
brca_pca[:,"is_lumB"] .= subgroups .== "Luminal B"
brca_pca[:,"is_bsllk"] .= subgroups .== "Basal-like"
brca_pca[:,"survt"] .= brca_prediction.survt ./ 10_000
brca_pca[:,"surve"] .= brca_prediction.surve


folds = split_train_test(Matrix(brca_pca[:,collect(1:25)]), brca_prediction.survt, brca_prediction.surve, vec(brca_pca[:,26]);nfolds =5)

params = Dict("insize"=>size(folds[1]["X_train"])[2],
    "hl1_size" => 20,
    "hl2_size" => 20,
    "acto"=>sigmoid,
    "nbsteps" => 10_000,
    "wd" => 1e-3
    )

mdl = build_cphdnn(params)
mdl = build_ae(params)

min(cpu(mdl(gpu(Matrix(folds[1]["X_train"]'))))...)
include("SurvivalDev.jl")


validate_cphdnn(params, folds;device =cpu)

fold = folds[1]
mdl = build_cphdnn(params)
sorted_ids = sortperm(fold["Y_t_train"])
X_train = gpu(Matrix(fold["X_train"][sorted_ids,:]'))
Y_t_train = gpu(fold["Y_t_train"][sorted_ids])
Y_e_train = gpu(fold["Y_e_train"][sorted_ids])
NE_frac_tr = sum(Y_e_train .== 1) != 0 ? 1 / sum(Y_e_train .== 1) : 0
lossf(mdl, X_train, Y_e_train, NE_frac_tr, wd)
concordance_index(vec(cpu(mdl(X_train))), cpu(Y_t_train), cpu(Y_e_train))

sorted_ids = gpu(sortperm(fold["Y_t_test"]))
X_test = gpu(Matrix(fold["X_test"][sorted_ids,:]'))
Y_t_test = gpu(fold["Y_t_test"][sorted_ids])
Y_e_test = gpu(fold["Y_e_test"][sorted_ids])



mdl = build_cphdnn(params)    
# cpu(vec(mdl(X_train)))
# X_train[:,end-3:end]
# concordance_index(vec(mdl(X_train)), Y_t_train, Y_e_train)
# NE_frac_tr = sum(Y_e_train .== 1) != 0 ? 1 / sum(Y_e_train .== 1) : 0 
# lossf(mdl, X_train, Y_e_train, NE_frac_tr, wd)

ltr, lvld, c_tr, c_vld = train_cphdnn!(mdl, X_train, Y_t_train, Y_e_train, X_test, Y_t_test, Y_e_test;nsteps =nbsteps,wd=wd)
mdl[1].weight
push!(outs, vec(cpu(mdl(X_test))) )
push!(survts, cpu(Y_t_test))
push!(surves, cpu(Y_e_test))



scores = vec(cpu(mdl(X_test)))
median(scores)
groups = ["low_risk" for i in 1:length(scores)]
high_risk = scores .> median(scores)
low_risk = scores .<= median(scores)
median(scores[high_risk])
median(scores[low_risk])


sum(high_risk)
groups[high_risk] .= "high_risk"
p_high, x_high, sc1_high, sc2_high = surv_curve(cpu(Y_t_test)[high_risk], cpu(Y_e_test)[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(cpu(Y_t_test)[low_risk], cpu(Y_e_test)[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low) 


concat_outs = vcat(outs...)
concat_survts = vcat(survts...)
concat_surves = vcat(surves...)
nsamples = length(concat_survts)
cis = []
bootstrapn = 1000
for i in 1:bootstrapn
sampling = rand(1:nsamples, nsamples)
push!(cis, concordance_index(concat_outs[sampling], concat_survts[sampling], concat_surves[sampling]))
end 
sorted_accs = sort(cis)
low_ci, med, upp_ci = sorted_accs[Int(round(bootstrapn * 0.025))], median(sorted_accs), sorted_accs[Int(round(bootstrapn * 0.975))]

concordance_index(concat_outs, concat_survts, concat_surves)

scores = concat_outs
median(scores)
groups = ["low_risk" for i in 1:length(scores)]
high_risk = scores .> median(scores)
low_risk = scores .<= median(scores)
groups[high_risk] .= "high_risk"
p_high, x_high, sc1_high, sc2_high = surv_curve(concat_survts[high_risk], concat_surves[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(concat_survts[low_risk], concat_surves[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low)    
fig = Figure();
ticks = collect(0:250:end_of_study)
Stf_hat_labels = ["$i\n$(label)" for (i,label) in zip(ticks, get_Stf_hat_surv_rates(ticks, sc1_high))] 
ylow = 0.2
lrt_pval = round(log_rank_test(concat_survts, concat_surves, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)

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
lines!(ax, Array{Int64}(sc1_low[sc1_low.e .== 1,:tf]), sc1_low[sc1_low.e .== 1, :Stf_hat], color = "blue", label = "low risk (scores < median)") 
conf_tf, lower_95, upper_95 = get_95_conf_interval(sc2_low.tf, sc2_low.nf, sc2_low.Stf_hat, end_of_study)
lines!(ax, Array{Int64}(conf_tf), upper_95, linestyle = :dot, color = "blue")
lines!(ax, Array{Int64}(conf_tf), lower_95, linestyle = :dot, color = "blue")
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
hist(fig[1,1], scores[high_risk], color = "red")
hist(fig[1,1], scores[low_risk], color = "blue")
hist(fig[1,1], concat_outs)

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
