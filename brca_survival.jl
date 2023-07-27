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
println(sort(names(CLIN_FULL)))
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
max(TCGA_all.data...)

BRCA_gdc_ids = findall([ in(x, BRCA_CLIN.case_id) for x in  TCGA_all.rows])
BRCA_ids = TCGA_all.rows[BRCA_gdc_ids]
BRCA_CLIN = BRCA_CLIN[sortperm(BRCA_CLIN.case_id),:]
names(BRCA_CLIN)
BRCA_CLIN[BRCA_CLIN[:,"PAM50 mRNA"] .== "NA","case_id"]
names(clin_data)
clin_data[:,"Complete TCGA ID"]
CLIN[CLIN.case_id .== "001cef41-ff86-4d3f-a140-a647ac4b10a1",:]

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
sort(BRCA_ids)
TCGA_BRCA_TPM_surv = GDC_data_surv(
    TCGA_all.data[BRCA_gdc_ids[sortperm(BRCA_ids)],:], # expression
    sort(BRCA_ids), 
    TCGA_all.cols, 
    Array{String}(BRCA_CLIN[:,"PAM50 mRNA"]), 
    survt, 
    surve) 



write_h5(TCGA_BRCA_TPM_surv, "Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5")


#### plot surv curves
figures = plot_brca_subgroups(brca_prediction, unique(brca_prediction.subgroups), outpath; end_of_study = end_of_study);

### ALL samples ###
### ANALYSIS using PCA Gene Expression COXPH
sum(BRCA_CLIN.survt .== brca_prediction.survt)
sum(BRCA_CLIN[:,"PAM50 mRNA"] .== brca_prediction.subgroups)
names(BRCA_CLIN)
BRCA_CLIN = BRCA_CLIN[:,1:14]
BRCA_CLIN = innerjoin(BRCA_CLIN, tmp, on = :case_id)

include("SurvivalDev.jl")

##### DATA loading
#brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);
highv = reverse(sortperm([var(x) for x in 1:size(brca_prediction.data)[2]]))
highv_25 = highv[1:Int(floor(length(highv)*0.25))]
brca_pred_subset = GDC_data_surv(brca_prediction.data[:,highv_25], brca_prediction.rows, brca_prediction.cols[highv_25], brca_prediction.subgroups, brca_prediction.survt, brca_prediction.surve)

##### PCA
using MultivariateStats
max(brca_pred_subset.data'...)
pca = fit(PCA, brca_pred_subset.data', maxoutdim=25, method = :cov);
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
