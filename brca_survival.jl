# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
using Distributions
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

brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true)

unique(brca_prediction.subgroups)
group = "Luminal A"
cohort = findall(brca_prediction.subgroups .== group)
survt = brca_prediction.survt[cohort]
surve = brca_prediction.surve[cohort]
max(brca_prediction.survt...)

##### DEV
include("SurvivalDev.jl")
subgroups = brca_prediction.subgroups
end_of_study = 365 * 5 # 5 years 
survt = brca_prediction.survt
surve = brca_prediction.surve
surve[survt .>= end_of_study] .= 0 
survt[survt .>= end_of_study] .= end_of_study 

get_Stf_hat_surv_rates(ticks, sc1)  = [round(Float32(sc1[findall(sc1.tf .>= i),"Stf_hat"][1]);digits=3) for i in ticks]
ticks = collect(0:250:end_of_study)
ylow = 0.5

colors = ["red","blue","green","purple", "magenta","orange","yellow","grey", "black"]
groups = unique(subgroups)
for group_of_interest in 1:length(groups)
fig = Figure(resolution =  (1000,1000));
grid = fig[1,1] = GridLayout()
axes = [grid[row,col] for row in 1:3 for col in 1:2]
for i in 1:size(groups)[1]
    comp_groups = [groups[group_of_interest], groups[i]]
    comp_cohort = findall(brca_prediction.subgroups .== comp_groups[1] .|| brca_prediction.subgroups .== comp_groups[2])
    comp_survt= brca_prediction.survt[comp_cohort]
    comp_surve= brca_prediction.surve[comp_cohort]
    comp_subgroups = brca_prediction.subgroups[comp_cohort]
    lrt_pval = round(log_rank_test(comp_survt, comp_surve, comp_subgroups, comp_groups); digits = 5)

    group = groups[group_of_interest]
    cohort = findall(subgroups .== group)
    # first subgroup
    p, x, sc1_ctrl, sc2_ctrl = surv_curve(survt[cohort], surve[cohort]; color = colors[i])
    p, x, sc1, sc2 = surv_curve(survt[findall(subgroups .== groups[i])], surve[findall(subgroups .== groups[i])]; color = colors[i])
    sc1_ctrl = sc1_ctrl[sortperm(sc1_ctrl.tf),:]
    # plot lines
    Stf_hat_labels = ["$i\n$(label)" for (i,label) in zip(ticks, get_Stf_hat_surv_rates(ticks, sc1_ctrl))]    
    Stf_hat_labels = ["$tick\n$(label)" for (tick,label) in zip(Stf_hat_labels, get_Stf_hat_surv_rates(ticks, sc1))]       
    ax = Axis(axes[i], limits = (0,end_of_study, ylow, 1.05), 
        yminorticksvisible = true, yminorgridvisible = true, yminorticks = IntervalsBetween(2),
        yticks = collect(0:10:100) ./ 100,
        xticks = (ticks, Stf_hat_labels),
        xlabel = "Elapsed time (days)",
        ylabel = "Survival (fraction still alive)",
        title = "Survival curves $group (n=$(length(cohort)),c=$(sum(surve[cohort].==0))) \nvs $(groups[i]) (n=$(length(findall(subgroups .== groups[i]))),c=$(sum(surve[findall(subgroups .== groups[i])].==0)))\nLog-Rank Test pval: $lrt_pval")
    
    lines!(ax, sc1_ctrl[sc1_ctrl.e .== 1,:tf], sc1_ctrl[sc1_ctrl.e .== 1, :Stf_hat], color = "grey", label = groups[group_of_interest]) 
    # plot censored
    scatter!(ax, sc1_ctrl[sc1_ctrl.e .== 0,:tf], sc1_ctrl[sc1_ctrl.e .== 0, :Stf_hat], marker = [:vline for i in 1:sum(sc1_ctrl.e .== 0)], color = "black")

    if i != group_of_interest
    # second subgroup
    sc1 = sc1[sortperm(sc1.tf),:] 
    # plot lines
    lines!(ax, sc1[sc1.e .== 1,:tf], sc1[sc1.e .== 1, :Stf_hat], color = colors[i], label = groups[i]) 
    # plot censored
    scatter!(ax, sc1[sc1.e .== 0,:tf], sc1[sc1.e .== 0, :Stf_hat], marker = [:vline for i in 1:sum(sc1.e .== 0)], color = "black")
    axislegend(ax, position = :rb)

end
end
CairoMakie.save("RES/SURV/surv_curves_$(groups[group_of_interest])_1v1.pdf", fig)
end
groups = ["Luminal B", "Luminal A"]
cohort = findall(brca_prediction.subgroups .== groups[1] .|| brca_prediction.subgroups .== groups[2])
survt= brca_prediction.survt[cohort]
surve= brca_prediction.surve[cohort]
subgroups = brca_prediction.subgroups[cohort]
log_rank_test(survt, surve, subgroups, groups)

function log_rank_test(survt, surve, subgroups, groups)
# LOG RANK test 
end_of_study = 365 * 5 # 5 years 
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



#end
surv_curves = extract_surv_curves(brca_prediction.survt, brca_prediction.surve, brca_prediction.subgroups);

for (i, curv) in enumerate(surv_curves)
    CairoMakie.save("RES/SURV/surv_curve_$(unique(brca_prediction.subgroups)[i]).pdf", curv[1])
    CairoMakie.save("RES/SURV/surv_curve_$(unique(brca_prediction.subgroups)[i]).png", curv[1])
    CSV.write("RES/SURV/surv_curve_$(unique(brca_prediction.subgroups)[i]).csv", curv[2])
    CSV.write("RES/SURV/surv_curve_$(unique(brca_prediction.subgroups)[i])_curve.csv", curv[3])
end
### ALL samples ###
p, x, sc1, sc2 = surv_curve(brca_prediction.survt, brca_prediction.surve; color = "black")
end_of_study = max(brca_prediction.survt...)
Stf_hat_labels = [round(Float32(sc1[findall(sc1.tf .>= i)[1],"Stf_hat"][1]);digits=3) for i in 0:1000:max(sc1.tf...)]
fig = draw(p + x ,
            axis = (;xlabel = "Elpased Time (days)", ylabel = "Survival (fraction alive)", title =  "ALL samples, (n=$(size(brca_prediction.survt)[1]), c=$(sum(brca_prediction.surve .== 0)))", 
                limits = (0,end_of_study,0,1), yminorticksvisible = true, yminorgridvisible = true, yminorticks = IntervalsBetween(2),
                yticks = collect(0:10:100) ./ 100,
                xticks = (collect(0:1000:max(sc1.tf...)), ["$i\n$(label)" for (i,label) in zip(0:1000:max(sc1.tf...),Stf_hat_labels)]),
                )
            )
CairoMakie.save("RES/SURV/surv_curve_all.pdf", fig)
CairoMakie.save("RES/SURV/surv_curve_all.png", fig)
CSV.write("RES/SURV/surv_curve_all.csv", sc1)
CSV.write("RES/SURV/surv_curve_all_curve.csv", sc2)
                

survt, surve = brca_prediction.survt, brca_prediction.surve

lumA = lumA[sortperm(lumA.survt),:]
length(brca_prediction.survt)

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

