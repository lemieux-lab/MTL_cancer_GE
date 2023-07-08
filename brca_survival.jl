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
end_of_study = 365 * 5 # 10 years 
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
brca_pca.case_id .== brca_prediction.rows
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
cox_nll(brca_prediction.survt, brca_prediction.surve, outs)
mdl_opt = Flux.ADAM(1e-3)
mdl = gpu(Flux.Chain(Flux.Dense(25, 10, relu), Flux.Dense(10, 1, identity)));
lossf(model, X, t, e, nE, wd) = cox_nll_vec(t, e, vec(model(X)), nE) + wd * l2_penalty(model)
sorted_ids = sortperm(brca_prediction.survt)
X = gpu(Matrix(brca_pca[:,1:25][sorted_ids,:])')
Y_t = gpu(brca_prediction.survt[sorted_ids])
Y_e = gpu(brca_prediction.surve[sorted_ids])
lossval = lossf(mdl, X, Y_t,Y_e, sum(E .==1))
ps = Flux.params(mdl)
gs = gradient(ps) do 
    lossf(mdl, X, Y_t,Y_e, sum(e .== 1))    
end 
wd = 1e-3
for step in 1:20_000
    lossval = lossf(mdl, X, Y_t,Y_e, sum(e .== 1),wd)
    if step % 1000 == 0
        println(lossval)
    end
    ps = Flux.params(mdl)
    gs = gradient(ps) do 
        lossf(mdl, X, Y_t,Y_e, sum(e .== 1), wd)    
    end 
    Flux.update!(mdl_opt, ps, gs)
end 

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
