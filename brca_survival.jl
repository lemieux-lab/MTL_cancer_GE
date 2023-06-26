# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")

brca_prediction = GDC_data("Data/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);

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
function  GDC_data_surv(inf::String;log_transf = false)
    f = h5open(inf, "r")
    TPM_data = f["data"][:,:]
    if log_transf
        TPM_data = log10.(TPM_data .+ 1)
    end 

    case_ids = f["rows"][:]
    gene_names = f["cols"][:]
    survt = f["survt"][:]
    surve = f["surve"][:]
    subgroups = f["subgroups"][:]
    close(f)
    return GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve)
end
brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true)

unique(brca_prediction.subgroups)
group = "Luminal A"
cohort = findall(brca_prediction.subgroups .== group)
survt = brca_prediction.survt[cohort]
surve = brca_prediction.surve[cohort]

function surv_curve(survt, surve; color="black")
    
    ordered_failure_times = vcat([0], sort(survt[findall(surve .== 1)]), [end_of_study])
    qfs = []
    nfs = [length(survt)]
    Stf_hat = [1.]
    for (f, tf) in enumerate(ordered_failure_times[1:end - 1])
        push!(qfs, sum(surve[findall(survt .> tf .&& survt .< ordered_failure_times[f + 1])] .== 0) )
        if f > 1
            push!(nfs, nfs[f - 1] - qfs[f - 1] - 1) # Risk set nb indiv. at risk before time tf
            push!(Stf_hat, Stf_hat[f - 1] * (nfs[f] - 1) / nfs[f])
        end 
        println([f - 1, tf, qfs[f], nfs[f], Stf_hat[f]])
    end
    censored_tf = survt[surve .== 0]
    censored_Shat = []
    for tf in censored_tf
        Shat_position = Stf_hat[argmax(ordered_failure_times[ordered_failure_times .<= tf])]
        push!(censored_Shat, Shat_position)
    end
    
    surv_curv_1_1 = DataFrame(:tf=>ordered_failure_times[1:end-1], :Stf_hat=>Stf_hat)
    surv_curv_1_2 = DataFrame(:tf=>ordered_failure_times[2:end], :Stf_hat=>Stf_hat)
    surv_curv_1 = append!(surv_curv_1_2, surv_curv_1_1)
    surv_curv_1 = surv_curv_1[sortperm(surv_curv_1.tf),:]
    surv_curv_2 = DataFrame(:tf => censored_tf, :Stf_hat=>censored_Shat)
    survt = AlgebraOfGraphics.data(surv_curv_1) * mapping(:tf, :Stf_hat) * visual(Lines, color = color, linewidth = 3)
    censored = AlgebraOfGraphics.data(surv_curv_2) * mapping(:tf, :Stf_hat) * visual(marker = [:vline for i in 1:size(surv_curv_2)[1]])
    return survt, censored
end 

function extract_surv_curves(survt, surve, subgroups)
    end_of_study = max(survt...) + 1
    curves = []
    colors = ["red","blue","green","purple","grey","black", "yellow","orange"]
    for (i,group) in enumerate(unique(subgroups))
        cohort = findall(subgroups .== group)
        p, x  = surv_curve(survt[cohort],surve[cohort];color=colors[i])
        fig = draw(p + x ,axis = (;xlabel = "Elpased Time (days)", ylabel = "Survival (% alive)", title =  group, 
            limits = (0,end_of_study,0,1), yminorticksvisible = true, yminorgridvisible = true, yminorticks = IntervalsBetween(2),
            yticks = collect(0:10:100) ./ 100,
            xticks = collect(0:1000:end_of_study)))
        
        push!(curves, fig)
    end
    return curves
end 
surv_curves = extract_surv_curves(brca_prediction.survt, brca_prediction.surve, brca_prediction.subgroups)

for (i, curv) in enumerate(surv_curves)
    CairoMakie.save("RES/SURV/surv_curve_$(unique(brca_prediction.subgroups)[i]).pdf", curv)
end


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
CSV.write("RES/SURV/TCGA_brca_pca_25_case_ids.csv", brca_pca_25_df)
CSV.write("RES/SURV/TCGA_brca_clinical_survival.csv", BRCA_CLIN)
# CSV tcga brca surv + clin data 

