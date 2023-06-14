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

function surv_curve(survt, surve)
    
    ratio_table = 
    return ratio_table
end 
brca_prediction.subgroups
group = "HER2-enriched" 
cohort = findall(brca_prediction.subgroups .== group)
survt = brca_prediction.survt
surve = brca_prediction.surve
surv_curv = surv_curve(survt[cohort], surve[cohort])
function surv_curve(survt, surve; color="black")
    curve = zeros((length(survt), 3))
    ranking = sortperm(survt)
    survt = survt[ranking]
    surve = surve[ranking] 
    riskset = reverse(collect(1:length(survt)) .- 1) ./ length(survt)
    for i in 1:size(survt)[1]
        curve[i,:] .= [survt[i], surve[i], riskset[i]]
    end 
    surv_curv = DataFrame(:t=>curve[:,1],:e=> curve[:,2], :R=>curve[:,3])
    surv_curv_1_1 = surv_curv[findall(surv_curv[:,"e"] .== 1),:]
    surv_curv_1_2 = surv_curv_1_1[2:size(surv_curv_1_1)[1],:]
    surv_curv_1_2.t = surv_curv_1_1[1:size(surv_curv_1_1)[1]-1,"t"]
    surv_curv_1 = append!(surv_curv_1_1, surv_curv_1_2 )
    surv_curv_1 = surv_curv_1[sortperm(surv_curv_1.t),:]
    # remove last data (undefined)
    # surv_curv_1 = surv_curv_1[1:end-1,:]
    # add (t = 0, S = 1), (t = 0, S = 1 - R(0,1))
    surv_curv_1 = append!(DataFrame(Dict(:t=>[0.0,surv_curv_1_1.t[1]],:e=>[1.0,1.0],:R=>[1.0,1.0])), surv_curv_1)
    surv_curv_2 = surv_curv[findall(surv_curv[:,"e"] .== 0),:]
    for (i,t) in enumerate(surv_curv_1.t)
        surv_curv_2[findall(surv_curv_2.t .>= t),"R"] .= surv_curv_1.R[i]
    end 
    survt = AlgebraOfGraphics.data(surv_curv_1) * mapping(:t,:R) * visual(Lines, color = color)
    censored = AlgebraOfGraphics.data(surv_curv_2) * mapping(:t,:R) * visual(marker = [:x for i in 1:size(surv_curv_2)[1]])

    return survt, censored, surv_curve 
end         
fig
function extract_surv_curves(survt, surve, subgroups)
    curves = []
    colors = ["red","blue","green","purple","grey","black", "yellow","orange"]
    for (i,group) in enumerate(unique(subgroups))
        cohort = findall(subgroups .== group)
        p, x, surv_curv = surv_curve(survt[cohort],surve[cohort];color=colors[i])
        fig = draw(p + x ,axis = (;xlabel = "Elpased Time (days)", ylabel = "Survival (% alive)", title =  group, limits = (0,7200,0,1)))
        push!(curves, fig)
    end
    return curves
end 

max(brca_prediction.survt...)
surv_curves = extract_surv_curves(brca_prediction.survt, brca_prediction.surve, brca_prediction.subgroups)
for i in 1:5
    CairoMakie.save("RES/SURV/tmp$i.png", surv_curves[i])
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

