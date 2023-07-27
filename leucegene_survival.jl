# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("lgn_data_processing.jl")
### loading data 
basepath = "/u/sauves/MTL_cancer_GE/"
clinical_fname = "$(basepath)/Data/LEUCEGENE/lgn_pronostic_CF"
ge_cds_fname = "$(basepath)/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
ge_lsc17_fname = "$(basepath)/Data/SIGNATURES/LSC17_lgn_pronostic_expressions.csv"

cf = CSV.read(clinical_fname, DataFrame)
interest_groups = [get_interest_groups(g) for g  in cf[:, "WHO classification"]]
cf.interest_groups = interest_groups
ge_cds_raw_data = CSV.read(ge_cds_fname, DataFrame)
lsc17 = CSV.read(ge_lsc17_fname, DataFrame)
ge_cds_all = log_transf_high_variance(ge_cds_raw_data, frac_genes=0.50, avg_norm = false)
ge_cds_all.data
lgn_prediction = GDC_data_surv(ge_cds_all.data, ge_cds_all.factor_1, ge_cds_all.factor_2, interest_groups, cf[:,"Overall_Survival_Time_days"], cf[:,"Overall_Survival_Status"])
lgn_lsc17_prediction = GDC_data_surv(Matrix(lsc17[:,2:end]), vec(lsc17[:,1]),vec(names(lsc17)[2:end]) , interest_groups, cf[:,"Overall_Survival_Time_days"], cf[:,"Overall_Survival_Status"])
#folds = split_train_test(lgn_prediction.data,lgn_prediction.survt,lgn_prediction.surve, lgn_prediction.rows;nfolds = 5)
folds =  split_train_test(lgn_lsc17_prediction.data,lgn_lsc17_prediction.survt,lgn_lsc17_prediction.surve, lgn_lsc17_prediction.rows;nfolds = 5)
folds[1]

params = Dict("insize"=>size(lgn_lsc17_prediction.data)[2],
    "hl1_size" => 20,
    "hl2_size" => 20,
    "acto"=>identity,
    "nbsteps" => 30_000,
    "wd" => 1e-3
    )

mdl = build_cphdnn(params)

outs, survts, surves = validate_cphdnn(params, folds;device = cpu)

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
            # concordant
            if S[j] < S[i]
                numerator += 1
                denominator += 1
            elseif S[j] > S[i]
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
    return numerator / denominator 
end
order = sortperm(lgn_prediction.survt)

T = lgn_prediction.survt[order]
E = lgn_prediction.surve[order]
S = vcat(collect(1:Int(floor(length(T)/1.25))), shuffle(collect(Int(floor(length(T)/1.25))+1:length(T))))
c_index_dev(T,E,S)

concordant_pairs = S .> S'
tied_pairs = vec(sum(S .== S', dims = 1)') - ones(length(S))
sum(tied_pairs)
admissable_pairs = T .< T'
c_ind = sum(E' .* admissable_pairs .* concordant_pairs) + 0.5 * sum(tied_pairs)
c_ind = c_ind / sum(E .* vec(sum(admissable_pairs, dims = 1)') .+ sum(tied_pairs))

concordance_index(lgn_prediction.survt, lgn_prediction.survt, lgn_prediction.surve)