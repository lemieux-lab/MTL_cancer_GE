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
folds[1]["X_train"]
# regular COXPH
cph_params =  Dict("insize"=>size(lgn_lsc17_prediction.data)[2],
"nbsteps" => 30_000,
"wd" => 1e-3,
"lr" => 1e-3
)
function build_cph(params)
    mdl = Flux.Dense(params["insize"], 1, identity)
    return mdl
end 
#Flux.params(mdl)[1] .= zeros(1,17)
# train
for (foldi, fold) in enumerate(folds) 
    mdl = build_cph(cph_params)
    mdl.weight .= zeros(1,17)
    opt = Flux.Adam(cph_params["lr"])
    ps = Flux.params(mdl)

    order = sortperm(folds[foldi]["Y_t_train"])
    X_train = Matrix(folds[foldi]["X_train"][order,:]')
    Yt_train = folds[foldi]["Y_t_train"][order]
    Ye_train = folds[foldi]["Y_e_train"][order]
    NE_frac_tr = sum(Ye_train .== 1) != 0 ? 1 / sum(Ye_train .== 1) : 0
        
    order = sortperm(folds[foldi]["Y_t_test"])
    X_test = Matrix(folds[foldi]["X_test"][order,:]')
    Yt_test = folds[foldi]["Y_t_test"][order]
    Ye_test = folds[foldi]["Y_e_test"][order]
    NE_frac_tst = sum(Ye_test .== 1) != 0 ? 1 / sum(Ye_test .== 1) : 0
        
    for i in 1:5000
        mdl(X_train)
        gs = gradient(ps) do 
            cox_nll(mdl,X_train, Ye_train, NE_frac_tr) + cph_params["wd"] * sum(abs2, mdl.weight)
        end
        Flux.update!(opt, ps, gs)
        if i % 100 == 0 || i == 1 
            lossv_tr = round(cox_nll(mdl,X_train, Ye_train, NE_frac_tr), digits = 3)
            cind_tr = round(c_index_dev(Yt_train, Ye_train,vec(mdl(X_train))), digits = 3)
            lossv_tst = round(cox_nll(mdl,X_test, Ye_test, NE_frac_tst), digits = 3)
            cind_tst = round(c_index_dev(Yt_test, Ye_test,vec(mdl(X_test))), digits = 3)
            
            println("$i TRAIN loss: $lossv_tr, c_ind : $cind_tr TEST loss: $lossv_tst, c_ind: $cind_tst")
        end 
    end 
end
foldi = 1
order = reverse(sortperm(folds[foldi]["Y_t_train"]))
X_train = Matrix(folds[foldi]["X_train"][order,:]')
Yt_train = folds[foldi]["Y_t_train"][order]
Ye_train = folds[foldi]["Y_e_train"][order]
NE_frac_tr = sum(Ye_train .== 1) != 0 ? 1 / sum(Ye_train .== 1) : 0
    
mdl = build_cph(cph_params)
mdl.weight .= zeros(1,17)
opt = Flux.Adam(cph_params["lr"])
ps = Flux.params(mdl)

mdl(X_train)
gs = gradient(ps) do 
    cox_nll(mdl,X_train, Ye_train, NE_frac_tr)
end
fig = Figure();
ax = Axis(fig[1,1], title = "histogram of scores")
hist!(ax, vec(mdl(X_train)))
fig
params = Dict("insize"=>size(lgn_lsc17_prediction.data)[2],
    "hl1_size" => 20,
    "hl2_size" => 20,
    "acto"=>identity,
    "nbsteps" => 30_000,
    "wd" => 1e-3
    )

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