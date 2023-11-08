# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("lgn_data_processing.jl")
include("cross_validation.jl")
### loading data 
outpath, session_id = set_dirs() 
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

#lgn_prediction = GDC_data_surv(ge_cds_all.data, ge_cds_all.factor_1, ge_cds_all.factor_2, interest_groups, cf[:,"Overall_Survival_Time_days"], cf[:,"Overall_Survival_Status"])
ge_cds_all
y_lbls = cf[:, "WHO classification"]
names(cf)
dat = Matrix(ge_cds_raw_data[:,2:end])
gens =names(ge_cds_raw_data)[2:end]
ndat, ngens = minmaxnorm(dat, gens)
lgn_prediction = GDC_data(ndat, Array(ge_cds_raw_data[:,1]), ngens, y_lbls)
#lgn_lsc17_prediction = GDC_data_surv(Matrix(lsc17[:,2:end]), vec(lsc17[:,1]),vec(names(lsc17)[2:end]) , interest_groups, cf[:,"Overall_Survival_Time_days"], cf[:,"Overall_Survival_Status"])

y_data = label_binarizer(y_lbls)
x_data = lgn_prediction.data

######### Leucegene AML cancer data
###### Proof of concept with Auto-Encoder classifier DNN. Provides directed dimensionality reductions
## 1 CLFDNN-AE 2D vs random x10 replicat accuracy BOXPLOT, train test samples by class bn layer 2D SCATTER. 
nepochs = 3000
nfolds, ae_nb_hls, dim_redux = 5, 1, 2
lgn_aeclfdnn_params = Dict("model_title"=>"AE_CLF_LGN_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
"model_type" => "aeclfdnn", "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(lgn_prediction.rows) - Int(round(length(lgn_prediction.rows) / nfolds)), "nsamples_test" => Int(round(length(lgn_prediction.rows) / nfolds)),
"nsamples" => length(lgn_prediction.rows) , "insize" => length(lgn_prediction.cols), "ngenes" => length(lgn_prediction.cols),  
"nfolds" => 5,  "nepochs" => nepochs, "ae_lr" => 1e-4, "wd" => 1e-3, "dim_redux" => dim_redux, 
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"clfdnn_lr" => 1e-3, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2], "model_cv_complete" => false)
lgn_clf_cb = dump_aeclfdnn_model_cb(1000, y_lbls, export_type = "pdf")

lgn_prediction.data

model = build(lgn_aeclfdnn_params;adaptative=false)
bmodel, bmfold, outs_test, y_test, outs_train, y_train = validate_aeclfdnn!(lgn_aeclfdnn_params, x_data, y_data, lgn_prediction.rows, lgn_clf_cb)





folds =  split_train_test(lgn_lsc17_prediction.data,lgn_lsc17_prediction.survt,lgn_lsc17_prediction.surve, lgn_lsc17_prediction.rows;nfolds = 5)
folds[1]["X_train"]
# regular COXPH
cph_params =  Dict("insize"=>size(lgn_lsc17_prediction.data)[2],
"nbsteps" => 3000,
"wd" => 1e-3,
"lr" => 1e-3
)
function build_cph(params)
    mdl = Flux.Dense(params["insize"], 1, identity)
    return mdl
end 
#Flux.params(mdl)[1] .= zeros(1,17)
# train

outs = []
survts = []
surves = []
for (foldi, fold) in enumerate(folds) 
    loss_tr = []
    loss_vld = []
    c_ind_tr = [] 
    c_ind_vld = []
    mdl = build_cph(cph_params)
    mdl.weight .= zeros(1,17)
    opt = Flux.Adam(cph_params["lr"])
    ps = Flux.params(mdl)

    orderl = sortperm(folds[foldi]["Y_t_train"])
    X_train = Matrix(folds[foldi]["X_train"][orderl,:]')
    Yt_train = folds[foldi]["Y_t_train"][orderl]
    Ye_train = folds[foldi]["Y_e_train"][orderl]
    #NE_frac_tr = sum(Ye_train .== 1) != 0 ? 1 / sum(Ye_train .== 1) : 0
        
    orderl = sortperm(folds[foldi]["Y_t_test"])
    X_test = Matrix(folds[foldi]["X_test"][orderl,:]')
    Yt_test = folds[foldi]["Y_t_test"][orderl]
    Ye_test = folds[foldi]["Y_e_test"][orderl]
    #NE_frac_tst = sum(Ye_test .== 1) != 0 ? 1 / sum(Ye_test .== 1) : 0
        
    for i in 1:cph_params["nbsteps"]
        gs = gradient(ps) do 
            cox_nll(vec(mdl(X_train)), Ye_train) + cph_params["wd"] * sum(abs2, mdl.weight)
        end
        Flux.update!(opt, ps, gs)
        if i % 100 == 0 || i == 1 
            lossv_tr = round(cox_nll(vec(mdl(X_train)), Ye_train), digits = 3)
            cind_tr = round(c_index_dev(Yt_train, Ye_train, -1 .* vec(mdl(X_train))), digits = 3)
            lossv_tst = round(cox_nll(vec(mdl(X_test)), Ye_test), digits = 3)
            cind_tst = round(c_index_dev(Yt_test, Ye_test, -1 .* vec(mdl(X_test))), digits = 3)
            push!(loss_tr, lossv_tr)
            push!(loss_vld, lossv_tst)
            push!(c_ind_tr, cind_tr)
            push!( c_ind_vld,cind_tst)
            println("FOLD $foldi - $i TRAIN loss: $lossv_tr, c_ind : $cind_tr TEST loss: $lossv_tst, c_ind: $cind_tst")
        end 
    end 
    fig = Figure();
    ax = Axis(fig[1,1],xlabel = "Nb. of gradient steps", ylabel ="Cox Negative-Likelihood", title = "FOLD: $foldi, sample size: $(size(X_train)[1])")
    lines!(ax, collect(1:length(loss_tr)), Vector{Float32}(loss_tr),color = "blue",label = "training")
    lines!(ax, collect(1:length(loss_vld)), Vector{Float32}(loss_vld),color = "orange",label = "test")
    axislegend(ax, position = :rb)

    CairoMakie.save("$outpath/training_curve_loss_fold_$foldi.pdf",fig)
    fig = Figure();
    ax = Axis(fig[1,1],xlabel = "Nb. of gradient steps", ylabel ="Concordance index", limits = (0,cph_params["nbsteps"],0,1))
    lines!(ax, vcat([1], collect(100:100:cph_params["nbsteps"])), Vector{Float32}(c_ind_tr),color = "blue",label = "training")
    lines!(ax, vcat([1], collect(100:100:cph_params["nbsteps"])), Vector{Float32}(c_ind_vld),color = "orange",label = "test")
    axislegend(ax, position = :rb)

    CairoMakie.save("$outpath/training_curve_c_index_fold_$foldi.pdf",fig)
    push!(outs, -1 .* vec(cpu(mdl(X_test))) )
    push!(survts, cpu(Yt_test))
    push!(surves, cpu(Ye_test))
end



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

C_IND = concordance_index(concat_outs, concat_survts, concat_surves)

end_of_study = Int(max(concat_survts...))
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
ylow = 0
lrt_pval = round(log_rank_test(concat_survts, concat_surves, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)

caption = "High risk vs Low risk based on median of risk scores by CPHDNN \n on Leucegene data using LSC17 Gene Expression (N=$nsamples)\n concordance index = $(round(C_IND,digits =3)) Log-rank-test pval = $lrt_pval"
            
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
CairoMakie.save("$outpath/aggregated_scores_high_vs_low_CPHDNN_LGN.pdf",fig)


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