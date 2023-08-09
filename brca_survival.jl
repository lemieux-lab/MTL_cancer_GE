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
brca_MTAE = GDC_data(brca_prediction.data, brca_prediction.rows, brca_prediction.cols, brca_prediction.subgroups)

highv = reverse(sortperm([var(x) for x in 1:size(brca_prediction.data)[2]]))
highv_25 = highv[1:Int(floor(length(highv)*0.25))]
brca_pred_subset = GDC_data_surv(brca_prediction.data[:,highv_25], brca_prediction.rows, brca_prediction.cols[highv_25], brca_prediction.subgroups, brca_prediction.survt, brca_prediction.surve)


##### MTAE for subgroup prediction
# build model
brca_mtae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_MTAE", 
"model_type" => "mtl_ae", "session_id" => session_id, "nsamples" => length(brca_MTAE.rows),
"insize" => length(brca_MTAE.cols), "ngenes" => length(brca_MTAE.cols), "nclasses"=> length(unique(brca_MTAE.targets)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-3, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)

dump_cb_brca = dump_model_cb(50*Int(floor(brca_mtae_params["nsamples"] / brca_mtae_params["mb_size"])), labs_appdf(brca_MTAE.targets), export_type = "pdf")

validate!(brca_mtae_params, brca_MTAE, dump_cb_brca)

##### MTAE for survival prediction
brca_mtcphae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "mtl_cph_ae", "session_id" => session_id, "nsamples" => length(brca_MTAE.rows),
"insize" => length(brca_prediction.cols), "ngenes" => length(brca_prediction.cols), "nclasses"=> length(unique(brca_prediction.subgroups)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-1, "dim_redux" => 17, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25,
"lr_cph" => 1e-5, "cph_nb_hl" => 2, "cph_hl_size" => 25)

dump_cb_brca = dump_model_cb(Int(floor(brca_mtcphae_params["nsamples"] / brca_mtcphae_params["mb_size"])), labs_appdf(brca_prediction.subgroups), export_type = "pdf")

#validate!(brca_mtcphae_params, brca_prediction, dump_cb_brca)
folds = split_train_test(Matrix(brca_prediction.data), brca_prediction.survt, brca_prediction.surve, brca_prediction.rows;nfolds =5)

fold = folds[1];
model = build(brca_mtcphae_params)
mkdir("RES/$(brca_mtcphae_params["session_id"])/$(brca_mtcphae_params["modelid"])")
# init results lists 
true_labs_list, pred_labs_list = [],[]
# create fold directories
[mkdir("RES/$(brca_mtcphae_params["session_id"])/$(brca_mtcphae_params["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:brca_mtcphae_params["nfolds"]]
# splitting, dumped 
#folds = split_train_test(Data.data, label_binarizer(Data.targets), nfolds = params["nfolds"])
dump_folds(folds, brca_mtcphae_params, brca_prediction.rows)
# dump params
bson("RES/$(brca_mtcphae_params["session_id"])/$(brca_mtcphae_params["modelid"])/params.bson",brca_mtcphae_params)
#function train!(model::mtl_cph_AE, fold, dump_cb, params)
    ## mtliative Auto-Encoder + Classifier NN model training function 
    ## Vanilla Auto-Encoder training function 
    batchsize = brca_mtcphae_params["mb_size"]
    nepochs= brca_mtcphae_params["nepochs"]
    wd = brca_mtcphae_params["wd"]
    order = sortperm(fold["Y_t_train"])
    train_x = Matrix(fold["train_x"][order,:]');
    train_y_t = Matrix(fold["Y_t_train"][order,:]');
    train_y_e = Matrix(fold["Y_e_train"][order,:]');
    NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0
    
    order = sortperm(fold["Y_t_test"])
    test_x = Matrix(fold["test_x"][order,:]');
    test_y_t = Matrix(fold["Y_t_test"][order,:]');
    test_y_e = Matrix(fold["Y_e_test"][order,:]');
    NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0
    
    #train_x_gpu = gpu(train_x)
    #train_y_gpu = gpu(train_y)
    nsamples = size(train_y_t)[2]
    nminibatches = Int(floor(nsamples/ batchsize))
    # dump init state
    learning_curve = []
    ae_loss = model.ae.lossf(model.ae, gpu(train_x), gpu(train_x), weight_decay = wd)
    #ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
    ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
    OUTS = model.cph.model(gpu(train_x))
    
    cph_loss = cox_nll_vec(model.cph.model,gpu(train_x),gpu(train_y_e), NE_frac_tr)
    cind_tr,cdnt_tr, ddnt_tr  = c_index_dev(train_y_t, train_y_e, OUTS)
    fig = Figure();
    ax = Axis(fig[1,1]);
    hist!(fig[1,1],cpu(vec(OUTS)));
    fig
    push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
    ae_loss_test = round(model.ae.lossf(model.ae, gpu(test_x), gpu(test_x), weight_decay = wd), digits = 3)
ae_cor_test = round(my_cor(vec(gpu(test_x)), vec(model.ae.net(gpu(test_x)))), digits= 3)
cph_loss_test = round(cox_nll_vec(model.cph.model,gpu(test_x),gpu(test_y_e), NE_frac_tst), digits= 3)
cind_test = round(c_index_dev(test_y_t, test_y_e, model.cph.model(gpu(test_x)))[1], digits =3)

    push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr, ae_loss_test, ae_cor_test, cph_loss_test, cind_test))
        
    brca_mtcphae_params["tr_acc"] =cind_tr
    learning_curves = learning_curve
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), :ae_loss=>[i[1] for i in learning_curves], :ae_cor => [i[2] for i in learning_curves],
    :cph_loss=>[i[3] for i in learning_curves], :cind_tr=> [i[4] for i in learning_curves],
    :ae_loss_test=>[i[5] for i in learning_curves], :ae_cor_test => [i[6] for i in learning_curves],
    :cph_loss_test=>[i[7] for i in learning_curves], :cind_test=> [i[8] for i in learning_curves])
    fig = Figure();
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    ae_loss_tr = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"ae_loss"], color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "CPH Cox-Negative Likelihood")
    cph_loss_tr = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"cph_loss"])
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder Pearson Corr.")
    ae_cort_tr = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"ae_cor"], color = "red")
    fig[2,2] = Axis(fig, xlabel = "steps", ylabel = "CPH Concordance index")
    cind_tr = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"cind_tr"]  )
    #fig
    ae_loss_test = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"ae_loss_test"], color = "red", linestyle = "--")
    ae_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"cph_loss_test"], linestyle = "--")
    ae_loss = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"ae_cor_test"], color = "red", linestyle = "--")
    ae_loss = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"cind_test"] , linestyle = "--")
    dump_cb_brca(model, learning_curve, brca_mtcphae_params, 0, fold)
    params_dict = brca_mtcphae_params
    lr_fig_outpath = "RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))_lr.pdf"
    
    plot_learning_curves(learning_curve, params_dict, lr_fig_outpath)
    X_tr = cpu(model.ae.encoder(gpu(fold["train_x"]')))
    labs_appdf(brca_prediction.subgroups)
    fold["train_ids"]
    "RES/$(brca_mtcphae_params["session_id"])/$(brca_mtcphae_params["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))_lr.pdf"

    learning_curve = []
    for iter in 1:nepochs #ProgressBar(1:nepochs)
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
        X_ = gpu(train_x[:,mb_ids])
        ## gradient Auto-Encoder 
        ps = Flux.params(model.ae.net)
        gs = gradient(ps) do
            model.ae.lossf(model.ae, X_, X_, weight_decay = wd)
        end
        Flux.update!(model.ae.opt, ps, gs)
        ## gradient CPH
        
        ps = Flux.params(model.cph.model)
        gs = gradient(ps) do
            model.cph.lossf(model.cph.model,gpu(train_x),gpu(train_y_e), NE_frac_tr, brca_mtae_params["wd"])
        end
        Flux.update!(model.cph.opt, ps, gs)
        ae_loss = model.ae.lossf(model.ae, X_, X_, weight_decay = wd)
        #ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
        ae_cor =  round(my_cor(vec(X_), vec(model.ae.net(gpu(X_)))),digits = 3)
        OUTS = model.cph.model(gpu(train_x))
        cph_loss = model.cph.lossf(model.cph.model,gpu(train_x),gpu(train_y_e), NE_frac_tr, brca_mtae_params["wd"])
        cind_tr, cdnt_tr, ddnt_tr  = c_index_dev(train_y_t, train_y_e, OUTS)
        brca_mtcphae_params["tr_acc"] = cind_tr
        #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
        # save model (bson) every epoch if specified 
        ae_loss_test = round(model.ae.lossf(model.ae, gpu(test_x), gpu(test_x), weight_decay = wd), digits = 3)
        ae_cor_test = round(my_cor(vec(gpu(test_x)), vec(model.ae.net(gpu(test_x)))), digits= 3)
        cph_loss_test = round(model.cph.lossf(model.cph.model,gpu(test_x),gpu(test_y_e), NE_frac_tst, brca_mtae_params["wd"]), digits= 3)
        cind_test = round(c_index_dev(test_y_t, test_y_e, model.cph.model(gpu(test_x)))[1], digits =3)
        push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr, ae_loss_test, ae_cor_test, cph_loss_test, cind_test))
        println("$iter\t TRAIN AE-loss $(round(ae_loss,digits =3)) \t AE-cor: $(round(ae_cor, digits = 3))\t cph-loss: $(round(cph_loss,digits =3)) \t cph-cind: $(round(cind_tr,digits =3))\t TEST AE-loss $(round(ae_loss_test,digits =3)) \t AE-cor: $(round(ae_cor_test, digits = 3))\t cph-loss: $(round(cph_loss_test,digits =3)) \t cph-cind: $(round(cind_test,digits =3))")
        dump_cb_brca(model, learning_curve, brca_mtcphae_params, iter, fold)

    end
    return params["tr_acc"]
#end 
ae_loss_test = round(model.ae.lossf(model.ae, gpu(test_x), gpu(test_x), weight_decay = wd), digits = 3)
ae_cor_test = round(my_cor(vec(gpu(test_x)), vec(model.ae.net(gpu(test_x)))), digits= 3)
cph_loss_test = round(cox_nll_vec(model.cph.model,gpu(test_x),gpu(test_y_e), NE_frac_tst), digits= 3)
cind_test = round(c_index_dev(test_y_t, test_y_e, model.cph.model(gpu(test_x)))[1], digits =3)
my_cor(gpu(vec(test_x)), vec(model.ae.net(gpu(test_x))))

ae_loss_test = cox_nll_vec(model.cph.model,gpu(test_x),gpu(test_y_e), NE_frac_tst)
ps = Flux.params(model.cph.model)
gs = gradient(ps) do
    cox_nll_vec(model.cph.model,gpu(train_x),gpu(train_y_e), NE_frac_tr)
end
Flux.update!(model.cph.opt, ps, gs)

model.cph.lossf(model.cph.model(gpu(train_x)), gpu(train_y_e))

function cox_nll_vec(mdl::Flux.Chain, X_, Y_e_, NE_frac)
    outs = vec(mdl(gpu(X_)))
    hazard_ratios = exp.(outs)
    log_risk = log.(cumsum(hazard_ratios))
    uncensored_likelihood = outs .- log_risk
    censored_likelihood = uncensored_likelihood .* gpu(Y_e_')
    #neg_likelihood = - sum(censored_likelihood) / sum(e .== 1)
    neg_likelihood = - sum(censored_likelihood) * NE_frac
    return neg_likelihood
end 

# train 
# init 
mkdir("RES/$(params["session_id"])/$(params["modelid"])")
# init results lists 
true_labs_list, pred_labs_list = [],[]
# create fold directories
[mkdir("RES/$(params["session_id"])/$(params["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params["nfolds"]]
# splitting, dumped 
#folds = split_train_test(Data.data, label_binarizer(Data.targets), nfolds = params["nfolds"])
dump_folds(folds, params, Data.rows)
# dump params
bson("RES/$(params["session_id"])/$(params["modelid"])/params.bson", params)

# test
# plot

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
