# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")

outpath, session_id = set_dirs() 

brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);
brca_MTAE = GDC_data(brca_prediction.data, brca_prediction.rows, brca_prediction.cols, brca_prediction.subgroups)

folds = split_train_test(Matrix(brca_prediction.data), brca_prediction.survt, brca_prediction.surve, brca_prediction.rows;nfolds =5)

fold = folds[1]

### single task Auto-Encoder

#### 2D -> 0.962 Pearson Cor.
brca_ae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "ae", "session_id" => session_id, "nsamples" => length(brca_prediction.rows),
"insize" => length(brca_prediction.cols), "ngenes" => length(brca_prediction.cols), "nclasses"=> length(unique(brca_prediction.subgroups)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-1, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25)
#### 50D -> 0.962 Pearson Cor.
brca_ae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "ae", "session_id" => session_id, "nsamples" => length(brca_prediction.rows),
"insize" => length(brca_prediction.cols), "ngenes" => length(brca_prediction.cols), "nclasses"=> length(unique(brca_prediction.subgroups)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-1, "dim_redux" => 50, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25)


#device!()
AE = AE_model(brca_ae_params);
batchsize = brca_ae_params["mb_size"]
nepochs= brca_ae_params["nepochs"]
wd = brca_ae_params["wd"]
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

nsamples = size(train_y_t)[2]
nminibatches = Int(floor(nsamples/ batchsize))
for iter in 1:brca_ae_params["nepochs"] #ProgressBar(1:nepochs)
    cursor = (iter -1)  % nminibatches + 1
    mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
    X_ = gpu(train_x[:,mb_ids])
    ## gradient Auto-Encoder 
    ps = Flux.params(AE.net)
    gs = gradient(ps) do
        AE.lossf(AE, X_, X_, weight_decay = wd)
    end
    Flux.update!(AE.opt, ps, gs)
    ae_loss = AE.lossf(AE, X_, X_, weight_decay = wd)
    ae_cor =  round(my_cor(vec(X_), vec(AE.net(gpu(X_)))),digits = 3)
    
    ae_loss_test = round(AE.lossf(AE, gpu(test_x), gpu(test_x), weight_decay = wd), digits = 3)
    ae_cor_test = round(my_cor(vec(gpu(test_x)), vec(AE.net(gpu(test_x)))), digits= 3)
        
    println("$iter\t TRAIN AE-loss $(round(ae_loss,digits =3)) \t AE-cor: $(round(ae_cor, digits = 3))\t TEST AE-loss $(round(ae_loss_test,digits =3)) \t AE-cor: $(round(ae_cor_test, digits = 3))")
        

end
# SINGLE TASK CPHDNN


##### DATA loading
#brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);

nfolds = 5
##### CPH DNN model
brca_cphdnn_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "enccphdnn", "session_id" => session_id, "nsamples_train" => length(brca_prediction.rows) - Int(round(length(brca_prediction.rows) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.rows) / nfolds)),
"nsamples" => length(brca_prediction.rows) , "insize" => length(brca_prediction.cols), "ngenes" => length(brca_prediction.cols), "nclasses"=> length(unique(brca_prediction.subgroups)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-1, "dim_redux" => 16, "enc_nb_hl" => 2, 
"enc_hl_size" => 32, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 32,
"cph_lr" => 1e-5, "cph_nb_hl" => 2, "cph_hl_size" => 32)

dump_cb_brca = dump_model_cb(Int(floor(brca_cphdnn_params["nsamples"] / brca_cphdnn_params["mb_size"])), labs_appdf(brca_prediction.subgroups), export_type = "pdf")
validate!(brca_cphdnn_params, brca_prediction, dump_cb_brca)
function validate!(brca_cphdnn_params, brca_prediction, dump_cb_brca)
    folds = split_train_test(Matrix(brca_prediction.data), brca_prediction.survt, brca_prediction.surve, brca_prediction.rows;nfolds =5)
    #device()
    #device!()
    mkdir("RES/$(brca_cphdnn_params["session_id"])/$(brca_cphdnn_params["modelid"])")
    # init results lists 
    scores_by_fold, yt_by_fold, ye_by_fold = [],[], []
    # create fold directories
    [mkdir("RES/$(brca_cphdnn_params["session_id"])/$(brca_cphdnn_params["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:brca_cphdnn_params["nfolds"]]
    # splitting, dumped 
    #folds = split_train_test(Data.data, label_binarizer(Data.targets), nfolds = params["nfolds"])
    # dump_folds(folds, brca_cphdnn_params, brca_prediction.rows)
    # dump params
    bson("RES/$(brca_cphdnn_params["session_id"])/$(brca_cphdnn_params["modelid"])/params.bson",brca_cphdnn_params)

    for fold in folds
        model = build(brca_cphdnn_params)
        #function train!(model::mtl_cph_AE, fold, dump_cb, params)
            ## mtliative Auto-Encoder + Classifier NN model training function 
            ## Vanilla Auto-Encoder training function 

        ## STATIC VARS    
        batchsize = brca_cphdnn_params["mb_size"]
        nepochs= brca_cphdnn_params["nepochs"]
        wd = brca_cphdnn_params["wd"]
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
            
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            cursor = (iter -1)  % nminibatches + 1
            mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            X_ = gpu(train_x[:,mb_ids])

            ## gradient CPH
            
            ps = Flux.params(model.net)
            gs = gradient(ps) do
                
                model.lossf(model.net,gpu(train_x),gpu(train_y_e), NE_frac_tr, brca_cphdnn_params["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            
            OUTS = model.net(gpu(train_x))
            cph_loss = model.lossf(model.net,gpu(train_x),gpu(train_y_e), NE_frac_tr, brca_cphdnn_params["wd"])
            cind_tr, cdnt_tr, ddnt_tr  = concordance_index(gpu(train_y_t), gpu(train_y_e), OUTS)
            brca_cphdnn_params["tr_acc"] = cind_tr
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model.net,gpu(test_x),gpu(test_y_e), NE_frac_tst, brca_cphdnn_params["wd"]), digits= 3)
            cind_test = round(concordance_index(gpu(test_y_t), gpu(test_y_e), model.net(gpu(test_x)))[1], digits =3)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss: $(round(cph_loss,digits =3)) \t cph-cind: $(round(cind_tr,digits =3))\t TEST cph-loss: $(round(cph_loss_test,digits =3)) \t cph-cind: $(round(cind_test,digits =3))")
            dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), brca_cphdnn_params, iter, fold)

        end
        OUTS = vec(cpu(model.net(gpu(test_x))))
        push!(scores_by_fold, OUTS)
        push!(yt_by_fold, vec(test_y_t))
        push!(ye_by_fold, vec(test_y_e))
    end 
    #### TESTS
    concat_OUTS = vcat(scores_by_fold...)
    concat_yt = vcat(yt_by_fold...)
    concat_ye = vcat(ye_by_fold...)
    Cis = []
    for i in 1:1000
        sampling = rand(1:length(concat_OUTS),length(concat_OUTS))
        push!(Cis, concordance_index(concat_yt[sampling], concat_ye[sampling], concat_OUTS[sampling])[1])
    end
    Cis
    median(Cis)
    groups = ["low_risk" for i in 1:length(concat_OUTS)]    
    high_risk = concat_OUTS .> median(concat_OUTS)
    low_risk = concat_OUTS .< median(concat_OUTS)
    end_of_study = 365 * 10
    groups[high_risk] .= "high_risk"
    p_high, x_high, sc1_high, sc2_high = surv_curve(concat_yt[high_risk], concat_ye[high_risk]; color = "red")
    p_low, x_low, sc1_low, sc2_low = surv_curve(concat_yt[low_risk], concat_ye[low_risk]; color = "blue")

    lrt_pval = round(log_rank_test(concat_yt, concat_ye, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)
    f = draw(p_high + x_high + p_low + x_low, axis = (;title = "CPHDNN Single Task - Predicted Low (blue) vs High (red) risk\nc-index: $(round(median(Cis), digits = 3))\nlog-rank-test pval: $lrt_pval"))
    CairoMakie.save("$outpath/$(brca_cphdnn_params["modelid"])/low_vs_high_surv_curves.pdf",f)
end 