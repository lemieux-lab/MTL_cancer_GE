# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")

outpath, session_id = set_dirs() 
#brca_prediction = GDC_data("Data/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);

##### DATA loading
#brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);

nfolds = 5
##### MTAE for survival prediction
brca_mtcphae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "mtl_cph_ae", "session_id" => session_id, "nsamples_train" => length(brca_prediction.rows) - Int(round(length(brca_prediction.rows) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.rows) / nfolds)),
"nsamples" => length(brca_prediction.rows) , "insize" => length(brca_prediction.cols), "ngenes" => length(brca_prediction.cols), "nclasses"=> length(unique(brca_prediction.subgroups)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-1, "dim_redux" => 30, "enc_nb_hl" => 2, 
"enc_hl_size" => 50, "dec_nb_hl" => 2, "dec_hl_size" => 50, "clf_nb_hl" => 2, "clf_hl_size"=> 50,
"lr_cph" => 1e-4, "cph_nb_hl" => 2, "cph_hl_size" => 50)

dump_cb_brca = dump_model_cb(Int(floor(brca_mtcphae_params["nsamples"] / brca_mtcphae_params["mb_size"])), labs_appdf(brca_prediction.subgroups), export_type = "pdf")
device!()

validate!(brca_mtcphae_params, brca_prediction, dump_cb_brca)
function validate!(brca_mtcphae_params, brca_prediction, dump_cb_brca)
    
    folds = split_train_test(Matrix(brca_prediction.data), brca_prediction.survt, brca_prediction.surve, brca_prediction.rows;nfolds =5)
    mkdir("RES/$(brca_mtcphae_params["session_id"])/$(brca_mtcphae_params["modelid"])")
    # init results lists 
    # init results lists 
    scores_by_fold, yt_by_fold, ye_by_fold = [],[], []
    
    # create fold directories
    [mkdir("RES/$(brca_mtcphae_params["session_id"])/$(brca_mtcphae_params["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:brca_mtcphae_params["nfolds"]]
    # splitting, dumped 
    #folds = split_train_test(Data.data, label_binarizer(Data.targets), nfolds = params["nfolds"])
    # dump_folds(folds, brca_mtcphae_params, brca_prediction.rows)
    # dump params
    bson("RES/$(brca_mtcphae_params["session_id"])/$(brca_mtcphae_params["modelid"])/params.bson",brca_mtcphae_params)
    #function train!(model::mtl_cph_AE, fold, dump_cb, params)
        ## mtliative Auto-Encoder + Classifier NN model training function 
        ## Vanilla Auto-Encoder training function 

    
    for fold in folds
        device()
        #device!()
        model = build(brca_mtcphae_params)
        ## STATIC VARS    
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
            
        learning_curve = []
        for iter in 1:nepochs#ProgressBar(1:nepochs)
            cursor = (iter -1)  % nminibatches + 1
            mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            X_ = gpu(train_x[:,mb_ids])
            ## gradient Auto-Encoder 
            ps = Flux.params(model.ae.net)
            gs = gradient(ps) do
                model.ae.lossf(model.ae, X_, X_, weight_decay = brca_mtcphae_params["wd"])
            end
            Flux.update!(model.ae.opt, ps, gs)
            ## gradient CPH
            
            ps = Flux.params(model.cph.model)
            gs = gradient(ps) do
                
                model.cph.lossf(model.cph.model,gpu(train_x),gpu(train_y_e), NE_frac_tr, brca_mtcphae_params["wd"])
            end
            Flux.update!(model.cph.opt, ps, gs)
            ae_loss = model.ae.lossf(model.ae, X_, X_, weight_decay = wd)
            #ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
            ae_cor =  round(my_cor(vec(X_), vec(model.ae.net(gpu(X_)))),digits = 3)
            OUTS = model.cph.model(gpu(train_x))
            cph_loss = model.cph.lossf(model.cph.model,gpu(train_x),gpu(train_y_e), NE_frac_tr, brca_mtcphae_params["wd"])
            cind_tr, cdnt_tr, ddnt_tr  = concordance_index(gpu(train_y_t), gpu(train_y_e), OUTS)
            brca_mtcphae_params["tr_acc"] = cind_tr
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            ae_loss_test = round(model.ae.lossf(model.ae, gpu(test_x), gpu(test_x), weight_decay = wd), digits = 3)
            ae_cor_test = round(my_cor(vec(gpu(test_x)), vec(model.ae.net(gpu(test_x)))), digits= 3)
            cph_loss_test = round(model.cph.lossf(model.cph.model,gpu(test_x),gpu(test_y_e), NE_frac_tst, brca_mtcphae_params["wd"]), digits= 3)
            cind_test = round(concordance_index(gpu(test_y_t), gpu(test_y_e), model.cph.model(gpu(test_x)))[1], digits =3)
            push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr, ae_loss_test, ae_cor_test, cph_loss_test, cind_test))
            println("$iter\t TRAIN AE-loss $(round(ae_loss,digits =3)) \t AE-cor: $(round(ae_cor, digits = 3))\t cph-loss: $(round(cph_loss,digits =3)) \t cph-cind: $(round(cind_tr,digits =3))\t TEST AE-loss $(round(ae_loss_test,digits =3)) \t AE-cor: $(round(ae_cor_test, digits = 3))\t cph-loss: $(round(cph_loss_test,digits =3)) \t cph-cind: $(round(cind_test,digits =3))")
            dump_cb_brca(model, learning_curve, brca_mtcphae_params, iter, fold)

        end
        OUTS = vec(cpu(model.cph.model(gpu(test_x))))
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
    f = draw(p_high + x_high + p_low + x_low, axis = (;title = "CPHDNN Auto-Encoder in Multi-Task - Predicted Low (blue) vs High (red) risk\nc-index: $(round(median(Cis), digits = 3))\nlog-rank-test pval: $lrt_pval"))
    CairoMakie.save("$outpath/$(brca_mtcphae_params["modelid"])/low_vs_high_surv_curves.pdf",f)
end 
#### TESTS
OUTS = vec(cpu(model.cph.model(gpu(test_x))))
groups = ["low_risk" for i in 1:length(OUTS)]    
high_risk = OUTS .> median(OUTS)
low_risk = OUTS .< median(OUTS)
median(OUTS)
end_of_study = 365 * 10
groups[high_risk] .= "high_risk"
p_high, x_high, sc1_high, sc2_high = surv_curve(vec(train_y_t)[high_risk], vec(train_y_e)[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(train_y_t[low_risk], train_y_e[low_risk]; color = "blue")

p_high, x_high, sc1_high, sc2_high = surv_curve(test_y_t[high_risk], test_y_e[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(test_y_t[low_risk], test_y_e[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low)
lrt_pval = round(log_rank_test(vec(test_y_t), vec(test_y_e), groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)

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

