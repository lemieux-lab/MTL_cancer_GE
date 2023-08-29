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
brca_fpkm = CSV.read("Data/GDC_processed/TCGA-BRCA.htseq_fpkm.tsv", DataFrame)
CLIN_FULL = CSV.read("Data/GDC_processed/GDC_clinical_raw.tsv", DataFrame)
# brca_prediction, infile = assemble_BRCA_data(CLIN_FULL, brca_fpkm_df)
infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
brca_prediction = BRCA_data(infile)

#brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
#brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);


nfolds = 5
##### MTAE for survival prediction
brca_mtcphae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "mtl_cph_ae", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "ae_lr" => 1e-4, "wd" => 1e-1, "dim_redux" => 32, "enc_nb_hl" => 2, 
"enc_hl1_size" => 128, "enc_hl2_size" => 128, "dec_nb_hl" => 2, "dec_hl1_size" => 128, "dec_hl2_size" => 128,"clf_nb_hl" => 2, "clf_hl_size"=> 128,
"nb_clinf"=>5, "cph_lr" => 1e-3, "cph_nb_hl" => 1, "cph_hl_size" => 64)
clinf = assemble_clinf(brca_prediction)
validate_mtcphae!(brca_mtcphae_params, brca_prediction, dump_cb_brca)

model = build(brca_mtcphae_params)
to_cpu(model)
model.cph.cphdnn
cpu(model.encoder)
cpu(model.cph)
cpu(model.ae)
folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:end]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
fold = folds[1]
dump_cb_brca = dump_model_cb(100, labs_appdf(brca_prediction.stage), export_type = "pdf")
device!()
wd = brca_mtcphae_params["wd"]
ordering = sortperm(-fold["Y_t_train"])
train_x = gpu(Matrix(fold["train_x"][order,:]'));
train_x_c = gpu(Matrix(fold["train_x_c"][ordering,:]'));

train_y_t = gpu(Matrix(fold["Y_t_train"][order,:]'));
train_y_e = gpu(Matrix(fold["Y_e_train"][order,:]'));
NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0
train_x
train_x_c
folds[1]["train_x_c"]
model.cph.encoder(train_x)
model.cph.lossf(model.cph, train_x, train_x_c, train_y_e, NE_frac_tr, brca_mtcphae_params["wd"])

function validate_mtcphae!(brca_mtcphae_params, brca_prediction, dump_cb_brca;device=gpu)
    
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:end]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
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
        #device!()
        model = build(brca_mtcphae_params)
        ## STATIC VARS    
        batchsize = brca_mtcphae_params["mb_size"]
        nepochs= brca_mtcphae_params["nepochs"]
        wd = brca_mtcphae_params["wd"]
        ordering = sortperm(-fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));

        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        #train_x_gpu = gpu(train_x)
        #train_y_gpu = gpu(train_y)
        nsamples = size(train_y_t)[2]
        nminibatches = Int(floor(nsamples/ batchsize))
            
        learning_curve = []
        for iter in 1:nepochs#ProgressBar(1:nepochs)
            cursor = (iter -1)  % nminibatches + 1
            mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            X_ = train_x[:,mb_ids]
            ## gradient Auto-Encoder 
            ps = Flux.params(model.ae.net)
            gs = gradient(ps) do
                model.ae.lossf(model.ae, X_, X_, weight_decay = brca_mtcphae_params["wd"])
            end
            Flux.update!(model.ae.opt, ps, gs)
            ## gradient CPH
            
            ps = Flux.params(model.cph.encoder, model.cph.cphdnn)
            gs = gradient(ps) do
                model.cph.lossf(model.cph, train_x, train_x_c, train_y_e, NE_frac_tr, brca_mtcphae_params["wd"])
            end
            Flux.update!(model.cph.opt, ps, gs)
            OUTS_tr = vec(model.cph.cphdnn(vcat(model.cph.encoder(train_x), train_x_c)))
            ae_loss = model.ae.lossf(model.ae, train_x, train_x, weight_decay = wd)
            #ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
            ae_cor =  round(my_cor(vec(train_x), vec(model.ae.net(train_x))),digits = 3)
            cph_loss = model.cph.lossf(model.cph,train_x, train_x_c, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            brca_mtcphae_params["tr_acc"] = cind_tr
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            ae_loss_test = round(model.ae.lossf(model.ae, test_x, test_x, weight_decay = wd), digits = 3)
            ae_cor_test = round(my_cor(vec(test_x), vec(model.ae.net(test_x))), digits= 3)
            cph_loss_test = round(model.cph.lossf(model.cph,test_x, test_x_c, test_y_e, NE_frac_tst, brca_mtcphae_params["wd"]), digits= 3)
            OUTS_tst =  vec(model.cph.cphdnn(vcat(model.encoder(test_x), test_x_c)))
            cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e,OUTS_tst)
            push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr, ae_loss_test, ae_cor_test, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN AE-loss $(round(ae_loss,digits =3)) \t AE-cor: $(round(ae_cor, digits = 3))\t cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\t TEST AE-loss $(round(ae_loss_test,digits =3)) \t AE-cor: $(round(ae_cor_test, digits = 3))\t cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
            dump_cb_brca(model, learning_curve, brca_mtcphae_params, iter, fold)

        end
        OUTS = cpu(vec(model.cph.cphdnn(vcat(model.encoder(test_x), test_x_c))))
        push!(scores_by_fold, OUTS)
        push!(yt_by_fold, cpu(vec(test_y_t)))
        push!(ye_by_fold, cpu(vec(test_y_e)))
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

