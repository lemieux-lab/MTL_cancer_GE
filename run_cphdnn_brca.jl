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
brca_prediction, infile = assemble_BRCA_data(CLIN_FULL, brca_fpkm)
infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
brca_prediction = BRCA_data(infile)

#brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
#brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);

brca_prediction.data
brca_prediction.survt
brca_prediction.surve
brca_prediction.stage
brca_prediction.samples
clinf = assemble_clinf(brca_prediction)
clinf[:,"stage"] = brca_prediction.stage
clinf[:,"survt"] = brca_prediction.survt
clinf[:,"surve"] = Array{Int}(brca_prediction.surve)
clinf[:,"age_years"] = Array{Int}(brca_prediction.age)

clinf[clinf[:,"stage"] .== "'--",:]
counter("stage", clinf)
#clin_data = CSV.read("Data/GDC_processed/TCGA_BRCA_clinicial_raw.csv", DataFrame, header = 2)
#CLIN_FULL = CSV.read("Data/GDC_processed/GDC_clinical_raw.tsv", DataFrame)

CSV.write("RES/SURV/TCGA_BRCA_clinical_survival.csv", clinf)

nfolds = 5
##### MTAE for survival prediction
brca_cphdnn_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "cphdnn", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50,"wd" => 1e-1,
"nb_clinf" => 5,"cph_lr" => 1e-5, "cph_nb_hl" => 2, "cph_hl_size" => 32)
model = build(brca_cphdnn_params)

dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
clinf = assemble_clinf(brca_prediction)
#clinf = DataFrame([:case_ids=>clinf[:,1], :dummy1=>zeros(size(clinf)[1])])
validate_4!(brca_cphdnn_params, brca_prediction, dump_cb_brca,clinf)
folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:end]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
fold = folds[2]            
#function train!(model::mtl_cph_AE, fold, dump_cb, params)
    ## mtliative Auto-Encoder + Classifier NN model training function 
    ## Vanilla Auto-Encoder training function 

## STATIC VARS    
batchsize = brca_cphdnn_params["mb_size"]
nepochs= brca_cphdnn_params["nepochs"]
wd = brca_cphdnn_params["wd"]
ordering = sortperm(fold["Y_t_train"])
train_x = gpu(Matrix(fold["train_x"][ordering,:]'));
train_x_c = gpu(Matrix(fold["train_x_c"][ordering,:]'));
train_x_ = vcat(train_x, train_x_c)
    
train_y_t = gpu(Matrix(fold["Y_t_train"][ordering,:]'));
train_y_e = gpu(Matrix(fold["Y_e_train"][ordering,:]'));
NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

ordering = sortperm(fold["Y_t_test"])
test_x = gpu(Matrix(fold["test_x"][ordering,:]'));
test_x_c = gpu(Matrix(fold["test_x_c"][ordering,:]'));

test_y_t = gpu(Matrix(fold["Y_t_test"][ordering,:]'));
test_y_e = gpu(Matrix(fold["Y_e_test"][ordering,:]'));
test_x_ = vcat(test_x, test_x_c)
    
NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

#train_x_gpu = gpu(train_x)
#train_y_gpu = gpu(train_y)
nsamples = size(train_y_t)[2]
nminibatches = Int(floor(nsamples/ batchsize))
    
learning_curve = []

for iter in 1:1000#ProgressBar(1:nepochs)
    cursor = (iter -1)  % nminibatches + 1
    mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
    #X_ = gpu(train_x[:,mb_ids])
    #X_c_ = gpu(train_x_c[:,mb_ids])
    ## gradient CPH
    ps = Flux.params(model.model)
    gs = gradient(ps) do
        
        model.lossf(model,train_x_, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
    end
    Flux.update!(model.opt, ps, gs)
    
    OUTS = vec(model.model(train_x_))
    cph_loss =model.lossf(model,train_x_, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
    cind_tr, cdnt_tr, ddnt_tr  = concordance_index(train_y_t, train_y_e, OUTS)
    brca_cphdnn_params["tr_acc"] = cind_tr
    #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
    # save model (bson) every epoch if specified 
    cph_loss_test = round(model.lossf(model,test_x_, test_y_e, NE_frac_tst, brca_cphdnn_params["wd"]), digits= 3)
    cind_test = round(concordance_index(test_y_t, test_y_e,  vec(model.model(test_x_)))[1], digits =3)
    push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
    println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss: $(round(cph_loss,digits =3)) \t cph-cind: $(round(cind_tr,digits =3))\t TEST cph-loss: $(round(cph_loss_test,digits =3)) \t cph-cind: $(round(cind_test,digits =3))")
    #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), brca_cphdnn_params, iter, fold)

end
fig = Figure();
ax = Axis(fig[1,1])
OUTS = cpu(vec(model.model(train_x_)))
hist!(fig[1,1], OUTS )
fig
concordance_index(cpu(train_y_t), cpu(train_y_e), OUTS .* -1)
fig = Figure();
ax = Axis(fig[1,1])
OUTS = cpu(vec(model.model(test_x_)))
hist!(fig[1,1], OUTS )
fig
concordance_index(cpu(test_y_t), cpu(test_y_e), OUTS)


function validate_4!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf;device = gpu)
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:end]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
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
        ordering = sortperm(fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));
        train_x_ = vcat(train_x, train_x_c)
            
        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
        test_x_ = vcat(test_x, test_x_c)
         
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        #train_x_gpu = gpu(train_x)
        #train_y_gpu = gpu(train_y)
        nsamples = size(train_y_t)[2]
        nminibatches = Int(floor(nsamples/ batchsize))
            
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            cursor = (iter -1)  % nminibatches + 1
            mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            #X_ = gpu(train_x[:,mb_ids])
            #X_c_ = gpu(train_x_c[:,mb_ids])
            ## gradient CPH
            ps = Flux.params(model.model)
            gs = gradient(ps) do
                
                model.lossf(model,train_x_, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            
            OUTS = vec(model.model(train_x_))
            cph_loss =model.lossf(model,train_x_, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
            cind_tr, cdnt_tr, ddnt_tr  = concordance_index(train_y_t, train_y_e, OUTS)
            brca_cphdnn_params["tr_acc"] = cind_tr
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model,test_x_, test_y_e, NE_frac_tst, brca_cphdnn_params["wd"]), digits= 3)
            cind_test = round(concordance_index(test_y_t, test_y_e,  vec(model.model(test_x_)))[1], digits =3)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss: $(round(cph_loss,digits =3)) \t cph-cind: $(round(cind_tr,digits =3))\t TEST cph-loss: $(round(cph_loss_test,digits =3)) \t cph-cind: $(round(cind_test,digits =3))")
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), brca_cphdnn_params, iter, fold)

        end
        OUTS = cpu(vec(model.model(test_x_)))
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




validate_3!(brca_cphdnn_params, brca_prediction, dump_cb_brca,clinf)
function validate_3!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf;device = gpu)
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:end]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
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
        train_x = device(Matrix(fold["train_x"][order,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][order,:]'));

        train_y_t = device(Matrix(fold["Y_t_train"][order,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][order,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        order = sortperm(fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][order,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][order,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][order,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][order,:]'));
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        #train_x_gpu = gpu(train_x)
        #train_y_gpu = gpu(train_y)
        nsamples = size(train_y_t)[2]
        nminibatches = Int(floor(nsamples/ batchsize))
            
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            cursor = (iter -1)  % nminibatches + 1
            mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            #X_ = gpu(train_x[:,mb_ids])
            #X_c_ = gpu(train_x_c[:,mb_ids])
            ## gradient CPH
            
            ps = Flux.params(model.encoder, model.cphdnn)
            gs = gradient(ps) do
                
                model.lossf(model,train_x, train_x_c, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            
            OUTS = vec(model.cphdnn(vcat(model.encoder(train_x),train_x_c)))
            cph_loss = model.lossf(model,train_x, train_x_c, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
            cind_tr, cdnt_tr, ddnt_tr  = concordance_index(train_y_t, train_y_e, OUTS)
            brca_cphdnn_params["tr_acc"] = cind_tr
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model,test_x,test_x_c,test_y_e, NE_frac_tst, brca_cphdnn_params["wd"]), digits= 3)
            cind_test = round(concordance_index(test_y_t, test_y_e,  vec(model.cphdnn(vcat(model.encoder(test_x),test_x_c))) )[1], digits =3)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss: $(round(cph_loss,digits =3)) \t cph-cind: $(round(cind_tr,digits =3))\t TEST cph-loss: $(round(cph_loss_test,digits =3)) \t cph-cind: $(round(cind_test,digits =3))")
            dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), brca_cphdnn_params, iter, fold)

        end
        OUTS = cpu(vec(model.cphdnn(vcat(model.encoder(test_x),test_x_c))))
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