include("init.jl")
include("data_processing.jl")
include("mtl_engines.jl")
include("cross_validation.jl")
include("SurvivalDev.jl")
include("utils.jl")
device!()
filename = "Data/LEUCEGENE/leucegene_GE_CDS_TPM_clinical.h5"

lgn_clinical = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)

info = lgn_clinical[:,"Sex"] 
lgn_clinical = lgn_clinical[:,names(lgn_clinical).!= "Sex"]
lgn_clinical[:,"Sex"] = Array{Int}(info .== "M")
info = lgn_clinical[:,"Cytogenetic risk"] 
lgn_clinical = lgn_clinical[:,names(lgn_clinical).!= "Cytogenetic risk"]
lgn_clinical[:,"adverse"] = label_binarizer(info)'[1,:]
lgn_clinical[:,"intermediate"] = label_binarizer(info)'[2,:]
lgn_clinical[:,"favorable"] = label_binarizer(info)'[3,:]
features = ["adverse","favorable", "intermediate", "Sex", "Age_at_diagnosis","FLT3-ITD mutation", "NPM1 mutation","IDH1-R132 mutation"]

CF = lgn_clinical[:,features]


#names(lgn_clinical)
#filename = leucegene_to_h5(filename)
lgn_prediction = read_leucegene_h5(filename)
# export H5
# import brca data 
# AE(Expr.)->Expr. + AE2D(bottleneck)->bottleneck + CPH(Cf+Expr)->Survival + DNN(Expr.)->Subtype
# learning curves 4 models
# by epoch 2D viz train + test  
outpath, session_id = set_dirs() 
# CPH clinf R
nepochs = 40_000
nfolds, ae_nb_hls, dim_redux = 5, 1, 125
lgn_prediction
y_data = label_binarizer(lgn_prediction[:labels])
lgn_aeclfdnn_params = Dict("model_title"=>"AE_AE_CLF_LGN_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
"model_type" => "aeaeclfdnn", "commit_head"=>read(`git log -1 --format="%H"1`, String)[1:12], "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(lgn_prediction[:samples]) - Int(round(length(lgn_prediction[:samples]) / nfolds)), "nsamples_test" => Int(round(length(lgn_prediction[:samples]) / nfolds)),
"nsamples" => length(lgn_prediction[:samples]) , "insize" => length(lgn_prediction[:genes]), "ngenes" => length(lgn_prediction[:genes]),  
"nfolds" => 5,  "nepochs" => nepochs, "ae_lr" => 1e-6, "wd" => 1e-3, "dim_redux" => dim_redux, 
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"clfdnn_lr" => 1e-6, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2], "model_cv_complete" => false)
lgn_clf_cb = dump_aeaeclfdnn_model_cb(1000, lgn_prediction[:labels], export_type = "pdf")
##### 2D-AE + AE + DNN (clf) 
#lgn_aeclfdnn_params
#### split 80 /20
# bson("test.bson", Dict("model"=> to_cpu(model)))
# model = BSON.load("model_000001000.bson")["model"]
#bmodel, bmfold, outs_test, y_test, outs_train, y_train = validate_aeaeclfdnn!(lgn_aeclfdnn_params, lgn_prediction[:tpm_data], y_data, lgn_prediction[:samples], lgn_clf_cb);
# vs 
y_data = Matrix(CF)
# CPH/CPH-DNN clinf, CPH-DNN clinf + PCA
lgn_cphdnn_params = Dict("model_title"=>"CPHDNN_LGN_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
"model_type" => "cphdnnclinf_noexpr", "commit_head"=>read(`git log -1 --format="%H"1`, String)[1:12], "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(lgn_prediction[:samples]) - Int(round(length(lgn_prediction[:samples]) / nfolds)), "nsamples_test" => Int(round(length(lgn_prediction[:samples]) / nfolds)),
"nsamples" => length(lgn_prediction[:samples]) , "insize" => length(lgn_prediction[:genes]), "ngenes" => length(lgn_prediction[:genes]),  
"nfolds" => 5,  "nepochs" => nepochs, "wd" => 1e-3, "n.-lin" => leakyrelu, "cph_lr" => 1e-6, "cph_nb_hl" => 2, "cph_hl_size" => 128,
"outsize" => 1, "model_cv_complete" => false, "nb_clinf"=>8)

lgn_cph_cb = dump_aeaeclfdnn_model_cb(1000, lgn_prediction[:labels], export_type = "pdf")
# best_model, best_model_fold, outs_test, test_yt, test_ye, outs_train, train_yt, train_ye  = validate_cphclinf_noexpr!(lgn_cphdnn_params,
# lgn_prediction, lgn_cph_cb, y_data)


M = fit(PCA, Matrix(lgn_prediction[:tpm_data]'), maxoutdim = 30);
x_data = Matrix(predict(M, Matrix(lgn_prediction[:tpm_data]'))')
new_y_data = hcat(y_data, x_data)
lgn_prediction[:tpm_data] = x_data 
lgn_cphdnnpca_params = Dict("model_title"=>"CPHDNN+PCA_LGN_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
"model_type" => "cphdnnclinf_noexpr", "commit_head"=>read(`git log -1 --format="%H"1`, String)[1:12], "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(lgn_prediction[:samples]) - Int(round(length(lgn_prediction[:samples]) / nfolds)), "nsamples_test" => Int(round(length(lgn_prediction[:samples]) / nfolds)),
"nsamples" => length(lgn_prediction[:samples]) , "insize" => size(new_y_data)[2], "ngenes" => length(lgn_prediction[:genes]),  
"nfolds" => 5,  "nepochs" => nepochs, "wd" => 1e-3, "n.-lin" => leakyrelu, "cph_lr" => 1e-6, "cph_nb_hl" => 2, "cph_hl_size" => 128,
"outsize" => 1, "model_cv_complete" => false, "nb_clinf"=>size(new_y_data)[2])

# best_model, best_model_fold, outs_test, test_yt, test_ye, outs_train, train_yt, train_ye  = validate_cphclinf!(lgn_cphdnn_params,
# lgn_prediction, lgn_cph_cb, new_y_data)
lgn_prediction = read_leucegene_h5(filename)
names(CF)
y_data = Matrix(CF[:,["adverse","intermediate","favorable"]])
nepochs = 5_000
nfolds, ae_nb_hls, dim_redux = 5, 1, 125
lgn_aecphclf_params = Dict("model_title"=>"CPH_AE_CLF_LGN", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
"model_type" => "aecphclf", "commit_head"=>read(`git log -1 --format="%H"1`, String)[1:12], "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(lgn_prediction[:samples]) - Int(round(length(lgn_prediction[:samples]) / nfolds)), "nsamples_test" => Int(round(length(lgn_prediction[:samples]) / nfolds)),
"nfolds" => 5,  "nepochs" => nepochs, "cph_wd" => 1e-3, "ae_wd" => 1e-3,"clfdnn_wd" => 1e-3, "nsamples" => length(lgn_prediction[:samples]) , "insize" => size(lgn_prediction[:genes])[1], "ngenes" => length(lgn_prediction[:genes]),  
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"dim_redux" => dim_redux, "n.-lin" => leakyrelu,  "ae_lr" => 1e-6, "cph_lr" => 1e-4, "cph_nb_hl" => 2, "cph_hl_size" => 128,
"nb_clinf"=>8,"clfdnn_lr" => 1e-6, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2],
"model_cv_complete" => false)
lgn_cph_cb = dump_aeaeclfdnn_model_cb(1000, lgn_prediction[:labels], export_type = "pdf")
model = build_ae_cph_dnn(lgn_aecphclf_params)
dataset = lgn_prediction
clinf = CF
hls, hl_sizes = build_internal_layers(lgn_aecphclf_params)
folds = split_train_test(Matrix(dataset[:tpm_data]), Matrix(clinf), dataset[:survt], dataset[:surve], dataset[:samples];nfolds =5)
params_dict = lgn_aecphclf_params    
fold = folds[1]
nepochs= params_dict["nepochs"]
ordering = sortperm(-fold["Y_t_train"])
train_x = gpu(Matrix(fold["train_x"][ordering,:]'));
train_x_c = gpu(Matrix(fold["train_x_c"][ordering,:]'));
train_y_t = gpu(Matrix(fold["Y_t_train"][ordering,:]'));
train_y_e = gpu(Matrix(fold["Y_e_train"][ordering,:]'));
NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

train_y =gpu(Matrix(fold["train_x_c"][:,1:3]'))
test_y = gpu(Matrix(fold["test_x_c"][:,1:3]'))
model["dnn"].model(train_x)
Matrix{Bool}(train_y)
accuracy(Matrix{Bool}(train_y), cpu(model["dnn"].model(train_x)))
round(accuracy(Matrix{Bool}(train_y), cpu(model["dnn"].model(train_x))),digits = 3)
validate_aecphdnn!(lgn_aecphclf_params, lgn_prediction, lgn_cph_cb ,CF)
function validate_aecphdnn!(params_dict, dataset, dump_cb, clinf;device=gpu)
    folds = split_train_test(Matrix(dataset[:tpm_data]), Matrix(clinf), dataset[:survt], dataset[:surve], dataset[:samples];nfolds =5)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    test_y_t_by_fold, test_y_e_by_fold, train_y_t_by_fold, train_y_e_by_fold = [],[], [], []
    test_scores_by_fold, train_scores_by_fold = [],[]
    train_x_pred_by_fold, test_x_pred_by_fold, test_xs, train_xs = [],[], [], []
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    for fold in folds
        model = build_ae_cph_dnn(params_dict)
        ## STATIC VARS    
        nepochs= params_dict["nepochs"]
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
        
        train_y =gpu(Matrix(fold["train_x_c"][:,1:3]'))
        test_y = gpu(Matrix(fold["test_x_c"][:,1:3]'))

        #nsamples = size(train_y_t)[2]
             
        learning_curve = []
        for iter in 1:nepochs
            ## gradient CPH            
            ps1 = Flux.params(model["cph"].model)
            gs1 = gradient(ps1) do
                model["cph"].lossf(model["cph"],model["enc"](train_x),  train_x_c, train_y_e, NE_frac_tr, params_dict["cph_wd"])
            end

            ## gradient Auto-Encoder 
            ps2 = Flux.params(model["ae"].net)
            gs2 = gradient(ps2) do
                model["ae"].lossf(model["ae"], train_x, train_x, weight_decay = params_dict["ae_wd"])
            end
            
            ## gradient Classfier DNN
            ps3 = Flux.params(model["dnn"].model)
            gs3 = gradient(ps3) do 
                model["dnn"].lossf(model["dnn"], train_x, train_y, weight_decay = params_dict["clfdnn_wd"])
            end

            Flux.update!(model["cph"].opt, ps1, gs1)
            Flux.update!(model["ae"].opt, ps2, gs2)
            Flux.update!(model["dnn"].opt, ps3, gs3)

            OUTS_tr = vec(model["cph"].model(vcat(model["enc"](train_x), train_x_c)))
            ae_loss = model["ae"].lossf(model["ae"], train_x, train_x, weight_decay = params_dict["ae_wd"])
            ae_cor =  round(my_cor(vec(train_x), vec(model["ae"].net(train_x))),digits = 3)
            cph_loss = model["cph"].lossf(model["cph"],model["enc"](train_x),  train_x_c, train_y_e, NE_frac_tr, params_dict["cph_wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            ae_loss_test = round(model["ae"].lossf(model["ae"], test_x, test_x, weight_decay = params_dict["ae_wd"]), digits = 3)
            ae_cor_test = round(my_cor(vec(test_x), vec(model["ae"].net(test_x))), digits= 3)
            cph_loss_test = round(model["cph"].lossf(model["cph"],model["enc"](test_x),  test_x_c, test_y_e, NE_frac_tst, params_dict["cph_wd"]), digits= 3)
            OUTS_tst =  vec(model["cph"].model(vcat(model["enc"](test_x), test_x_c)))
            cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e,OUTS_tst)
            
            train_clf_loss = round(model["dnn"].lossf(model["dnn"], train_x, train_y, weight_decay = params_dict["clfdnn_wd"]),digits=3)
            train_clf_acc =  round(accuracy(Matrix{Bool}(train_y), cpu(model["dnn"].model(train_x))),digits = 3)
            
            test_clf_loss = round(model["dnn"].lossf(model["dnn"], test_x, test_y, weight_decay = params_dict["clfdnn_wd"]),digits=3)
            test_clf_acc =  round(accuracy(Matrix{Bool}(test_y), cpu(model["dnn"].model(test_x))),digits = 3)
            
            push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr, ae_loss_test, ae_cor_test, cph_loss_test, cind_test, train_clf_loss, train_clf_acc, test_clf_loss, test_clf_acc))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN AE-loss $(round(ae_loss,digits =3)) \t AE-cor: $(round(ae_cor, digits = 3))\t cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\t CLF loss $(round(train_clf_loss,digits =3)) \t acc.%: $(round(train_clf_acc, digits = 3))")
            println("\t\tTEST AE-loss $(round(ae_loss_test,digits =3)) \t AE-cor: $(round(ae_cor_test, digits = 3))\t cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]  CLF loss $(round(test_clf_loss,digits =3)) \t acc.%: $(round(test_clf_acc, digits = 3))")
            #dump_cb_brca(model, learning_curve, params_dict, iter, fold)

        end
        if learning_curve[end][8] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][8]
            best_model_fold = fold
            params_dict["bm_tst_c_ind"] = round(best_accuracy, digits =3 )
        end 
        ### AUTO-ENCODER 
        ## test set correlations
        push!(test_x_pred_by_fold, cpu(vec(model.ae.net(test_x))))
        push!(test_xs, cpu(vec(test_x)))
        ## train set correlations
        push!(train_x_pred_by_fold, cpu(vec(model.ae.net(train_x))))
        push!(train_xs, cpu(vec(train_x)))
        #### CPH 
        ### TRAIN 
        push!(train_scores_by_fold, cpu(vec(model.cph.cphdnn(vcat(model.encoder(train_x), train_x_c)))))
        push!(train_y_t_by_fold, cpu(vec(train_y_t)))
        push!(train_y_e_by_fold, cpu(vec(train_y_e)))
        ### TEST 
        push!(test_scores_by_fold, cpu(vec(model.cph.cphdnn(vcat(model.encoder(test_x), test_x_c)))))
        push!(test_y_t_by_fold, cpu(vec(test_y_t)))
        push!(test_y_e_by_fold, cpu(vec(test_y_e)))
        
    end
    #### AUTO-ENCODER
    x_train= vcat(train_xs...)
    x_test = vcat(test_xs...)
    ae_outs_train = vcat(train_x_pred_by_fold...)
    ae_outs_test = vcat(test_x_pred_by_fold...)
    #### CPH 
    ### TESTS
    cph_outs_test = vcat(test_scores_by_fold...)
    yt_test = vcat(test_y_t_by_fold...)
    ye_test = vcat(test_y_e_by_fold...)
    ### TRAIN
    cph_outs_train = vcat(train_scores_by_fold...)
    yt_train = vcat(train_y_t_by_fold...)
    ye_train = vcat(train_y_e_by_fold...)
    
    params_dict["aecphdnn_tst_c_ind"] = concordance_index(yt_test, ye_test, -1 .* cph_outs_test)[1]
    params_dict["aecphdnn_train_c_ind"] = concordance_index(yt_train, ye_train, -1 .* cph_outs_train)[1]
    params_dict["aecphdnn_tst_corr"] = my_cor(ae_outs_test, x_test)
    params_dict["aecphdnn_train_corr"] = my_cor(ae_outs_train, x_train)
    
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    return Dict(
    ## best model for inspection
    "best_model"=>best_model, "best_model_fold" => best_model_fold, 
    ### CPH train set and test sets 
    "cph_outs_test" => cph_outs_test, "yt_test" => yt_test, "ye_test" => ye_test, 
    "cph_outs_train" => cph_outs_train, "yt_train" => yt_train, "ye_train" => ye_train,
    ### AE train set and test sets 
    "ae_outs_test" => ae_outs_test, "x_test" =>x_test, 
    "ae_outs_train" => ae_outs_train, "x_train" => x_train
    )
end 