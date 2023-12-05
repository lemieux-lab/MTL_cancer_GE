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
vars = var(lgn_prediction[:tpm_data], dims = 1)
keep = vec(var(lgn_prediction[:tpm_data], dims = 1) .>= median(vars))
lgn_prediction[:tpm_data] = lgn_prediction[:tpm_data][:,keep]
lgn_prediction[:genes] = lgn_prediction[:genes][keep] 
names(CF)
y_data = Matrix(CF[:,["adverse","intermediate","favorable"]])
y_data = label_binarizer(lgn_prediction[:labels])
nepochs = 2000
nfolds, ae_nb_hls, dim_redux = 5, 1, 30
lgn_aecphclf_params = Dict("model_title"=>"CPH_AE_CLF_LGN", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
"model_type" => "aecphclf", "commit_head"=>read(`git log -1 --format="%H"1`, String)[1:12], "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(lgn_prediction[:samples]) - Int(round(length(lgn_prediction[:samples]) / nfolds)), "nsamples_test" => Int(round(length(lgn_prediction[:samples]) / nfolds)),
"nfolds" => 5,  "nepochs" => nepochs, "cph_wd" => 1e-3, "ae_wd" => 1e-3,"clfdnn_wd" => 1e-3, "nsamples" => length(lgn_prediction[:samples]) , "insize" => size(lgn_prediction[:genes])[1], "ngenes" => length(lgn_prediction[:genes]),  
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"dim_redux" => dim_redux, "n.-lin" => leakyrelu,  "ae_lr" => 1e-6, "cph_lr" => 1e-6, "cph_nb_hl" => 2, "cph_hl_size" => 64,
"nb_clinf"=>8,"clfdnn_lr" => 1e-6, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2],
"model_cv_complete" => false)
lgn_cph_cb = dump_aecphclf_model_cb(500, lgn_prediction[:labels], export_type = "pdf")
Y_data = hcat(Matrix(CF), y_data)
validate_aecphdnn_dev!(lgn_aecphclf_params, lgn_prediction, lgn_cph_cb ,Y_data)

