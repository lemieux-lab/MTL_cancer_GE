include("init.jl")
include("data_processing.jl")
include("mtl_engines.jl")
include("cross_validation.jl")
include("SurvivalDev.jl")
include("utils.jl")
filename = "Data/LEUCEGENE/leucegene_GE_CDS_TPM_clinical.h5"
#filename = leucegene_to_h5(filename)
lgn_prediction = read_leucegene_h5(filename)
# export H5
# import brca data 
# AE(Expr.)->Expr. + AE2D(bottleneck)->bottleneck + CPH(Cf+Expr)->Survival + DNN(Expr.)->Subtype
# learning curves 4 models
# by epoch 2D viz train + test  
# vs 
# CPH/CPH-DNN clinf, CPH-DNN clinf + PCA
outpath, session_id = set_dirs() 
# CPH clinf R
nepochs = 10_000
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
lgn_aeclfdnn_params
#### split 80 /20
# bson("test.bson", Dict("model"=> to_cpu(model)))
# model = BSON.load("model_000001000.bson")["model"]
bmodel, bmfold, outs_test, y_test, outs_train, y_train = validate_aeaeclfdnn!(lgn_aeclfdnn_params, lgn_prediction[:tpm_data], y_data, lgn_prediction[:samples], lgn_clf_cb);
   
params(bmodel.ae)