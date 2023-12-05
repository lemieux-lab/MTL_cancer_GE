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
lgn_cphdnn_params = Dict("model_title"=>"CPHDNN+PCA_LGN_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
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
nfolds, ae_nb_hls, dim_redux = 5, 1, 125
lgn_aeaecphclf_params = Dict("model_title"=>"CPH_AE_AE2D_CLF_LGN", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
"model_type" => "cphc_noexpr", "commit_head"=>read(`git log -1 --format="%H"1`, String)[1:12], "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(lgn_prediction[:samples]) - Int(round(length(lgn_prediction[:samples]) / nfolds)), "nsamples_test" => Int(round(length(lgn_prediction[:samples]) / nfolds)),
"nsamples" => length(lgn_prediction[:samples]) , "insize" => size(new_y_data)[2], "ngenes" => length(lgn_prediction[:genes]),  
"nfolds" => 5,  "nepochs" => nepochs, "wd" => 1e-3, "n.-lin" => leakyrelu, "cph_lr" => 1e-6, "cph_nb_hl" => 2, "cph_hl_size" => 128,
"outsize" => 1, "model_cv_complete" => false, "nb_clinf"=>8,"clfdnn_lr" => 1e-6, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2])
prms = lgn_aeaecphclf_params
enc, cph, dnn, ae = build_ae_cph_dnn(prms)

function build_internal_cph(encoder, model_params)
    gpu(Chain(encoder, Dense(model_params["dim_redux"] + model_params["nb_clinf"] , model_params["cph_hl_size"], leakyrelu),
    #Dense(model_params["cph_hl_size"] , model_params["cph_hl_size"], leakyrelu),
    Dense(model_params["cph_hl_size"] , 1, sigmoid, bias = false)))
    opt = Flux.ADAM(model_params["cph_lr"])
    return dnn(chain, opt, cox_l2)

end 
function build_internal_dnn(encoder, model_params)
    hl_sizes = [Int(floor(model_params["dim_redux"] * c ^ x)) for x in 1:model_params["enc_nb_hl"]]
    hl_sizes = adaptative ?  hl_sizes  : Array{Int}(ones(10) .* model_params["ae_hl_size"])
    for i in 1:model_params["enc_nb_hl"]
        in_size = i == 1 ? model_params["insize"] : reverse(hl_sizes)[i - 1]
        out_size = reverse(hl_sizes)[i]
        push!(enc_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
    end 
    clf_chain = gpu(Chain(encoder..., hls..., Dense(hl_sizes[end], model_params["outsize"], identity)))
        
    clf_opt = Flux.ADAM(model_params["clfdnn_lr"])
    clf_lossf = crossentropy_l2
    return dnn(clf_chain, clf_opt, clf_lossf)
end 
function build_ae_cph_dnn(model_params)
    encoder = build_encoder(model_params)
    decoder = Chain(bnsize, insize)
    output_layer = gpu(Flux.Dense(hl_sizes[end], model_params["insize"], leakyrelu))
    decoder = Flux.Chain(dec_hls..., output_layer)
    dnn = build_internal_dnn(encoder, model_params)
    cph =  build_internal_cph(encoder, model_params)
    ae = Flux.Chain(enc_hls..., redux_layer, dec_hls..., output_layer)
    AE = AE_model(ae, encoder, decoder, output_layer, Flux.ADAM(model_params["ae_lr"]), mse_l2)   
    aecphdnn = Dict(  "enc"=> enc, 
                        "cph"=> cph,
                        "dnn"=> dnn,
                        "ae" => AE)  
    return aecphdnn
end 
function build_encoder(model_params)
    c = compute_c(model_params["insize"], model_params["dim_redux"], model_params["enc_nb_hl"] )
    enc_hls = []
    hl_sizes = [Int(floor(model_params["dim_redux"] * c ^ x)) for x in 1:model_params["enc_nb_hl"]]
    hl_sizes = adaptative ?  hl_sizes  : Array{Int}(ones(10) .* model_params["ae_hl_size"])
    for i in 1:model_params["enc_nb_hl"]
        in_size = i == 1 ? model_params["insize"] : reverse(hl_sizes)[i - 1]
        out_size = reverse(hl_sizes)[i]
        push!(enc_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
    end 
    redux_layer = gpu(Flux.Dense(reverse(hl_sizes)[end], model_params["dim_redux"],identity))
    return Flux.Chain(enc_hls..., redux_layer)
end 
lgn_aeclfdnn_params = Dict("model_title"=>"AE_AE_CLF_LGN_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "lgn_prediction", 
"model_type" => "aeaeclfdnn", "commit_head"=>read(`git log -1 --format="%H"1`, String)[1:12], "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(lgn_prediction[:samples]) - Int(round(length(lgn_prediction[:samples]) / nfolds)), "nsamples_test" => Int(round(length(lgn_prediction[:samples]) / nfolds)),
"nsamples" => length(lgn_prediction[:samples]) , "insize" => length(lgn_prediction[:genes]), "ngenes" => length(lgn_prediction[:genes]),  
"nfolds" => 5,  "nepochs" => nepochs, "ae_lr" => 1e-6, "wd" => 1e-3, "dim_redux" => dim_redux, 
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"clfdnn_lr" => 1e-6, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2], "model_cv_complete" => false)
