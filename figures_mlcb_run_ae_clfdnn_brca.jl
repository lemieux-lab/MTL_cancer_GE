include("init.jl")
include("data_processing.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("mtl_engines.jl")
include("cross_validation.jl")
device!()
#### DATA 
outpath, session_id = set_dirs() 
infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
brca_prediction = BRCA_data(infile, minmax_norm = true)
nsamples = size(brca_prediction.samples)[1]
#clinf = assemble_clinf(brca_prediction)
#CSV.write("Data/GDC_processed/TCGA_clinical_features_survival_pam50.csv", PAM50_data )
clinf = CSV.read("Data/GDC_processed/TCGA_clinical_features_survival_pam50.csv", DataFrame)
keep = clinf[:, "clinical_data_PAM50MRNA"] .!= "NA"
y_lbls = clinf[keep, "clinical_data_PAM50MRNA"]
y_data = label_binarizer(y_lbls)
x_data = brca_prediction.data[keep,:]
#### dump processed data
outfile = h5open("Data/GDC_processed/TCGA_BRCA_fpkm_hv_norm_PAM50.h5", "w")
outfile["data"] = x_data
outfile["genes"] = brca_prediction.genes
outfile["samples"] = brca_prediction.samples[keep]
outfile["labels"] = Array{String}(y_lbls)
close(outfile)
######### TCGA breast cancer
###### Proof of concept with Auto-Encoder classifier DNN. Provides directed dimensionality reductions
## 1 CLFDNN-AE 2D vs random x10 replicat accuracy BOXPLOT, train test samples by class bn layer 2D SCATTER. 
nepochs = 50_000
nfolds, ae_nb_hls, dim_redux = 5, 1, 125
brca_aeclfdnn_params = Dict("model_title"=>"AE_AE_CLF_BRCA_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "aeaeclfdnn", "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(brca_prediction.samples[keep]) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples[keep]) / nfolds)),
"nsamples" => length(brca_prediction.samples[keep]) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => nepochs, "ae_lr" => 1e-6, "wd" => 1e-3, "dim_redux" => dim_redux, 
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"clfdnn_lr" => 1e-6, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2], "model_cv_complete" => false)
brca_clf_cb = dump_aeaeclfdnn_model_cb(1000, y_lbls, export_type = "pdf")


##### 2D-AE + AE + DNN (clf) 

#### split 80 /20
folds = split_train_test(x_data, y_data, brca_prediction.samples;nfolds = nfolds)
model = build(brca_aeclfdnn_params; adaptative=true)
model.ae2d
# bson("test.bson", Dict("model"=> to_cpu(model)))
# model = BSON.load("model_000001000.bson")["model"]
bmodel, bmfold, outs_test, y_test, outs_train, y_train = validate_aeaeclfdnn!(brca_aeclfdnn_params, x_data, y_data, brca_prediction.samples[keep], brca_clf_cb)

X_proj = Matrix(cpu(bmodel.ae2d.encoder(bmodel.encoder(gpu(bmfold["train_x"]')))'))
tr_labels = y_lbls[bmfold["train_ids"]]
tr_embed = DataFrame(:emb1=>X_proj[:,1], :emb2=>X_proj[:,2], :cancer_type => tr_labels)
train = AlgebraOfGraphics.data(tr_embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type) * visual(markersize =20)
fig = draw(train, axis = (;aspect = AxisAspect(1), autolimitaspect = 1, width = 1024, height =1024,
title="$(brca_aeclfdnn_params["model_type"]) on $(brca_aeclfdnn_params["dataset"]) data\naccuracy by DNN TRAIN: $(round(brca_aeclfdnn_params["clf_tr_acc"] * 100, digits=2))% TEST: $(round(brca_aeclfdnn_params["clf_tst_acc"]*100, digits=2))%"))
fig    
    
emb_fig_outpath = "RES/$session_id/$(brca_aeclfdnn_params["model_type"])_$(brca_aeclfdnn_params["modelid"])_AE_2D_final_model_30000.pdf"
CairoMakie.save(emb_fig_outpath, fig)
            

bmodel, bmfold, outs_test, y_test, outs_train, y_train = validate_aeclfdnn!(brca_aeclfdnn_params, x_data, y_data, brca_prediction.samples[keep], brca_clf_cb)


## 2 CLFDNN-AE 3D vs random x10 replicat accuracy BOXPLOT, train test samples by class bn layer 3D SCATTER.
nfolds, ae_nb_hls, dim_redux = 5, 1, 3
brca_aeclfdnn_params = Dict("model_title"=>"AE_CLF_BRCA", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "aeclfdnn", "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
"nsamples_train" => length(brca_prediction.samples[keep]) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples[keep]) / nfolds)),
"nsamples" => length(brca_prediction.samples[keep]) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => nepochs, "ae_lr" => 1e-4, "wd" => 1e-3, "dim_redux" => dim_redux, 
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"clfdnn_lr" => 1e-3, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2], "model_cv_complete" => false)
brca_clf_cb = dump_aeclfdnn_model_cb(1000, y_lbls, export_type = "pdf")

bmodel, bmfold, outs_tst, y_test, outs_tr, y_train = validate_aeclfdnn!(brca_aeclfdnn_params, x_data, y_data, brca_prediction.samples[keep], brca_clf_cb);

## 3 CLFDNN-AE in f. of bottleneck size x10 replicate BOXPLOT, train test accuracy. Random benchmark. 
nfolds, ae_nb_hls = 5, 1
dim_redux_sizes = [1, 5,10,15,20,30,50,100,200,300,500,1000,2000]
for (i,dim_redux) in enumerate(dim_redux_sizes)
    brca_aeclfdnn_params = Dict("model_title"=>"AE_CLF_BRCA", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
    "model_type" => "aeclfdnn", "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", 
    "nsamples_train" => length(brca_prediction.samples[keep]) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples[keep]) / nfolds)),
    "nsamples" => length(brca_prediction.samples[keep]) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
    "nfolds" => 5,  "nepochs" => nepochs, "ae_lr" => 1e-4, "wd" => 1e-3, "dim_redux" => dim_redux, 
    "ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
    "clfdnn_lr" => 1e-3, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2], "model_cv_complete" => false)
    brca_clf_cb = dump_aeclfdnn_model_cb(1000, y_lbls, export_type = "pdf")
    bmodel, bmfold, outs_test, y_test, outs_train, y_train = validate_aeclfdnn!(brca_aeclfdnn_params, x_data, y_data, brca_prediction.samples[keep], brca_clf_cb);
end

###### Bottleneck dimensionality in Auto-Encoders 
## 4 Auto-Encoder train, test recontstruction correlation by bottleneck size. x10 replicate BOXPLOT.

nfolds, ae_nb_hls, nepochs = 5, 1, 30_000
dim_redux_sizes = [1,2,3,4,5,10,15,20,30,50,100,200,300,500,1000,2000]
for (i,dim_redux) in enumerate(dim_redux_sizes)
    brca_ae_params = Dict("model_title"=>"AE_BRCA_DIM_REDUX_$dim_redux", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
        "model_type" => "auto_encoder", "session_id" => session_id,  "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
        "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
        "nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
        "nfolds" => 5,  "nepochs" => nepochs, "mb_size" => 200, "ae_lr" => 1e-3, "wd" => 1e-4, "dim_redux" => dim_redux, 
        "ae_hl_size"=> 128, "enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, 
        "nb_clinf"=>5, "model_cv_complete" => false)
    brca_ae_cb = dump_ae_model_cb(1000, export_type = "pdf") # dummy

    bmodel, bmfold, outs_test, x_test, outs_train, x_train = validate_auto_encoder!(brca_ae_params, brca_prediction, brca_ae_cb, clinf;build_adaptative=false);
end

fig = Figure(resolution = (1024,1024));
fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
steps = collect(1:100)
loss_tr = sin.(steps)
loss_tst = sin.(steps .+1)

lines!(fig[1,1], steps, loss_tst, linewidth = 4,  color = (:blue,0.75))
lines!(fig[1,1], steps, loss_tr, linewidth = 4,  color = (:red,0.6))
fig

###### CPHDNN noisy input features leads to overfitting problem
## no noise
for i in 1:10
    brca_cphdnnclinf_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
    "model_type" => "cphdnnclinf", "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
    "nsamples" => length(brca_prediction.samples) , "insize" => 0, "nfolds" => 5,  "nepochs" =>nepochs, "wd" =>  1e-3,  
    "nb_clinf" => 5,"cph_lr" => 1e-4, "cph_nb_hl" => 2, "cph_hl_size" => 32, "model_cv_complete" => false)
    brca_cphdnnclinf_cb = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
    best_model, best_model_fold, outs_test, yt_test, ye_test, outs_train, yt_train, ye_train = validate_cphdnn_clinf!(brca_cphdnnclinf_params, brca_prediction, brca_cphdnnclinf_cb, clinf)
end
## 5 CPHDNN c-index
noise_sizes = [1,2,3,4,5,10,15,20,30,50,100,200,300,500,1000,2000]
for (i, noise_size) in enumerate(noise_sizes)
    brca_prediction_NOISE = BRCA_data(reshape(rand(nsamples *noise_size), (nsamples,noise_size)),brca_prediction.samples, brca_prediction.genes, brca_prediction.survt,brca_prediction.surve,brca_prediction.age, brca_prediction.stage, brca_prediction.ethnicity)
    brca_cphdnnclinf_params_noise = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction_noise", 
    "model_type" => "cphdnnclinf", "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
    "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
    "nsamples" => length(brca_prediction.samples) , "insize" => noise_size, "nfolds" => 5,  "nepochs" =>nepochs, "wd" =>  2e-2,  
    "nb_clinf" => 5,"cph_lr" => 1e-4, "cph_nb_hl" => 1, "cph_hl_size" => 32, "model_cv_complete" => false)
    brca_cphdnnnoise_cb = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
    best_model, best_model_fold, outs_test, yt_test, ye_test, outs_train, yt_train, ye_train = validate_cphdnn_clinf_noise!(brca_cphdnnclinf_params_noise, brca_prediction_NOISE, brca_cphdnnnoise_cb, clinf)
end 


###### AECPHDNN performance and overfitting by bottleneck size. 
## 7 AECPHDNN C-index vs random BOXPLOT by bn size, vs CF benchmark. 2D SCATTER

###### Alternate learning and learning rate tuning in AECPHDNN
## 6 AECPHDNN C-index AE-learning rate 2D SCATTER


###### AECPHDNN performs as well as regular CPHDNN and CPH on CF 
## 8 TCGA-BRCA benchmark BOXPLOT. CPHCF, CPHDNNCF, AECPHDNNCF, random.  
## 9 LGN benchmark
## 10 OV benchmark