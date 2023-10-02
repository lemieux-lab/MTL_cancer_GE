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
#clinf = assemble_clinf(brca_prediction)
#CSV.write("Data/GDC_processed/TCGA_clinical_features_survival_pam50.csv", PAM50_data )
clinf = CSV.read("Data/GDC_processed/TCGA_clinical_features_survival_pam50.csv", DataFrame)
keep = clinf[:, "clinical_data_PAM50MRNA"] .!= "NA"
y_lbls = clinf[keep, "clinical_data_PAM50MRNA"]
y_data = label_binarizer(y_lbls)
x_data = brca_prediction.data[keep,:]
nfolds, ae_nb_hls, dim_redux, nepochs = 5, 1, 2, 200
brca_aeclfdnn_params = Dict("model_title"=>"AE_CLF_BRCA_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "aeclfdnn", "session_id" => session_id, "machine_id"=>strip(read(`hostname`, String)), "device" => device(), 
"nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => nepochs, "ae_lr" => 1e-5, "wd" => 1e-3, "dim_redux" => dim_redux, 
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"clfdnn_lr" => 1e-4, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2], "model_cv_complete" => false)
brca_clf_cb = dump_aeclfdnn_model_cb(200, y_lbls, export_type = "pdf")

######### TCGA breast cancer
###### Proof of concept with Auto-Encoder classifier DNN. Provides directed dimensionality reductions
## 1 CLFDNN-AE 2D vs random x10 replicat accuracy BOXPLOT, train test samples by class bn layer 2D SCATTER. 
bmodel, bmfold, outs_test, y_test, outs_train, y_train = validate_aeclfdnn!(brca_aeclfdnn_params, x_data, y_data, brca_prediction.samples[keep], brca_clf_cb)

X_tr = bmodel.ae.encoder(gpu(bmfold["train_x"]'))
X_tst = bmodel.ae.encoder(gpu(bmfold["test_x"]'))
tr_labels = y_lbls[bmfold["train_ids"]]
tst_labels = y_lbls[bmfold["test_ids"]]

plot_embed_train_test_2d(X_tr, X_tst, tr_labels, tst_labels, brca_aeclfdnn_params,  )
## hexbin
## 2d scatter 


## 2 CLFDNN-AE 3D vs random x10 replicat accuracy BOXPLOT, train test samples by class bn layer 3D SCATTER.
 ## 3 CLFDNN-AE in f. of bottleneck size x10 replicate BOXPLOT, train test accuracy. Random benchmark. 
###### Bottleneck dimensionality in Auto-Encoders 
## 4 Auto-Encoder train, test recontstruction correlation by bottleneck size. x10 replicate BOXPLOT.
###### CPHDNN noisy input features leads to overfitting problem
## 5 CPHDNN c-index
###### Alternate learning and learning rate tuning in AECPHDNN
## 6 AECPHDNN C-index AE-learning rate 2D SCATTER
###### AECPHDNN performance and overfitting by bottleneck size. 
## 7 AECPHDNN C-index vs random BOXPLOT by bn size, vs CF benchmark. 2D SCATTER
###### AECPHDNN performs as well as regular CPHDNN and CPH on CF 
## 8 TCGA-BRCA benchmark BOXPLOT. CPHCF, CPHDNNCF, AECPHDNNCF, random.  
## 9 LGN benchmark
## 10 OV benchmark