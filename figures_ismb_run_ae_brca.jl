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
nfolds, ae_nb_hls, nepochs = 5, 1, 30_000
dim_redux_sizes = [2000]
insize = size(x_data)[2]
#dim_redux = 2000
for (i,dim_redux) in enumerate(dim_redux_sizes)
    brca_ae_params = Dict("model_title"=>"AE_BRCA_DIM_REDUX_$dim_redux", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
        "model_type" => "auto_encoder", "session_id" => session_id,  "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
        "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
        "nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
        "nfolds" => 5,  "nepochs" => nepochs, "mb_size" => size(x_data)[1], "ae_lr" => 1e-3, "wd" => 0, "dim_redux" => dim_redux, 
        "ae_hl_size"=> Int(floor(dim_redux * sqrt(insize/dim_redux))), "enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, 
        "nb_clinf"=>5, "model_cv_complete" => false)
    brca_ae_cb = dump_ae_model_cb(1000, export_type = "pdf") # dummy

    bmodel, bmfold, outs_test, x_test, outs_train, x_train = validate_auto_encoder!(brca_ae_params, brca_prediction, brca_ae_cb, clinf;build_adaptative=false);
end