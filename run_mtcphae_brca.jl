# survival data 
include("init.jl") # first time connect is slow
#Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("cross_validation.jl")
outpath, session_id = set_dirs() 
#brca_prediction = GDC_data("Data/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);
device!()
##### DATA loading
#brca_fpkm = CSV.read("Data/GDC_processed/TCGA-BRCA.htseq_fpkm.tsv", DataFrame)
#CLIN_FULL = CSV.read("Data/GDC_processed/GDC_clinical_raw.tsv", DataFrame)
# brca_prediction, infile = assemble_BRCA_data(CLIN_FULL, brca_fpkm_df)
infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
brca_prediction = BRCA_data(infile, minmax_norm = true)
clinf = assemble_clinf(brca_prediction)
#brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
#brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);


nfolds, ae_nb_hls = 5, 1
##### MTAE for survival prediction
brca_mtcphae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "mtl_cph_ae", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => 5_000, "mb_size" => 200, "ae_lr" => 1e-3, "ae_wd" => 5e-5, "cph_wd" => 1e-3, "dim_redux" => 1, 
"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, 
"nb_clinf"=>5, "cph_lr" => 1e-4, "cph_nb_hl" => 2, "cph_hl_size" => 64)
clinf = assemble_clinf(brca_prediction)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
validate_mtcphae!(brca_mtcphae_params, brca_prediction, dump_cb_brca, clinf)
# implement dropout
# implement hl nb selector DONE 
# implement phase training
