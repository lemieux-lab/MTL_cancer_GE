# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("cross_validation.jl")
outpath, session_id = set_dirs() 

infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
brca_prediction = BRCA_data(infile, minmax_norm = true, remove_unexpressed=true)
clinf = assemble_clinf(brca_prediction)
#brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
#brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);

sum(sum(brca_prediction.data, dims = 1) .== 0)
nfolds, ae_nb_hls = 5, 1
##### MTAE for survival prediction
brca_ae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "auto_encoder", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => 3_000, "mb_size" => 200, "ae_lr" => 1e-3, "wd" => 1e-6, "dim_redux" => 2, 
"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, 
"nb_clinf"=>5, "cph_lr" => 1e-4, "cph_nb_hl" => 1, "cph_hl_size" => 64)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
#model = build(brca_ae_params)
outs, test_xs, model, train_x, test_x = validate_auto_encoder!(brca_ae_params, brca_prediction, dump_cb_brca, clinf) 

df = DataFrame(:outs=>outs, :true_x=>test_xs)
fig = Figure(resolution = (1024,1024));
ax = Axis(fig[1,1];xlabel="Predicted", ylabel = "True Expr.", aspect = DataAspect())
hexbin!(fig[1,1], outs, test_xs, cellsize=(0.02, 0.02), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
#scatter!(fig[1,1], outs, test_xs, markersize = 0.1)
fig
sum(test_xs.==0) / length(outs)

fig = Figure(resolution = (1024,1024));
ax = Axis(fig[1,1];xlabel="Bottleneck Layer 1", ylabel = "Bottleneck Layer 2")#, aspect = DataAspect())
bneck_layer = cpu(model.encoder(train_x))
#hist!(fig[1,1], bneck_layer[1,:])
scatter!(fig[1,1], bneck_layer[1,:], bneck_layer[2,:])
fig