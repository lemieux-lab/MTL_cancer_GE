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

clinf = CSV.read("Data/GDC_processed/TCGA_clinical_features_survival_pam50.csv", DataFrame)
keep = clinf[:, "clinical_data_PAM50MRNA"] .!= "NA"
#y_lbls = clinf[keep, "clinical_data_PAM50MRNA"]
y_lbls = clinf[:, "clinical_data_PAM50MRNA"]
y_data = label_binarizer(y_lbls)
x_data = brca_prediction.data[keep,:]
nepochs, ae_nb_hls, nfolds = 3000, 1, 5
dim_redux_sizes = [1,2,3,4,5,10,15,20,30,50,100,200]
for dim_redux in dim_redux_sizes
    ### AECPHDNN by bottleneck size 
    brca_aecphdnn_params = Dict(
        ## run infos 
        "session_id" => session_id, "nfolds" =>5,  "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
        "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", "model_title"=>"AECPHDNN",
        ## data infos 
        "dataset" => "BRCA_data(norm=true)", "nsamples" => size(brca_prediction.samples)[1],
        "nsamples_test" => Int(round(size(brca_prediction.samples)[1] / nfolds)), "ngenes" => size(brca_prediction.genes)[1],
        "nsamples_train" => size(brca_prediction.samples)[1] - Int(round(size(brca_prediction.samples)[1] / nfolds)),
        ## optim infos 
        "nepochs" => nepochs, "ae_lr" =>1e-3, "cph_lr" => 1e-4, "ae_wd" => 1e-6, "cph_wd" => 1e-6,
        ## model infos
        "model_type"=> "aecphdnn", "dim_redux" => dim_redux, "ae_nb_hls" => ae_nb_hls, "ae_hl_size"=> 128,
        "enc_nb_hl" => ae_nb_hls, "enc_hl_size"=> 128,  "dec_nb_hl" => ae_nb_hls, "dec_hl_size"=> 128,
        "nb_clinf" => 5, "cph_nb_hl" => 2, "cph_hl_size" => 64, 
        "insize" => length(brca_prediction.genes),
        ## metrics
        "model_cv_complete" => false
    )
    metrics = validate_aecphdnn!(brca_aecphdnn_params, brca_prediction, dummy_dump_cb, clinf)
end
# brca_aecphdnn_params
# mod = metrics["best_model"]
# X_tr = cpu(mod.encoder(gpu(metrics["best_model_fold"]["train_x"]')))
# X_tst= cpu(mod.encoder(gpu(metrics["best_model_fold"]["test_x"]')))
# Y_t_tr = metrics["best_model_fold"]["Y_t_train"]
# Y_e_tr = metrics["best_model_fold"]["Y_e_train"]

# Y_t_tst = metrics["best_model_fold"]["Y_t_test"]
# Y_e_tst = metrics["best_model_fold"]["Y_e_test"]


# PI_tr = cpu(vec(mod.cph.cphdnn(vcat(mod.encoder(gpu(metrics["best_model_fold"]["train_x"]')), gpu(metrics["best_model_fold"]["train_x_c"]')))))
# PI_tst = cpu(vec(mod.cph.cphdnn(vcat(mod.encoder(gpu(metrics["best_model_fold"]["test_x"]')), gpu(metrics["best_model_fold"]["test_x_c"]')))))

# #labels = metrics["best_model_fold"]["Y_e_train"]
# fig = Figure();
# ax = Axis(fig[1,1])
# metrics["cph_outs_train"]
# y_lbls
# scatter!(ax, X_tr[1,:],X_tr[2,:], color = PI_tr )#,color = y_lbls[metrics["best_model_fold"]["train_ids"]])
# scatter!(ax, X_tst[1,:],X_tst[2,:], color = PI_tst, strokewidth=2)#,color = y_lbls[metrics["best_model_fold"]["train_ids"]])

# #xlims!(ax, (-0.02,0.02))

# fig

# fig = Figure();
# ax = Axis(fig[1,1])
# metrics["cph_outs_train"]
# y_lbls
# scatter!(ax,-log.(sort(PI_tst)),Y_t_tst[sortperm(PI_tst)],color=Y_e_tst[sortperm(PI_tst)])#,color = y_lbls[metrics["best_model_fold"]["train_ids"]])
# #hist!(ax, Y_t_tr[sortperm(PI_tr)],color=Y_e_tr[sortperm(PI_tr)])#,color = y_lbls[metrics["best_model_fold"]["train_ids"]])
# sort(PI_tst)
# #ylims!(ax, (0,10))
# #xlims!(ax, (-0.02,0.02))

# fig


# brca_aecphdnn_params