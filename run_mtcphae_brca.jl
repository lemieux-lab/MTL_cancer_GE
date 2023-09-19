using Pkg
Pkg.activate(".")
# survival data 
include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("cross_validation.jl")
outpath, session_id = set_dirs() 
#brca_prediction = GDC_data("Data/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);

##### DATA loading
brca_fpkm = CSV.read("Data/GDC_processed/TCGA-BRCA.htseq_fpkm.tsv", DataFrame)
CLIN_FULL = CSV.read("Data/GDC_processed/GDC_clinical_raw.tsv", DataFrame)
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

##### ST-CPH-ridge on clinical (BENCHMARK)
brca_cphclinf_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "cphclinf", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes), 
 "nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50,"wd" => 1e-3, "enc_nb_hl" =>ae_nb_hls, "enc_hl_size" => 128, "dim_redux"=> 1, 
"nb_clinf" => 5,"cph_lr" => 1e-3, "cph_nb_hl" => 2, "cph_hl_size" => 128)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
#validate_cphdnn_clinf!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)
validate_cphdnn_clinf!(brca_cphclinf_params, brca_prediction, dummy_dump_cb, clinf)



##### ST-CPHDNN on clinical (BENCHMARK)
noise_sizes = [1,2,3,4,5,10,15,20,30,40,50,75,100,200]
noise_sizes = [300,400,500,1000,2000]
c_inds = []
nepochs = 5_000
for (i, noise_size) in enumerate(noise_sizes)
    brca_prediction_NOISE = BRCA_data(reshape(rand(1050 *noise_size), (1050,noise_size)),brca_prediction.samples, brca_prediction.genes, brca_prediction.survt,brca_prediction.surve,brca_prediction.age, brca_prediction.stage, brca_prediction.ethnicity)
    brca_cphdnnclinf_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
    "model_type" => "cphdnnclinf", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
    "nsamples" => length(brca_prediction.samples) , "insize" => noise_size, 
    "nfolds" => 5,  "nepochs" =>nepochs, "wd" =>  2e-2,  
    "nb_clinf" => 5,"cph_lr" => 1e-4, "cph_nb_hl" => 1, "cph_hl_size" => 16)
    dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
    #validate_cphdnn_clinf!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)
    c_ind = validate_cphdnn_clinf!(brca_cphdnnclinf_params, brca_prediction_NOISE, dummy_dump_cb, clinf)
    push!(c_inds, c_ind)
    ### Update benchmark figure
    if i > 1 
        test_points = zeros(length(c_inds))
        test_points[1:i] .= c_inds
        fig = Figure(resolution = (1024, 512));
        ax = Axis(fig[1,1];xticks=(log10.(noise_sizes[1:i]), ["$x" for x in noise_sizes[1:i]]), xlabel = "Nb of added noisy features (log10 scale)",ylabel = "C-index (test)", title = "Performance of CPHDNN on BRCA clinical features by number of extra noisy input features")
        scatter!(fig[1,1], log10.(noise_sizes[1:i]), test_points, color = "blue", label = "test")
        lines!(fig[1,1], log10.(noise_sizes[1:i]), test_points, color = "blue", linestyle = "--")
        Label(fig[2,1], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(brca_cphdnnclinf_params))")
        CairoMakie.save("$outpath/CPHDNNCLINF_BRCA_BY_NB_DIM.pdf", fig)
    end 
end 

noise_size = 2
brca_prediction_NOISE = BRCA_data(reshape(rand(1050 *noise_size), (1050,noise_size)),brca_prediction.samples, brca_prediction.genes, brca_prediction.survt,brca_prediction.surve,brca_prediction.age, brca_prediction.stage, brca_prediction.ethnicity)
brca_cphdnnclinf_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "cphdnnclinf", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => noise_size, 
"nfolds" => 5,  "nepochs" =>nepochs, "mb_size" => 50,"wd" =>  2e-2, "enc_nb_hl" =>ae_nb_hls, "enc_hl_size" => 128, "dim_redux"=> 1, 
"nb_clinf" => 5,"cph_lr" => 1e-4, "cph_nb_hl" => 2, "cph_hl_size" => 32)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
#validate_cphdnn_clinf!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)
c_ind = validate_cphdnn_clinf!(brca_cphdnnclinf_params, brca_prediction_NOISE, dummy_dump_cb, clinf)
push!(c_inds, c_ind)
println(c_inds)
### Update benchmark figure
i = 2
fig = Figure(resolution = (1024, 512));
ax = Axis(fig[1,1];xticks=(log10.(noise_sizes[1:i]), ["$x" for x in noise_sizes[1:i]]), xlabel = "Nb of added noisy features (log10 scale)",ylabel = "C-index (test)", title = "Performance of CPHDNN on BRCA clinical features by number of extra noisy dimension")
log10.(noise_sizes[1:i])
c_inds
test_points = zeros(2)
test_points[1:i] .= c_inds 
scatter!(fig[1,1], log10.(noise_sizes[1:i]), test_points, color = "blue", label = "test")
lines!(fig[1,1], log10.(noise_sizes[1:i]), test_points, color = "blue", linestyle = "--")
Label(fig[2,1], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(brca_cphdnnclinf_params))")
axislegend(ax, position = :rb)
CairoMakie.save("$outpath/CPHDNNCLINF_BRCA_BY_NB_DIM.png", fig)

##### ST-CPHDNN on clinical + gene expressions
brca_cphdnn_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "enccphdnn", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes), 
 "nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50,"wd" => 1e-3, "enc_nb_hl" =>ae_nb_hls, "enc_hl_size" => 128, "dim_redux"=> 1, 
"nb_clinf" => 5,"cph_lr" => 1e-3, "cph_nb_hl" => 2, "cph_hl_size" => 128)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
#validate_cphdnn_clinf!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)
validate_enccphdnn!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)

model = build(brca_cphdnn_params)

to_cpu(model)
model.cph.cphdnn
cpu(model.encoder)
cpu(model.cph)
cpu(model.ae)
folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:end]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
fold = folds[1]
device!()
wd = brca_mtcphae_params["wd"]
ordering = sortperm(-fold["Y_t_train"])
train_x = gpu(Matrix(fold["train_x"][ordering,:]'));
train_x_c = gpu(Matrix(fold["train_x_c"][ordering,:]'));

train_y_t = gpu(Matrix(fold["Y_t_train"][ordering,:]'));
train_y_e = gpu(Matrix(fold["Y_e_train"][ordering,:]'));
NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0
train_x
train_x_c
folds[1]["train_x_c"]
model.cph.encoder(train_x)
model.cph.lossf(model.cph, train_x, train_x_c, train_y_e, NE_frac_tr, brca_mtcphae_params["wd"])
model.encoder(train_x)
model.lossf(model, train_x, train_x_c, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
#### TESTS
OUTS = vec(cpu(model.cph.model(gpu(test_x))))
groups = ["low_risk" for i in 1:length(OUTS)]    
high_risk = OUTS .> median(OUTS)
low_risk = OUTS .< median(OUTS)
median(OUTS)
end_of_study = 365 * 10
groups[high_risk] .= "high_risk"
p_high, x_high, sc1_high, sc2_high = surv_curve(vec(train_y_t)[high_risk], vec(train_y_e)[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(train_y_t[low_risk], train_y_e[low_risk]; color = "blue")

p_high, x_high, sc1_high, sc2_high = surv_curve(test_y_t[high_risk], test_y_e[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(test_y_t[low_risk], test_y_e[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low)
lrt_pval = round(log_rank_test(vec(test_y_t), vec(test_y_e), groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)

ae_loss_test = round(model.ae.lossf(model.ae, gpu(test_x), gpu(test_x), weight_decay = wd), digits = 3)
ae_cor_test = round(my_cor(vec(gpu(test_x)), vec(model.ae.net(gpu(test_x)))), digits= 3)
cph_loss_test = round(cox_nll_vec(model.cph.model,gpu(test_x),gpu(test_y_e), NE_frac_tst), digits= 3)
cind_test = round(c_index_dev(test_y_t, test_y_e, model.cph.model(gpu(test_x)))[1], digits =3)
my_cor(gpu(vec(test_x)), vec(model.ae.net(gpu(test_x))))

ae_loss_test = cox_nll_vec(model.cph.model,gpu(test_x),gpu(test_y_e), NE_frac_tst)
ps = Flux.params(model.cph.model)
gs = gradient(ps) do
    cox_nll_vec(model.cph.model,gpu(train_x),gpu(train_y_e), NE_frac_tr)
end
Flux.update!(model.cph.opt, ps, gs)

model.cph.lossf(model.cph.model(gpu(train_x)), gpu(train_y_e))

