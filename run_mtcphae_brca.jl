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
    
#brca_prediction = GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve) 
#brca_prediction = GDC_data_surv("Data/GDC_processed/TCGA_BRCA_TPM_lab_surv.h5";log_transf = true);

function minmaxnorm(data, genes)
    # remove unexpressed
    genes = genes[vec(sum(data, dims = 1) .!= 0)]
    data = data[:, vec(sum(data, dims = 1) .!= 0)]
    # normalize
    vmax = maximum(data, dims = 1)
    vmin = minimum(data, dims = 1)
    newdata = (data .- vmin) ./ (vmax .- vmin)
    genes = genes[vec(var(newdata, dims = 1) .> 0.02)]
    newdata = newdata[:, vec(var(newdata, dims = 1) .> 0.02)]
    return newdata, genes 
end 

nfolds = 5
##### MTAE for survival prediction
brca_mtcphae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "mtl_cph_ae", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 200, "ae_lr" => 1e-4, "wd" => 1e-1, "dim_redux" => 16, "enc_nb_hl" => 2, 
"enc_hl1_size" => 128, "enc_hl2_size" => 128, "dec_nb_hl" => 2, "dec_hl1_size" => 128, "dec_hl2_size" => 128,
"nb_clinf"=>5, "cph_lr" => 1e-4, "cph_nb_hl" => 1, "cph_hl_size" => 64)
clinf = assemble_clinf(brca_prediction)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
validate_mtcphae!(brca_mtcphae_params, brca_prediction, dump_cb_brca)

model = build(brca_mtcphae_params)
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

train_y_t = gpu(Matrix(fold["Y_t_train"][order,:]'));
train_y_e = gpu(Matrix(fold["Y_e_train"][order,:]'));
NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0
train_x
train_x_c
folds[1]["train_x_c"]
model.cph.encoder(train_x)
model.cph.lossf(model.cph, train_x, train_x_c, train_y_e, NE_frac_tr, brca_mtcphae_params["wd"])

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

