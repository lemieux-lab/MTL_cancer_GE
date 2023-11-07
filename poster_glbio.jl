include("init.jl") # first time connect is slow
Pkg.instantiate() # should be quick! 
include("data_processing.jl")
include("mtl_engines.jl")
include("utils.jl")
#device!()
outpath, session_id = set_dirs()

### TCGA (all 33 cancer types)
tcga_prediction = GDC_data("Data/GDC_processed/TCGA_TPM_hv_subset.h5", log_transform = true, shuffled =true);
tmp1 = GDC_data("Data/GDC_processed/TCGA_TPM_hv_subset.h5")
tmp2 = GDC_data("Data/GDC_processed/TCGA_TPM_lab.h5")
tmp1.targets .== tmp2.targets

tcga_prediction.cols
#tcga_prediction = GDC_data("Data/GDC_processed/TCGA_TPM_lab.h5", log_transform = true, shuffled =true);
abbrv = tcga_abbrv()
##### BRCA (5 subtypes)
brca_prediction = GDC_data("Data/GDC_processed/TCGA_BRCA_TPM_lab.h5", log_transform = true, shuffled = true);
highv = reverse(sortperm([var(x) for x in 1:size(brca_prediction.data)[2]]))
highv_25 = highv[1:Int(floor(length(highv)*0.25))]
brca_pred_subset = GDC_data(brca_prediction.data[:,highv_25], brca_prediction.rows, brca_prediction.cols[highv_25], brca_prediction.targets)


#### TSNE 
using TSne

function run_TSNE_dump_h5(tpm_data; ndim = 3, red_dim = 50, max_iter = 1000, perplexity = 30.0, prefix = "TCGA")
    tsne_data = tsne(tpm_data.data, ndim, red_dim, max_iter, perplexity;verbose=true,progress=true)
    fname = "RES/TSNE/$(prefix)_tsne_$(ndim)d"
    params = "tsne($(size(tpm_data.data)), ndims=$ndim, reduce_dims (PCA init) =$red_dim,
    max_iter=$max_iter, perplexity=$perplexity)"
    f = h5open("$fname.h5", "w")
    f["tsne"] = tsne_data
    f["rows"] = tpm_data.rows
    f["cols"] = collect(1:ndim)
    f["targets"] = tpm_data.targets
    f["params"] = params
    close(f)
    df = DataFrame(:tsne1=>tsne_data[:,1], :tsne2=>tsne_data[:,2], :subtype=>tpm_data.targets)
    p= AlgebraOfGraphics.data(df) * mapping(:tsne1, :tsne2, color = :subtype, marker = :subtype) 
    fig = draw(p, axis = (;width = 1024, height = 1024, title = params, aspect = AxisAspect(1), autolimitaspect = 1))
    CairoMakie.save("$fname.pdf", fig)
    return fname
end
    

outfname = run_TSNE_dump_h5(brca_pred_subset, ndim = 2, red_dim = 50, max_iter = 1000, perplexity = 25, prefix = "BRCA_0.25_most_var")

using MultivariateStats
pca = fit(PCA, brca_pred_subset.data', maxoutdim=25, method = :cov);
brca_prediction_pca = predict(pca, brca_pred_subset.data')';
size(brca_prediction_pca)
brca_prediction_pca = GDC_data(brca_prediction_pca, brca_prediction.rows, collect(1:25),brca_prediction.targets)
#CSV.write("Data/GDC_processed/TCGA_BRCA_ids_pam50_subtypes.csv",DataFrame(:sample_id => brca_prediction.rows, :subtype => brca_prediction.targets))

mtl_ae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "tcga_prediction", 
"model_type" => "mtl_ae", "session_id" => session_id, "nsamples" => length(tcga_prediction.rows),
"insize" => length(tcga_prediction.cols), "ngenes" => length(tcga_prediction.cols), "nclasses"=> length(unique(tcga_prediction.targets)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 1000, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-7, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)


dump_cb_dev = dump_model_cb(50*Int(floor(mtl_ae_params["nsamples"] / mtl_ae_params["mb_size"])), labs_appdf(tcga_abbrv(tcga_prediction.targets)), export_type = "pdf")


brca_mtae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "mtl_ae", "session_id" => session_id, "nsamples" => length(brca_prediction.rows),
"insize" => length(brca_prediction.cols), "ngenes" => length(brca_prediction.cols), "nclasses"=> length(unique(brca_prediction.targets)), 
"nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-3, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)


dump_cb_brca = dump_model_cb(50*Int(floor(brca_mtae_params["nsamples"] / brca_mtae_params["mb_size"])), labs_appdf(brca_prediction.targets), export_type = "pdf")

brca_pca_mtae_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "mtl_ae", "session_id" => session_id, "nsamples" => length(brca_prediction.rows),
"insize" => size(brca_prediction_pca.data)[2], "ngenes" => length(brca_prediction_pca.cols), "nclasses"=> length(unique(brca_prediction_pca.targets)), 
"nfolds" => 5,  "nepochs" => 10, "mb_size" => 50, "lr_ae" => 1e-5, "lr_clf" => 1e-4,  "wd" => 1e-3, "dim_redux" => 2, "enc_nb_hl" => 2, 
"enc_hl_size" => 25, "dec_nb_hl" => 2, "dec_hl_size" => 25, "clf_nb_hl" => 2, "clf_hl_size"=> 25)


validate!(brca_mtae_params, brca_prediction, dump_cb_brca)
validate!(brca_pca_mtae_params, brca_prediction_pca, dump_cb_brca)
validate!(mtl_ae_params, tcga_prediction, dump_cb_dev)

model = BSON.load("RES/$session_id/FOLD001/model_000020000.bson")["model"]

using TSne
@time TCGA_tsne = tsne(tcga_prediction.data, 2, 50, 1000, 30.0;verbose=true,progress=true)
TCGA_tsne
labs_appd
tsne2d_df = DataFrame(Dict("tsne_1"=>TCGA_tsne[:,1], "tsne_2"=>TCGA_tsne[:,2], "labels"=>labs_appdf(tcga_abbrv(tcga_prediction.targets)))) 
p= AlgebraOfGraphics.data(tsne2d_df) * mapping(:tsne_1, :tsne_2, color=:labels, marker=:labels) 
fig = draw(p, axis = (;aspect=DataAspect(), title="2D T-SNE (p=30.0) from TCGA gene expression data $(size(tcga_prediction.data)) \nwith PCA init. Cancer types groups.", width = 800, height = 1000));
fig
CairoMakie.save("figures/tsne_tcga.pdf", fig)
### dimensionality reductions 
model = build(mtl_ae_params)
params = mtl_ae_params
folds = split_train_test(tcga_prediction.data, label_binarizer(tcga_prediction.targets), nfolds=5)
mkdir("RES/$(params["session_id"])/$(params["modelid"])")
[mkdir("RES/$(params["session_id"])/$(params["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params["nfolds"]]    
learning_curves = train!(model, folds[1], dump_cb_dev, mtl_ae_params)
### plot correlations X -> X_hat
X_hat = cpu(model.ae.decoder(model.ae.encoder(gpu(folds[1]["test_x"]')))')
X = folds[1]["test_x"]
corr = round(my_cor(vec(X_hat), vec(X)), digits = 3)
df = DataFrame(:inputs=> vec(X),:outputs=>vec(X_hat))
pvst = AlgebraOfGraphics.data(df) * mapping(:inputs, :outputs) * visual(Hexbin;bins=(100,80), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
fig_pvst = draw(pvst; axis=(;aspect = 1, width = 1024, height = 1024,
    title="Pred vs True gene expression in TCGA test set $(size(X)), Pearson corr: $corr",
    xlabel = "true gene expr. log(TPM + 1)",
    ylabel = "predicted gene expr. log(TPM + 1)")
    )
fig_pvst
CairoMakie.save("figures/hexbins_pvst_test_set_tcga.pdf", fig_pvst)
X_hat = cpu(model.ae.encoder(gpu(folds[1]["train_x"]')))
train_df = DataFrame(Dict("EMB1"=>X_hat[1,:], "EMB2"=>X_hat[2,:], "labels"=>labs_appdf(tcga_abbrv(tcga_prediction.targets[folds[1]["train_ids"]])))) 
p= AlgebraOfGraphics.data(train_df) * mapping(:EMB1, :EMB2, color=:labels, marker=:labels) 
fig = draw(p, axis = (;aspect=1, title="2D MT-AE from TCGA gene expression data $(size(folds[1]["train_x"])) \n Cancer types groups.", width = 1000, height = 1000));
CairoMakie.save("figures/MT_AE_tcga_train.pdf", fig)

### dimensionality reductions 
model = build(brca_mtae_params)
params = brca_mtae_params
dump_cb_dev = dump_model_cb(50*Int(floor(params["nsamples"] / params["mb_size"])), brca_prediction.targets, export_type = "pdf")

folds = split_train_test(brca_prediction.data, label_binarizer(brca_prediction.targets), nfolds=5)
mkdir("RES/$(params["session_id"])/$(params["modelid"])")
[mkdir("RES/$(params["session_id"])/$(params["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params["nfolds"]]    
folds[1]["train_y"]
learning_curves = train!(model, folds[1], dump_cb_dev, brca_mtae_params)
### plot correlations X -> X_hat
X_hat = cpu(model.ae.decoder(model.ae.encoder(gpu(folds[1]["test_x"]')))')
X = folds[1]["test_x"]
corr = round(my_cor(vec(X_hat), vec(X)), digits = 3)
df = DataFrame(:inputs=> vec(X),:outputs=>vec(X_hat))
pvst = AlgebraOfGraphics.data(df) * mapping(:inputs, :outputs) * visual(Hexbin;bins=(100,80), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
fig_pvst = draw(pvst; axis=(;aspect = 1, width = 1024, height = 1024,
    title="Pred vs True gene expression in TCGA (BRCA) test set $(size(X)), Pearson corr: $corr",
    xlabel = "true gene expr. log(TPM + 1)",
    ylabel = "predicted gene expr. log(TPM + 1)")
    )
fig_pvst
CairoMakie.save("figures/hexbins_BRCA_pvst_test_set_tcga.pdf", fig_pvst)

X_hat = cpu(model.ae.encoder(gpu(folds[1]["train_x"]')))
train_df = DataFrame(Dict("EMB1"=>X_hat[1,:], "EMB2"=>X_hat[2,:], "labels"=>labs_appdf(brca_prediction.targets[folds[1]["train_ids"]]))) 
p= AlgebraOfGraphics.data(train_df) * mapping(:EMB1, :EMB2, color=:labels, marker=:labels) 
fig = draw(p, axis = (;aspect=1, title="2D MT-AE from TCGA (BRCA) gene expression data $(size(folds[1]["train_x"])) \n Cancer types groups.", width = 1000, height = 1000));
CairoMakie.save("figures/MT_AE_brca_train.pdf", fig)


X_hat = cpu(model.ae.encoder(gpu(folds[1]["test_x"]')))
accuracy(model.clf.model, gpu(folds[1]["test_x"]'), gpu(folds[1]["test_y"]'))
test_df = DataFrame(Dict("EMB1"=>X_hat[1,:], "EMB2"=>X_hat[2,:], "labels"=>brca_prediction.targets[folds[1]["test_ids"]])) 
p= AlgebraOfGraphics.data(test_df) * mapping(:EMB1, :EMB2, color=:labels, marker=:labels) * visual(markersize=20)
fig = draw(p, axis = (;aspect=1, title="2D MT-AE from TCGA (BRCA) gene expression data $(size(folds[1]["test_x"])) \n Cancer types groups.", width = 1000, height = 1000));
CairoMakie.save("figures/MT_AE_brca_test.pdf", fig)

#tcga_ae_red, tr_metrics = fit_transform!(model, folds["train_x"], mtl_ae_params);
mtl_ae_params["tr_acc"] = accuracy(gpu(label_binarizer(tcga_prediction.targets)'), model.clf.model(gpu(tcga_prediction.data')))

lr_fig_outpath = "RES/$(params["session_id"])/$(params["model_type"])/FOLD($(zpad(foldn))_lr_curve.svg"
plot_learning_curves(tr_metrics, mtl_ae_params, lr_fig_outpath)
 
tcga_prediction_mtl_ae_res = validate(mtl_ae_params, tcga_prediction;nfolds = mtl_ae_params["nfolds"])
bson("$outpath/tcga_prediction_mtl_ae_params.bson", mtl_ae_params)



