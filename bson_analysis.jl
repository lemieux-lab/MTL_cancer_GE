include("init.jl")
include("data_processing.jl")
include("cross_validation.jl")
include("mtl_engines.jl")
using BSON
using DataFrames
#device!()


function gather_params(basedir=".")
    df = DataFrame()
    for (root, dirs, files) in walkdir(basedir)
        for file in files
            if file == "params.bson"
                # println("Loading $root/$file")
                d = BSON.load("$root/$file")
                push!(df, d, cols=:union)
            end
        end
    end
    return df
end

df = gather_params("RES/")

#df[:,"model_cv_complete"]
df = df[(df[:,"nepochs"] .>= 30000),:] # .& (df[:,"model_cv_complete"] ),:] # cleanup

unique(df[:, "model_type"])
## train_x
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
## load model BSON
df = df[(df[:, "model_type"] .== "aeaeclfdnn") .& (df[:, "ae_lr"] .== 1e-6),:]
inpath = "./RES/$(bmodel[1])/$(bmodel[2])_$(bmodel[3])/FOLD001/model_000030000.bson"
## AE
model = BSON.load(inpath)["model"]
## AE2D

od = df[1:2,findall(Matrix(df)[1,:] .!== missing)]
CSV.write("tmp.csv", od)
df[1,findall(vec(df[1,:] .!== missing))]
### AE by BN size
ae = df[df[:,"model_type"] .== "auto_encoder",:]
ae = ae[ae[:,"ae_tst_corr"] .!== missing,:]
ae = ae[ae[:,"dim_redux"] .!== missing,:]
fig = Figure(resolution = (1024,512));
ax = Axis(fig[1,1],
    ylabel = "Pearson correlation reconstruction ", 
    xlabel = "Nb of nodes in hidden layer",
    xticks = (log2.(unique(ae[:, "dim_redux"])), ["$x" for x in unique(ae[:, "dim_redux"])] ));
boxplot!(ax, log2.(ae[:, "dim_redux"]), ae[:,"ae_tr_corr"], label = "train")
boxplot!(ax, log2.(ae[:, "dim_redux"]), ae[:,"ae_tst_corr"], label = "test")
fig
axislegend(ax,position =:rb)
CairoMakie.save("figures/AE_by_bn_size_corr.pdf",fig)

#### AECLF 
ae_clf = df[df[:, "model_type"] .== "aeclfdnn",:]
ae_clf = ae_clf[ae_clf[:,"clf_tst_acc"] .!== missing,:]
ae_clf = ae_clf[ae_clf[:,"dim_redux"] .!== missing,:]
fig = Figure(resolution = (1024,512));
ax = Axis(fig[1,1],
    ylabel = "Accuracy of predictions", 
    xlabel = "Nb of nodes in hidden layer",
    xticks = (log2.(unique(ae_clf[:, "dim_redux"])), ["$x" for x in unique(ae_clf[:, "dim_redux"])] ));
boxplot!(ax, log2.(ae_clf[:, "dim_redux"]), ae_clf[:,"clf_tr_acc"], label = "train")
boxplot!(ax, log2.(ae_clf[:, "dim_redux"]), ae_clf[:,"clf_tst_acc"], label = "test")

axislegend(ax,position =:rb)
fig
CairoMakie.save("figures/AE_CLF_by_bn_size_corr.pdf",fig)

df = df[(df[:,"nepochs"] .== 100),:] # .& (df[:,"model_cv_complete"] ),:] # cleanup

#### AE+AE+CLF 
aeae_clf = df[df[:, "model_type"] .== "aeaeclfdnn",:]
aeae_clf = aeae_clf[aeae_clf[:, "ae_lr"] .== 1e-6,:]
aeae_clf = aeae_clf[aeae_clf[:,"clf_tst_acc"] .!== missing,:]
aeae_clf = aeae_clf[aeae_clf[:,"dim_redux"] .!== missing,:]
fig = Figure(resolution = (1024,812));
aeae_clf[:,"dim_redux"]
xticks = sort(sortperm(unique(aeae_clf[:,"dim_redux"])))
ax = Axis(fig[1,1],
    ylabel = "Accuracy of predictions", 
    xlabel = "Auto-Encoder bottleneck width",
    xticks = (xticks, ["$x" for x in sort(unique(aeae_clf[:, "dim_redux"]))] ));
boxplot!(ax, xticks, aeae_clf[:,"clf_tr_acc"][sortperm(aeae_clf[:,"dim_redux"])], label = "train")
boxplot!(ax, xticks, aeae_clf[:,"clf_tst_acc"][sortperm(aeae_clf[:,"dim_redux"])], label = "test")
ax = Axis(fig[2,1],
    ylabel = "Auto-Encoder \n Pearson Correlation", 
    xlabel = "Auto-Encoder bottleneck width",
    xticks = (xticks, ["$x" for x in sort(unique(aeae_clf[:, "dim_redux"]))] ));
boxplot!(ax, xticks, aeae_clf[:,"ae_tr_cor"][sortperm(aeae_clf[:,"dim_redux"])], label = "train")
boxplot!(ax, xticks, aeae_clf[:,"ae_tst_cor"][sortperm(aeae_clf[:,"dim_redux"])], label = "test")

axislegend(ax,position =:rb)
fig
CairoMakie.save("figures/AE_AE_CLF_by_bn_size_acc.pdf",fig)


##### CPHDNN clinf noise
cph = df[(df[:,"nepochs"] .>= 2000) .& (df[:,"model_cv_complete"] ) .& (df[:,"model_type"] .== "cphdnnclinf"),:]
cph = cph[cph[:,"cphdnn_train_c_ind"] .!== missing,:]
cph = cph[cph[:,"insize"] .!== missing,:]
fig = Figure(resolution = (1024,512));
cph[:, "insize"]
ax = Axis(fig[1,1],
    ylabel = "Concordance index ", 
    xlabel = "Nb of added random features to clinical features",
    xticks = (log.(unique(cph[:, "insize"]) .+ 1), ["$x" for x in unique(cph[:, "insize"])] ));

boxplot!(ax, log.(cph[:, "insize"] .+ 1) , Array{Float64}(cph[:,"cphdnn_train_c_ind"]), width = 0.3, label = "train")
boxplot!(ax, log.(cph[:, "insize"] .+ 1), Array{Float64}(cph[:,"cphdnn_tst_c_ind"]), width = 0.3,label = "test")
ylims!(ax, (0.5,1))
axislegend(ax,position =:rc)
fig
CairoMakie.save("figures/CPHDNN_clinf_by_nb_extra_noise.pdf",fig)
df = gather_params("RES/")
cphdnn = df[(df[:,"nepochs"] .>= 3000) .& (df[:,"model_cv_complete"] ) .& (df[:,"model_type"] .== "aecphdnn"),:] # cleanup

#df_subset[:,"aecphdnn_tst_c_ind"]
fig = Figure(resolution = (1024,512));
ax = Axis(fig[1,1],
    ylabel = "Concordance index ", 
    xlabel = "Size of bottleneck layer",
    xticks = (log10.(unique(cphdnn[:, "dim_redux"]) .+ 1), ["$x" for x in unique(cphdnn[:, "dim_redux"])] ));

boxplot!(ax, log10.(cphdnn[:, "dim_redux"] .+ 1) , Array{Float64}(cphdnn[:,"aecphdnn_train_c_ind"]), width = 0.2, label = "train")
boxplot!(ax, log10.(cphdnn[:, "dim_redux"] .+ 1), Array{Float64}(cphdnn[:,"aecphdnn_tst_c_ind"]), width = 0.2,label = "test")
#ylims!(ax, (0.5,1))
axislegend(ax,position =:rt)
fig
CairoMakie.save("figures/AECPHDNN_clinf_by_bn_size.pdf",fig)
aeclfdnn2d = df[(df[:,"nepochs"] .>= 3000) .& (df[:,"model_cv_complete"] ) .& (df[:,"model_type"] .== "aeclfdnn") .& (df[:,"dim_redux"] .== 2),:] # cleanup
keep = findall(aeclfdnn2d[:,"clf_tst_acc"] .== maximum(aeclfdnn2d[:,"clf_tst_acc"]))
aeclfdnn2d[keep,["session_id", "modelid"]]