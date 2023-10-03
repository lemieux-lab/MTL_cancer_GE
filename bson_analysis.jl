using Pkg
Pkg.activate(".")
using BSON
roots = []
for dir1 in readdir("RES/")
    if isdir("RES/$dir1/")
    for dir2 in readdir("RES/$dir1/")
        if isdir("RES/$dir1/$dir2")
            root = "RES/$dir1/$dir2"
            #println(readdir(root))
            if "params.bson" in readdir(root)
                push!(roots, BSON.load("$root/params.bson"))
            end 
        end 
    end 
end
end
roots


params_dict = Dict()
size = 0
for root in roots
    for key in keys(root)
        if !(key in keys(params_dict))
            newcol = Vector{Union{Missing, Any}}(missing, size + 1)
            newcol[end] = root[key]
            params_dict[key] = newcol
        else 
            push!(params_dict[key],root[key])
        end
    end 
    size += 1
end
[size(params_dict[key])[1] for key in keys(params_dict)]
features = ["clf_tst_acc", "dim_redux"]
params_dict["dim_redux"]
DataFrame(Dict([(k, params_dict[k]) for k in features]))

using BSON
using DataFrames

Params = Dict{String, Any}

BSON.bson("test/1/params.bson", Params("a" => 1.0, "b" => 2))
BSON.bson("test/2/params.bson", Params("b" => 3, "c" => 4f0))
BSON.bson("test/3/params.bson", Params("a" => 5.0, "b" => 6, "c" => 7f0))

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
df[:,"model_cv_complete"]
df = df[(df[:,"nepochs"] .>= 3000) .& (df[:,"model_cv_complete"] ),:] # cleanup

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

##### CPHDNN clinf noise
cph = df[df[:, "model_type"] .== "cphdnnclinf",:]
cph = cph[cph[:,"cphdnn_train_c_ind"] .!== missing,:]
cph = cph[cph[:,"insize"] .!== missing,:]
fig = Figure(resolution = (1024,512));
cph[:, "insize"]
ax = Axis(fig[1,1],
    ylabel = "Concordance index ", 
    xlabel = "Nb of added random features to clinical features",
    xticks = (log2.(unique(cph[:, "insize"])), ["$x" for x in unique(cph[:, "insize"])] ));

boxplot!(ax, log2.(cph[:, "insize"]), Array{Float64}(cph[:,"cphdnn_train_c_ind"]), label = "train")
boxplot!(ax, log2.(cph[:, "insize"]), Array{Float64}(cph[:,"cphdnn_tst_c_ind"]), label = "test")

axislegend(ax,position =:rc)
fig
CairoMakie.save("figures/CPHDNN_clinf_by_nb_extra_noise.pdf",fig)
