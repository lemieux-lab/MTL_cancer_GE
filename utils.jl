# dump folds 
function dump_folds(folds, params::Dict, case_ids)
    f = h5open("RES/$(params["session_id"])/$(params["modelid"])/fold_ids.bson", "w")
    test_ids = Array{String, 2}(undef, (length(folds), length(folds[1]["test_ids"])))
    [test_ids[i,:] = case_ids[folds[i]["test_ids"]] for i in 1:length(folds)]
    f["test_ids"] =  test_ids
    train_ids = Array{String, 2}(undef, (length(folds), length(folds[1]["train_ids"])))
    [train_ids[i,:] = case_ids[folds[i]["train_ids"]] for i in 1:length(folds)]
    f["train_ids"] = train_ids 
    close(f)
end
function load_folds(params)
    # fold ids loading 
    inf = h5open("RES/$(params["session_id"])/$(params["modelid"])/fold_ids.bson", "r")
    test_ids = inf["test_ids"][:,:]
    train_ids = inf["train_ids"][:,:]
    close(inf)
    return train_ids, test_ids
end 

get_fold_ids(foldn, ids, case_ids) = findall([in(r, ids[foldn,:]) for r in case_ids])
get_fold_data(foldn, ids, cdata) = cdata.data[find_fold_ids(foldn, ids, cdata.rows),:]

############################
###### General utilities ###
############################
counter(feature, clin_data) = Dict([(x, sum(clin_data[:, feature] .== x)) for x in unique(clin_data[:,feature])])
getcurrentcommitnb(;digits = 7) =  read(`git rev-parse HEAD`, String)[1:digits]
zpad(n::Int;pad::Int=9) = lpad(string(n),pad,'0')
function labs_appdf(labs) 
    lbd = Dict([(t,sum(labs .== t )) for t in unique(labs)])
    append(lbl, lbd)= "$lbl ($(lbd[lbl]))" 
    labs_appd = [append(lb,lbd) for lb in labs]
    return labs_appd
end 
function add_n_labels(lbls, counter_dict)
    return ["$x (n=$(counter_dict[x]))" for  x in lbls]
end
function stringify(p::Dict;spacer = 80)  
    s = join(["$key: $val" for (key, val) in p], ", ")
    for i in collect(spacer:spacer:length(s))
        s = "$(s[1:i])\n$(s[i + 1:end])"
    end
    return s 
end 
##########################################
####### Loss Plotting functions / ########
####### Callback functions        ########
##########################################
## train loss data dump!
function dump_ae_model_cb(dump_freq;export_type=".pdf")
    return (model, tr_metrics, params_dict, iter::Int, fold) -> begin
        # check if end of epoch / start / end 
        if iter % dump_freq == 0 || iter == 0 || iter == params_dict["nepochs"]
            model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
            # saves model
            # bson("RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad =3))/model_$(zpad(iter)).bson", Dict("model"=>to_cpu(model)))
            # plot learning curve
            lr_fig_outpath = "RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad=3))_lr.pdf"
            plot_learning_curves_ae(tr_metrics, params_dict, lr_fig_outpath)
        end
    end 
end


##########################################
####### Plotting functions    ############
##########################################
function plot_learning_curves_ae(tr_metrics,params_dict,lr_fig_outpath)
    loss_tst = [x[3] for x in tr_metrics]
    loss_tr = [x[1] for x in tr_metrics]
    corr_tst = [x[4] for x in  tr_metrics]
    corr_tr =[x[2] for x in  tr_metrics]
    CSV.write(lr_fig_outpath, DataFrame(Dict(:loss_tst=>loss_tst, :loss_tr=>loss_tr,:corr_tst=>corr_tst,:corr_tr=>corr_tr)))
    steps = collect(1:size(loss_tst)[1])
    fig = Figure(resolution = (1024,1024))
    ax1 = Axis(fig[1,1], xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    lines!(fig[1,1], steps, loss_tr, linewidth = 5,  color = (:blue, 0.55),label ="train")
    lines!(fig[1,1], steps, loss_tr, linewidth = 2,  color = :black)
    lines!(fig[1,1], steps, loss_tst, linewidth = 5,  color = (:red, 0.55), label = "test")
    lines!(fig[1,1], steps, loss_tst, linewidth = 2,  color = :black)
    
    axislegend(ax1, position = :rc)
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "Pearson correlation coefficient")
    lines!(fig[2,1], steps, corr_tr, linewidth = 5,  color = (:blue, 0.55))
    lines!(fig[2,1], steps, corr_tr, linewidth = 2,  color = :black)
    lines!(fig[2,1], steps, corr_tst, linewidth = 5,  color = (:red, 0.55))
    lines!(fig[2,1], steps, corr_tst, linewidth = 2,  color = :black)
    
    CairoMakie.save(lr_fig_outpath, fig)
end

function plot_embed_train_test_2d(model, fold, params_dict)
    X_tr = cpu(model.ae.encoder(gpu(fold["train_x"]')))
    X_tst = cpu(model.ae.encoder(gpu(fold["test_x"]')))
    tr_labels = cpu(y_lbls[fold["train_ids"]])
    tst_labels = cpu(y_lbls[fold["test_ids"]])

    fig_outpath= "RES/$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])_scatter.pdf"
    # plot final 2d embed from Auto-Encoder
    tr_embed = DataFrame(:emb1=>X_tr[1,:], :emb2=>X_tr[2,:], :cancer_type => tr_labels)
    tst_embed = DataFrame(:emb1=>X_tst[1,:], :emb2=>X_tst[2,:], :cancer_type => tst_labels)
    
    train = AlgebraOfGraphics.data(tr_embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type) * visual(markersize =20)
    test = AlgebraOfGraphics.data(tst_embed) * mapping(:emb1,:emb2,marker = :cancer_type) * visual(color ="white", strokewidth = 1, strokecolor = "black", markersize =20)
    
    fig = draw(train + test, axis = (;aspect = AxisAspect(1), autolimitaspect = 1, width = 1024, height =1024, 
    title="$(params_dict["model_type"]) on $(params_dict["dataset"]) data\naccuracy by DNN TRAIN: $(round(params_dict["clf_tr_acc"] * 100, digits=2))% TEST: $(round(params_dict["clf_tst_acc"]*100, digits=2))%"))
    CairoMakie.save(fig_outpath, fig)
end

function plot_embed_train_test_3d(model, fold, params_dict)
    fig_outpath= "RES/$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])_scatter.pdf"
    
    X_tr = cpu(model.ae.encoder(gpu(fold["train_x"]')))
    X_tst = cpu(model.ae.encoder(gpu(fold["test_x"]')))
    tr_labels = cpu(y_lbls[fold["train_ids"]])
    tst_labels = cpu(y_lbls[fold["test_ids"]])
    
    fig = Figure(resolution = (1024, 1024));
    az = -0.2
    ax3d = Axis3(fig[1,1], title = "3D inner layer representation of dataset\nAcc% TRAIN: $(round(params_dict["clf_tr_acc"] * 100, digits=2))% TEST: $(round(params_dict["clf_tst_acc"]*100, digits=2))%",  azimuth=az * pi)
    colors = ["#436cde","#2e7048", "#dba527", "#f065eb", "#919191"]
    for (i,lab) in enumerate(unique(tr_labels))
        scatter!(ax3d, X_tr[:,tr_labels .== lab], color = colors[i], label = lab)
        scatter!(ax3d, X_tst[:,tst_labels .== lab], color = colors[i], strokewidth=1)
    end
    test_y = fold["test_y"]
    preds_y = Matrix(cpu(model.clf.model(gpu(fold["test_x"]')) .== maximum(model.clf.model(gpu(fold["test_x"]')), dims =1))')
    nclasses =size(test_y)[2] 
    convertm = reshape(collect(1:nclasses), (nclasses,1))
    test_y = vec(test_y * convertm)
    preds_y = vec(preds_y * convertm)
    scatter!(ax3d, X_tst[:,test_y .!= preds_y], color = "black", marker = [:x for i in 1:sum(test_y .!= preds_y)],markersize = 10, label ="errors")
    axislegend(ax3d)
    CairoMakie.save(fig_outpath,fig)

end

function plot_hexbin_pred_true_ae(model, fold, params_dict)
    fig_outpath= "RES/$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])_hexbin.pdf"
    X_pred = vec(model.net(gpu(fold["test_x"]')))
    X_true = gpu(vec(Matrix(fold["test_x"]')))
    pearson = round(my_cor(X_pred, X_true), digits = 3)
    fig = Figure(resolution = (1024,1024));
    ax = Axis(fig[1,1], aspect = DataAspect(), ylabel = "True Expressions", xlabel = "Predicted Expreessions", title = "Predicted vs True Expression TCGA BRCA \nPearson Cor: $pearson")
    hexbin!(ax, cpu(X_pred), cpu(X_true), cellsize=(0.02,0.02), colormap = cgrad([:grey,:yellow], [0.00000001, 0.1]));
    lines!(ax, [0,1],[0,1], linestyle="--")
    CairoMakie.save(fig_outpath, fig)
end

function plot_hexbin_pred_true_aeclfdnn(model, fold, params_dict)
    fig_outpath= "RES/$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])_hexbin.pdf"
    X_pred = vec(model.ae.net(gpu(fold["test_x"]')))
    X_true = gpu(vec(Matrix(fold["test_x"]')))
    pearson = round(my_cor(X_pred, X_true), digits = 3)
    fig = Figure(resolution = (1024,1024));
    ax = Axis(fig[1,1], aspect = DataAspect(), ylabel = "True Expressions", xlabel = "Predicted Expreessions", title = "Predicted vs True Expression TCGA BRCA \nPearson Cor: $pearson")
    hexbin!(ax, cpu(X_pred), cpu(X_true), cellsize=(0.02,0.02), colormap = cgrad([:grey,:yellow], [0.00000001, 0.1]));
    lines!(ax, [0,1],[0,1], linestyle="--")
    CairoMakie.save(fig_outpath, fig)
end

function plot_embed(X_tr, X_tst, tr_labels, tst_labels, assoc_ae_params,fig_outpath;acc ="tr_acc")
    # plot final 2d embed from Auto-Encoder
    tr_acc = round(assoc_ae_params["clf_tr_acc"], digits = 3)
    tst_acc = round(assoc_ae_params["clf_tst_acc"], digits = 3)
    
    tr_embed = DataFrame(:emb1=>X_tr[1,:], :emb2=>X_tr[2,:], :cancer_type => tr_labels)
    tst_embed = DataFrame(:emb1=>X_tst[1,:], :emb2=>X_tst[2,:], :cancer_type => tst_labels)
    
    train = AlgebraOfGraphics.data(tr_embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type) * visual(markersize =20)
    test = AlgebraOfGraphics.data(tst_embed) * mapping(:emb1,:emb2,marker = :cancer_type) * visual(color ="white", strokewidth = 1, strokecolor = "black", markersize =20)
    
    fig = draw(train + test, axis = (;aspect = AxisAspect(1), autolimitaspect = 1, width = 1024, height =1024, 
    title="$(assoc_ae_params["model_type"]) on $(assoc_ae_params["dataset"]) data\naccuracy by DNN TRAIN: $tr_acc% TEST: $tst_acc%"))
    CairoMakie.save(fig_outpath, fig)
end 

function plot_learning_curves_aeaeclf(learning_curves, assoc_ae_params, fig_outpath)
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), :train_clf_loss=>[i[1] for i in learning_curves], :train_clf_acc => [i[2] for i in learning_curves],
    :train_ae_loss=>[i[3] for i in learning_curves], :train_ae_cor => [i[4] for i in learning_curves],
    :train_ae2_loss=>[i[5] for i in learning_curves], :train_ae2_cor => [i[6] for i in learning_curves],
    :test_clf_loss=>[i[7] for i in learning_curves], :test_clf_acc => [i[8] for i in learning_curves],
    :test_ae_loss=>[i[9] for i in learning_curves], :test_ae_cor => [i[10] for i in learning_curves],
    :test_ae2_loss=>[i[11] for i in learning_curves], :test_ae2_cor => [i[12] for i in learning_curves])
    
    fig = Figure(resolution = (1024, 824))
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    train_ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"train_ae_loss"], color = "red")
    test_ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"test_ae_loss"], linestyle = "--", color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel ="Auto-Encoder 2D MSE loss")
    train_ae2_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"train_ae2_loss"], color = "red")
    test_ae2_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"test_ae2_loss"], linestyle = "--", color = "red")
    fig[3,1] = Axis(fig, xlabel = "steps", ylabel = "Classifier Crossentropy loss")
    train_clf_loss = lines!(fig[3,1], lr_df[:,"step"], lr_df[:,"train_clf_loss"])
    test_clf_loss = lines!(fig[3,1], lr_df[:,"step"], lr_df[:,"test_clf_loss"], linestyle = "--")
    
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder \n Pearson Corr.")
    train_ae_cor = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"train_ae_cor"], color = "red")
    test_ae_cor = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"test_ae_cor"], linestyle = "--",color = "red")
    fig[2,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder 2D \n Pearson Corr.")
    train_ae_cor = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"train_ae2_cor"], color = "red")
    test_ae_cor = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"test_ae2_cor"], linestyle = "--",color = "red")
    
    ax6= Axis(fig[3,2], xlabel = "steps", ylabel = "Classfier Accuracy (%)")
    train_clf_acc = lines!(fig[3,2], lr_df[:,"step"], lr_df[:,"train_clf_acc"] ,label = "train")
    test_clf_acc = lines!(fig[3,2], lr_df[:,"step"], lr_df[:,"test_clf_acc"], linestyle = "--", label = "test") 
    axislegend(ax6, position = :rb)
    
    Label(fig[4,:], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(assoc_ae_params))")
    CairoMakie.save(fig_outpath, fig)
end 

function plot_learning_curves_aecphclf(learning_curves, prms, fig_outpath)
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), 
    :train_ae_loss=>[i[1] for i in learning_curves], :train_ae_cor => [i[2] for i in learning_curves],
    :test_ae_loss=>[i[3] for i in learning_curves], :test_ae_cor => [i[4] for i in learning_curves],
    :train_cph_loss=>[i[5] for i in learning_curves], :train_cph_cind => [i[6] for i in learning_curves],
    :test_cph_loss=>[i[7] for i in learning_curves], :test_cph_cind => [i[8] for i in learning_curves],
    :train_clf_loss=>[i[9] for i in learning_curves], :train_clf_acc => [i[10] for i in learning_curves],
    :test_clf_loss=>[i[11] for i in learning_curves], :test_clf_acc=> [i[12] for i in learning_curves])
    
    fig = Figure(resolution = (1024, 824))
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    train_ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"train_ae_loss"], color = "red")
    test_ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"test_ae_loss"], linestyle = "--", color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel ="Cox NLL loss")
    train_cph_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"train_cph_loss"], color = "blue")
    test_cph_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"test_cph_loss"], linestyle = "--", color = "blue")
    fig[3,1] = Axis(fig, xlabel = "steps", ylabel = "Classifier Crossentropy loss")
    train_clf_loss = lines!(fig[3,1], lr_df[:,"step"], lr_df[:,"train_clf_loss"])
    test_clf_loss = lines!(fig[3,1], lr_df[:,"step"], lr_df[:,"test_clf_loss"], linestyle = "--")
    
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder \n Pearson Corr.")
    train_ae_cor = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"train_ae_cor"], color = "red")
    test_ae_cor = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"test_ae_cor"], linestyle = "--",color = "red")
    fig[2,2] = Axis(fig, xlabel = "steps", ylabel = "CPHDNN\nconcordance index")
    train_ae_cor = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"train_cph_cind"], color = "blue")
    test_ae_cor = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"test_cph_cind"], linestyle = "--",color = "blue")
    
    ax6= Axis(fig[3,2], xlabel = "steps", ylabel = "Classfier Accuracy (%)")
    train_clf_acc = lines!(fig[3,2], lr_df[:,"step"], lr_df[:,"train_clf_acc"] .* 100,label = "train")
    test_clf_acc = lines!(fig[3,2], lr_df[:,"step"], lr_df[:,"test_clf_acc"] .* 100 , linestyle = "--", label = "test") 
    axislegend(ax6, position = :rb)
    
    Label(fig[4,:], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(prms))")
    CairoMakie.save(fig_outpath, fig)
end 

function plot_learning_curves_aeclf(learning_curves, assoc_ae_params, fig_outpath)
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), :train_clf_loss=>[i[1] for i in learning_curves], :train_clf_acc => [i[2] for i in learning_curves],
    :train_ae_loss=>[i[3] for i in learning_curves], :train_ae_cor => [i[4] for i in learning_curves],
    :test_clf_loss=>[i[5] for i in learning_curves], :test_clf_acc => [i[6] for i in learning_curves],
    :test_ae_loss=>[i[7] for i in learning_curves], :test_ae_cor => [i[8] for i in learning_curves])
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    train_ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"train_ae_loss"], color = "red")
    test_ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"test_ae_loss"], linestyle = "--", color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "Classifier Crossentropy loss")
    train_clf_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"train_clf_loss"])
    test_clf_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"test_clf_loss"], linestyle = "--")
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder \n Pearson Corr.")
    train_ae_cor = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"train_ae_cor"], color = "red")
    test_ae_cor = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"test_ae_cor"], linestyle = "--",color = "red")
    ax4= Axis(fig[2,2], xlabel = "steps", ylabel = "Classfier Accuracy (%)")
    train_clf_acc = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"train_clf_acc"] ,label = "train")
    test_clf_acc = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"test_clf_acc"], linestyle = "--", label = "test") 
    axislegend(ax4, position = :rb)
    Label(fig[3,:], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(assoc_ae_params))")
    CairoMakie.save(fig_outpath, fig)
end 

struct EncCPHDNN_LearningCurve
    tr_loss::Array
    tr_c_index::Array
    tst_loss::Array
    tst_c_index::Array
    
end 
function EncCPHDNN_LearningCurve(learning_curve::Array)
    return EncCPHDNN_LearningCurve([i[1] for i in learning_curve], 
    [i[2] for i in learning_curve],
    [i[3] for i in learning_curve],
    [i[4] for i in learning_curve])
end 

function plot_learning_curves(LC::EncCPHDNN_LearningCurve, model_params::Dict, fig_outpath)
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(LC.tr_loss)),
    :tr_loss=>LC.tr_loss, :tr_c_index => LC.tr_c_index, 
    :tst_loss=>LC.tst_loss, :tst_c_index => LC.tst_c_index)
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Cox-Negative Likelihood loss")
    cox_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"tr_loss"], color = "blue")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "Concordance index")
    c_index = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"tr_c_index"], color = "blue")
    ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"tst_loss"], color ="blue", linestyle = "--")
    ae_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"tst_c_index"], color = "blue", linestyle = "--")
    Label(fig[3,:], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(model_params))")
    CairoMakie.save(fig_outpath, fig)
end


function plot_learning_curves(learning_curves, params_dict::Dict, fig_outpath)
    
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), :ae_loss=>[i[1] for i in learning_curves], :ae_cor => [i[2] for i in learning_curves],
    :cph_loss=>[i[3] for i in learning_curves], :cind_tr=> [i[4] for i in learning_curves],
    :ae_loss_test=>[i[5] for i in learning_curves], :ae_cor_test => [i[6] for i in learning_curves],
    :cph_loss_test=>[i[7] for i in learning_curves], :cind_test=> [i[8] for i in learning_curves])
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    ae_loss_tr = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"ae_loss"], color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "CPH Cox-Negative Likelihood")
    cph_loss_tr = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"cph_loss"])
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder \nPearson Corr.")
    ae_cort_tr = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"ae_cor"], color = "red")
    fig[2,2] = Axis(fig, xlabel = "steps", ylabel = "CPH Concordance index")
    cind_tr = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"cind_tr"]  )
    ae_loss_test = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"ae_loss_test"], color = "red", linestyle = "--")
    ae_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"cph_loss_test"], color ="blue", linestyle = "--")
    ae_loss = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"ae_cor_test"], color = "red", linestyle = "--")
    ae_loss = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"cind_test"] , color ="blue", linestyle = "--")
    Label(fig[3,:], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(params_dict)))")
    CairoMakie.save(fig_outpath, fig)
end 