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
getcurrentcommitnb(;digits = 7) =  read(`git rev-parse HEAD`, String)[1:digits]
zpad(n::Int;pad::Int=9) = lpad(string(n),pad,'0')
function labs_appdf(labs) 
    lbd = Dict([(t,sum(labs .== t )) for t in unique(labs)])
    append(lbl, lbd)= "$lbl ($(lbd[lbl]))" 
    labs_appd = [append(lb,lbd) for lb in labs]
    return labs_appd
end 

function stringify(p::Dict;spacer = 80)  
    s = join(["$key: $val" for (key, val) in p], ", ")
    for i in collect(spacer:spacer:length(s))
        s = "$(s[1:i])\n$(s[i:end])"
    end
    return s 
end 

##########################################
####### Plotting functions    ############
##########################################

function plot_embed(X_tr, labels, assoc_ae_params,fig_outpath;acc ="tr_acc")
    # plot final 2d embed from Auto-Encoder
    tr_acc = round(assoc_ae_params[acc], digits = 3) * 100
    embed = DataFrame(:emb1=>X_tr[1,:], :emb2=>X_tr[2,:], :cancer_type => labels)
    p = AlgebraOfGraphics.data(embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type) * visual(markersize =20)
    fig = draw(p, axis = (;aspect = AxisAspect(1), autolimitaspect = 1, width = 1024, height =1024, 
    title="$(assoc_ae_params["model_type"]) on $(assoc_ae_params["dataset"]) data\naccuracy by DNN : $tr_acc%"))
    CairoMakie.save(fig_outpath, fig)
end 

function plot_learning_curves(learning_curves, assoc_ae_params, fig_outpath)
    # learning curves 
    lr_df = DataFrame(:step => collect(1:length(learning_curves)), :ae_loss=>[i[1] for i in learning_curves], :ae_cor => [i[2] for i in learning_curves],
    :clf_loss=>[i[3] for i in learning_curves], :clf_acc => [i[4] for i in learning_curves])
    fig = Figure()
    fig[1,1] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder MSE loss")
    ae_loss = lines!(fig[1,1], lr_df[:,"step"], lr_df[:,"ae_loss"], color = "red")
    fig[2,1] = Axis(fig, xlabel = "steps", ylabel = "Classifier Crossentropy loss")
    ae_loss = lines!(fig[2,1], lr_df[:,"step"], lr_df[:,"clf_loss"])
    fig[1,2] = Axis(fig, xlabel = "steps", ylabel = "Auto-Encoder Pearson Corr.")
    ae_loss = lines!(fig[1,2], lr_df[:,"step"], lr_df[:,"ae_cor"], color = "red")
    fig[2,2] = Axis(fig, xlabel = "steps", ylabel = "Classfier Accuracy (%)")
    ae_loss = lines!(fig[2,2], lr_df[:,"step"], lr_df[:,"clf_acc"] .* 100 )
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