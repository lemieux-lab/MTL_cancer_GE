include("init.jl")
include("data_processing.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("mtl_engines.jl")
include("cross_validation.jl")
# 0 - Presentation of Data
# 0.A => Data table / bar plot (n, c, nbgenes, PAM50 subt., stages, ages) # 30-45 min 
# 0.B => Surv curves vs PAM50 subtype vs stage vs age # 1h
# 0.C => T-SNE(PCA init) vs PAM50 subtype + %Accuracy (DNN) # 20 min 
# 1 - Proof of concept AE-DNN for subtype classification
# 1.A => 2D AE-DNN PAM50 vs PAM50 subtype + %Accuracy # 1h 
# 1.B => AE-DNN True vs. predicted expression reconstruction + Pearson Cor # 10 min 
# 2 - AE Bottleneck size vs Pearson Cor 
# 2.A => AE with BN size = 1,2,3,4,5...1000  vs Pearson Cor train + test # 0
# 2.B => True vs. predicted expression reconstruction + Pearson Cor # 0 min 
# 3 - CPHDNN Added noisy features vs c-index
# 3.A => CPH clin. f. Surv Curves hi-risk vs lo-risk + C-index (95% interval) # 20 min  
# 3.B => CPHDNN clin. f. Surv Curves hi-risk vs lo-risk + C-index (95% interval) # 20 min 
# 3.C => CPHDNN on clinf + 1,2,3,4,5...1000 extra random features vs mean C-index train + test # 2h
# 4 - CPHDNN-AE training  
# 4.A => Learning curves # 1h
# 4.B => True vs. predicted expression reconstruction + Pearson Cor # 10 min
# 4.C => Surv Curves hi-risk vs lo-risk + C-index (95% interval) # 10 min 

### 0 ####
outpath, session_id = set_dirs() 
infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
brca_prediction = BRCA_data(infile, minmax_norm = true)
clinf = assemble_clinf(brca_prediction)
TCIA_data = CSV.read("Data/GDC_processed/TCIA-ClinicalData.tsv", DataFrame)
barcode = [join(split(x, "-")[1:3],"-") for x in clinf[:,"samples"]]
clinf[:,"barcode"] .= barcode
PAM50_data = leftjoin(clinf, TCIA_data[:,["barcode","clinical_data_PAM50MRNA"]], on =:barcode)
PAM50_data = PAM50_data[sortperm(PAM50_data[:,"samples"]),:]
counter("clinical_data_PAM50MRNA",PAM50_data)
### 0.C ###
brca_tsne = tsne(brca_prediction.data,2, 50, 3000, 30; verbose = true, progress = true)
TSNE_df = DataFrame(Dict(:tsne_1=>brca_tsne[:,1], :tsne_2=>brca_tsne[:,2]))
TSNE_df[:,"samples"] .= brca_prediction.samples
TSNE_df = leftjoin(TSNE_df, PAM50_data, on = :samples)
keep = TSNE_df[:, "clinical_data_PAM50MRNA"] .!= "NA"
TSNE_df = TSNE_df[keep,:]
## DNN prediction Performance
lbls = label_binarizer(TSNE_df[:, "clinical_data_PAM50MRNA"])
x_data = Matrix(TSNE_df[:,["tsne_1", "tsne_2"]])
y_data = lbls
brca_clfdnn_params = Dict(
    "modelid" =>"$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "session_id" => session_id,
    "model_type" => "clfdnn",
    "nfolds" => 5,
    "nepochs" => 3000,
    "lr" => 1e-3,
    "wd" => 1e-3,
    "nb_hl" => 2,
    "hl_size" => 128,
    "n.-lin" => relu, 
    "insize" => size(x_data)[2],
    "nsamples" => size(x_data)[1], 
    "outsize" => size(y_data)[2]
)
x_preds, test_xs = validate_clfdnn!(brca_clfdnn_params,Matrix(TSNE_df[:,["tsne_1", "tsne_2"]]), lbls, TSNE_df[:,"samples"]; nfolds= brca_clfdnn_params["nfolds"])
TSNE_2d_clfdnn_acc = accuracy(test_xs, x_preds)
brca_clfdnn_params = Dict(
    "modelid" =>"$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "session_id" => session_id,
    "model_type" => "clfdnn","nfolds" => 5,"nepochs" => 3000,"lr" => 1e-3,"wd" => 1e-3,
    "nb_hl" => 2,"hl_size" => 128, "n.-lin" => leakyrelu, "insize" => size(brca_prediction.data)[2],
    "nsamples" => size(brca_prediction.data[keep,:])[1], "outsize" => size(y_data)[2]
)
x_preds, test_xs = validate_clfdnn!(brca_clfdnn_params,Matrix(brca_prediction.data[keep,:]), lbls, brca_prediction.samples[keep]; nfolds= brca_clfdnn_params["nfolds"])
trscr_clfdnn_acc = accuracy(test_xs, x_preds)


TSNE_df[:,"lbls"] = add_n_labels(TSNE_df[:,"clinical_data_PAM50MRNA"], counter("clinical_data_PAM50MRNA",TSNE_df))
p = AlgebraOfGraphics.data(TSNE_df) * mapping(:tsne_1, :tsne_2, color = :lbls, marker = :lbls)
fig = draw(p;axis= (;title = "2D TSNE with PCA init (50D) n=$(length(brca_prediction.samples[keep]))\nDNN Classification Accuracy: \n $(length(brca_prediction.genes)) features input: $(round(trscr_clfdnn_acc,digits=1))% | TSNE $(round(TSNE_2d_clfdnn_acc,digits=1))% ", aspect = DataAspect()))
fig
CairoMakie.save("$outpath/0C_TSNE_BRCA_PAM50.pdf",fig)

### 1 ####
### 1.A ##
##### AE only BRCA
x_data = brca_prediction.data[keep,:]
nfolds, ae_nb_hls, dim_redux, nepochs = 5, 1, 2, 10_000
brca_aeclfdnn_params = Dict("model_title"=>"AE_CLF_BRCA_2D", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "aeclfdnn", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
"nfolds" => 5,  "nepochs" => nepochs, "ae_lr" => 1e-4, "wd" => 1e-3, "dim_redux" => dim_redux, 
"ae_hl_size"=>128,"enc_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, "n.-lin" => leakyrelu,
"clfdnn_lr" => 1e-3, "clfdnn_nb_hl" => 2, "clfdnn_hl_size" => 64, "outsize" => size(y_data)[2])
brca_clf_cb = dump_model_cb(1000, TSNE_df[:,"lbls"], export_type = "pdf")
validate_aeclfdnn!(brca_aeclfdnn_params, x_data, y_data, brca_prediction.samples[keep], brca_clf_cb)

function validate_aeclfdnn!(params_dict, x_data, y_data, samples, dump_cb_brca;build_adaptative=false,nfolds=5,device =gpu)
    folds = split_train_test(x_data, y_data, samples;nfolds = nfolds)
    mkdir("RES/$(params_dict["session_id"])/$(params_dict["modelid"])")
    x_pred_by_fold, test_xs = [],[]
    [mkdir("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/params.bson",params_dict)
    for fold in folds
        model = build(params_dict;adaptative=build_adaptative)
        ## STATIC VARS    
        nepochs= params_dict["nepochs"]
        wd = params_dict["wd"]
        train_x = device(Matrix(fold["train_x"]'));
        train_y = device(Matrix(fold["train_y"]'));
        test_x = device(Matrix(fold["test_x"]'));
        test_y = device(Matrix(fold["test_y"]'));

        nsamples = size(train_y)[2]    
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            ## gradient Auto-Encoder 
            ps = Flux.params(model.ae.net)
            gs = gradient(ps) do
                model.ae.lossf(model.ae, train_x, train_x, weight_decay = wd)
            end
            Flux.update!(model.ae.opt, ps, gs)
            ## gradient Classfier DNN
            ps = Flux.params(model.clf.model)
            gs = gradient(ps) do 
                model.clf.lossf(model.clf, train_x, train_y, weight_decay = wd)
            end
            Flux.update!(model.clf.opt, ps, gs)

            train_clf_loss = round(model.clf.lossf(model.clf, train_x, train_y, weight_decay = wd),digits=3)
            train_clf_acc =  round(accuracy(train_y, model.clf.model(train_x)),digits = 3)
            train_ae_loss = round(model.ae.lossf(model.ae, train_x, train_x, weight_decay = wd),digits=3)
            train_ae_cor = round(my_cor(vec(train_x), vec(model.ae.net(train_x))),digits=3)
            
            params_dict["clf_tr_acc"] = train_clf_acc
            test_clf_loss = round(model.clf.lossf(model.clf, test_x, test_y, weight_decay = wd), digits = 3)
            test_clf_acc = round(accuracy(test_y, model.clf.model(test_x)),digits = 3)
            test_ae_loss =  round(model.ae.lossf(model.ae, test_x, test_x, weight_decay = wd),digits=3)
            test_ae_cor = round(my_cor(vec(test_x), vec(model.ae.net(test_x))),digits=3)
            
            params_dict["clf_tst_acc"] = test_clf_acc
            push!(learning_curve, (train_clf_loss, train_clf_acc, train_ae_loss, train_ae_cor, test_clf_loss, test_clf_acc, test_ae_loss, test_ae_cor))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN CLF loss $(round(train_clf_loss,digits =3)) \t acc.%: $(round(train_clf_acc, digits = 3))\tAE loss: $train_ae_loss \tcor: $train_ae_cor\t TEST CLF loss $(round(test_clf_loss,digits =3)) \t acc.%: $(round(test_clf_acc, digits = 3)) AE loss: $test_ae_loss \t cor $test_ae_cor")
            dump_cb_brca(model, learning_curve, params_dict, iter, fold)
        end
        push!(x_pred_by_fold, Matrix(cpu(model.clf.model(test_x))'))
        push!(test_xs, Matrix(cpu(test_y)'))
    end
    concat_OUTs = Matrix(vcat(x_pred_by_fold...)')
    concat_tests = Matrix(vcat(test_xs...)')

    return concat_OUTs, concat_tests#Dict(:tr_acc=>accuracy(concat_tests,concat_OUTs))
end 


### 2 ####
### 2.A ##
nfolds, ae_nb_hls = 5, 1
dim_redux_sizes = [1,2,3,4,5,10,15,20,30,50,100,200,300,500,1000, 2000]
train_corrs = []
tst_corrs = []
nepochs = 3000
for (i,dim_redux) in enumerate(dim_redux_sizes)
    ##### AE only BRCA
    brca_ae_params = Dict("model_title"=>"AE_BRCA_DIM_REDUX_$dim_redux", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
    "model_type" => "auto_encoder", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
    "nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
    "nfolds" => 5,  "nepochs" => nepochs, "mb_size" => 200, "ae_lr" => 1e-3, "wd" => 1e-4, "dim_redux" => dim_redux, 
    "enc_hl_size" => "adaptative", "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => "adaptative", "enc_nb_hl" =>ae_nb_hls, 
    "nb_clinf"=>5)
    dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
    #model = build(brca_ae_params)
    outs, test_xs, model, train_x, test_x = validate_auto_encoder!(brca_ae_params, brca_prediction, dump_cb_brca, clinf; build_adaptative= true) 
    ae_cor_train =  my_cor(vec(train_x), vec(model.net(train_x)))
    ae_cor_test = my_cor(vec(test_x), vec(model.net(test_x)))
    push!(train_corrs, ae_cor_train)
    push!(tst_corrs, ae_cor_test)
    ### Update benchmark figure
    fig = Figure(resolution = (1024, 512));
    ax = Axis(fig[1,1];xticks=(log10.(dim_redux_sizes[1:i]), ["$x" for x in dim_redux_sizes[1:i]]), xlabel = "Bottleneck layer size (log10 scale)",ylabel = "Test set reconstruction \n(Pearson Correlation)", title = "Performance of Auto-Encoder on BRCA gene expression profile by bottleneck layer size")
    test_points = zeros(length(dim_redux_sizes[1:i]))
    train_points = zeros(length(dim_redux_sizes[1:i]))
    test_points[1:i] .= tst_corrs
    train_points[1:i] .= train_corrs
    scatter!(fig[1,1], log10.(dim_redux_sizes[1:i]), test_points[1:i], color = "blue", label = "test")
    scatter!(fig[1,1], log10.(dim_redux_sizes[1:i]), train_points[1:i], color = "red", label = "train")
    lines!(fig[1,1], log10.(dim_redux_sizes[1:i]), test_points[1:i], color = "blue", linestyle = "--")
    lines!(fig[1,1], log10.(dim_redux_sizes[1:i]), train_points[1:i], color = "red", linestyle = "--")
    
    Label(fig[2,1], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(brca_ae_params))")
    axislegend(ax, position = :rb)
    CairoMakie.save("$outpath/2A_AE_BRCA_BY_DIM_REDUX.pdf", fig)
    ### Provide hexbin of true vs predicted expr. profiles
    fig = Figure(resolution = (1024,1024));
    ax = Axis(fig[1,1];xlabel="Predicted", ylabel = "True Expr.", title = "Predicted vs True of $(brca_ae_params["ngenes"]) Genes Expression Profile TCGA BRCA with AE \n$(round(ae_cor_test;digits =3))", aspect = DataAspect())
    hexbin!(fig[1,1], outs, test_xs, cellsize=(0.02, 0.02), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
    CairoMakie.save("$outpath/2B_AE_BRCA_SCATTER_DIM_REDUX_$dim_redux.pdf", fig)
end
