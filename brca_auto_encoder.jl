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
device()
sum(sum(brca_prediction.data, dims = 1) .== 0)
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
    "nb_clinf"=>5, "cph_lr" => 1e-4, "cph_nb_hl" => 1, "cph_hl_size" => 64)
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
    CairoMakie.save("$outpath/AE_BRCA_BY_DIM_REDUX.pdf", fig)
    ### Provide hexbin of true vs predicted expr. profiles
    fig = Figure(resolution = (1024,1024));
    ax = Axis(fig[1,1];xlabel="Predicted", ylabel = "True Expr.", title = "Predicted vs True of $(brca_ae_params["ngenes"]) Genes Expression Profile TCGA BRCA with AE \n$(round(ae_cor_test;digits =3))", aspect = DataAspect())
    hexbin!(fig[1,1], outs, test_xs, cellsize=(0.02, 0.02), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
    CairoMakie.save("$outpath/AE_BRCA_SCATTER_DIM_REDUX_$dim_redux.pdf", fig)
end 
dim_redux = 300 
nfolds, ae_nb_hls = 5, 1
nepochs = 1000
##### AE only BRCA
 brca_ae_params = Dict("model_title"=>"AE_BRCA_DIM_REDUX_$dim_redux", "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
 "model_type" => "auto_encoder", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
 "nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes),  
 "nfolds" => 5,  "nepochs" => nepochs, "mb_size" => 200, "ae_lr" => 1e-3, "wd" => 5e-5, "dim_redux" => dim_redux, 
 "ae_hl_size" => 128, "dec_nb_hl" => ae_nb_hls, "dec_hl_size" => 128, "enc_nb_hl" =>ae_nb_hls, 
 "nb_clinf"=>5, "cph_lr" => 1e-4, "cph_nb_hl" => 1, "cph_hl_size" => 64)
 dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")


model = build(brca_ae_params;adaptative=false)
outs, test_xs, model, train_x, test_x = validate_auto_encoder!(brca_ae_params, brca_prediction, dump_cb_brca, clinf) 
# wd = 2e-5, dim_redux = 100    wd = 5e-5, dim_redux = 300

dim_redux_sizes = [1,2,3,4,5,10,15,20,30,40,50,75,100,200,300,500,1000,2000]
i = 1
 train_corrs = []
tst_corrs = []
 ae_cor_train =  my_cor(vec(train_x), vec(model.net(train_x)))
 ae_cor_test = my_cor(vec(test_x), vec(model.net(test_x)))
 push!(train_corrs, ae_cor_train)
 push!(tst_corrs, ae_cor_test)
 ### Update benchmark figure
 fig = Figure(resolution = (1024,512));

 ax = Axis(fig[1,1];xticks=(log10.(dim_redux_sizes), ["$x" for x in dim_redux_sizes]), xlabel = "Bottleneck layer size (log10 scale)",ylabel = "Test set reconstruction \n(Pearson Correlation)", title = "Performance of Auto-Encoder on BRCA gene expression profile by bottleneck layer size")
 data_points = zeros(length(dim_redux_sizes))
 data_points[1:i] .= tst_corrs
 scatter!(fig[1,1], log10.(dim_redux_sizes), data_points, color = "blue")
 lines!(fig[1,1], log10.(dim_redux_sizes), data_points, color = "black", linestyle = "--")
fig 

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