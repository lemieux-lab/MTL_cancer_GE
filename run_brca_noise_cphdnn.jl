include("init.jl")
include("data_processing.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("mtl_engines.jl")
include("cross_validation.jl")
device!()
### 0 ####
outpath, session_id = set_dirs() 
hostname = strip(read(`hostname`, String))
infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
brca_prediction = BRCA_data(infile, minmax_norm = true)
clinf = assemble_clinf(brca_prediction)
TCIA_data = CSV.read("Data/GDC_processed/TCIA-ClinicalData.tsv", DataFrame)
barcode = [join(split(x, "-")[1:3],"-") for x in clinf[:,"samples"]]
clinf[:,"barcode"] .= barcode
PAM50_data = leftjoin(clinf, TCIA_data[:,["barcode","clinical_data_PAM50MRNA"]], on =:barcode)
PAM50_data = PAM50_data[sortperm(PAM50_data[:,"samples"]),:]
#counter("clinical_data_PAM50MRNA",PAM50_data)
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
    "device"=>device(),
    "hostname"=>hostname,
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

CairoMakie.save("$outpath/0C_TSNE_BRCA_PAM50.pdf",fig)
#3.C
noise_sizes = [1,2,3,4,5,10,15,20,30,40,50,75,100,200,300,400,500,1000,2000]
c_inds = []
nfolds, nepochs = 5,10_000
for (i, noise_size) in enumerate(noise_sizes)
    brca_prediction_NOISE = BRCA_data(reshape(rand(1050 *noise_size), (1050,noise_size)),brca_prediction.samples, brca_prediction.genes, brca_prediction.survt,brca_prediction.surve,brca_prediction.age, brca_prediction.stage, brca_prediction.ethnicity)
    brca_cphdnnclinf_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
    "model_type" => "cphdnnclinf", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
    "nsamples" => length(brca_prediction.samples) , "insize" => noise_size, 
    "nfolds" => nfolds,  "nepochs" =>nepochs, "wd" =>  2e-2,  
    "nb_clinf" => 5,"cph_lr" => 1e-4, "cph_nb_hl" => 1, "cph_hl_size" => 64)
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
        CairoMakie.save("$outpath/3C_CPHDNNCLINF_BRCA_BY_NB_DIM.pdf", fig)
    end 
end 