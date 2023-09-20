include("init.jl")
include("data_processing.jl")
include("utils.jl")
include("SurvivalDev.jl")
include("mtl_engines.jl")
# 0 - Presentation of Data
# 0.A => Data table / bar plot (n, c, nbgenes, PAM50 subt., stages, ages) # 30-45 min 
# 0.B => Surv curves vs PAM50 subtype vs stage vs age # 1h
# 0.C => T-SNE(PCA init) vs PAM50 subtype + %Accuracy (DNN) # 20 min 
# 1 - Proof of concept AE-DNN for subtype classification
# 1.B => 2D AE-DNN PAM50 vs PAM50 subtype + %Accuracy # 1h 
# 1.C => AE-DNN True vs. predicted expression reconstruction + Pearson Cor # 10 min 
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
TSNE_df = TSNE_df[TSNE_df[:, "clinical_data_PAM50MRNA"] .!= "NA",:]
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
accuracy(test_xs, x_preds)

p = AlgebraOfGraphics.data(TSNE_df) * mapping(:tsne_1, :tsne_2, color = :clinical_data_PAM50MRNA, marker = :clinical_data_PAM50MRNA)
fig = draw(p;axis= (;title = "2D TSNE with PCA init (50D)", aspect = DataAspect()))
CairoMakie.save("$outpath/0C_TSNE_BRCA_PAM50.pdf",fig)

### 1 ####
counter(data, feature) = Dict([(x, sum(data[:,feature].==x)) for x in unique(data[:,feature])])
brca_prediction.samples
unique(TCIA_data[:,"clinical_data_PAM50MRNA"])
sum(TCIA_data[:,"clinical_data_PAM50MRNA"] .== "NA")
brca_prediction.samples
clinf[:, "samples"]