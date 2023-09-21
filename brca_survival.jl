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
brca_prediction = BRCA_data(infile,minmax_norm = true)

##### PCA
using MultivariateStats
using TSne
maximum(brca_prediction.data')
pca = fit(PCA, brca_prediction.data', maxoutdim=25, method = :cov);
brca_prediction.stage
brca_prediction_pca = predict(pca, brca_prediction.data')';
brca_pca_25_df = DataFrame(Dict([("pc_$i",brca_prediction_pca[:,i]) for i in 1:size(brca_prediction_pca)[2]]))
brca_pca_25_df[:,"case_id"] .= brca_prediction.samples 
CSV.write("RES/SURV/TCGA_BRCA_pca_25_case_ids.csv", brca_pca_25_df)
#CSV.write("RES/SURV/TCGA_BRCA_clinical_survival.csv", BRCA_CLIN)
# CSV tcga brca surv + clin data 
brca_pca = CSV.read("RES/SURV/TCGA_brca_pca_25_case_ids.csv", DataFrame)

brca_tsne = tsne(brca_prediction.data,2, 0, 1000, 30;pca_init = false, verbose = true, progress = true)

brca_pca[:,"is_her2"] .= subgroups .== "HER2-enriched"
brca_pca[:,"is_lumA"] .= subgroups .== "Luminal A"
brca_pca[:,"is_lumB"] .= subgroups .== "Luminal B"
brca_pca[:,"is_bsllk"] .= subgroups .== "Basal-like"
brca_pca[:,"survt"] .= brca_prediction.survt ./ 10_000
brca_pca[:,"surve"] .= brca_prediction.surve
clinf = assemble_clinf(brca_prediction)
function plabls(stages)  
    stages_dict = Dict("Stage I" =>"stage_i", "Stage IA" =>"stage_i", "Stage IB"=>"stage_i", "Stage IC"=>"stage_i",
    "Stage II" =>"stage_ii", "Stage IIA" =>"stage_ii", "Stage IIB"=>"stage_ii", "Stage IIC"=>"stage_ii",
    "Stage III" =>"stage_iii+", "Stage IIIA" =>"stage_iii+", "Stage IIIB"=>"stage_iii+", "Stage IIIC"=>"stage_iii+",
    "Stage IV" =>"stage_iii+", "Stage X" => "stage_iii+", "'--"=>"NA")
    return [stages_dict[x] for x in stages]
end 
brca_prediction
brca_tsne_df = DataFrame(Dict(:tsne_1=>brca_tsne[:,1], :tsne_2=>brca_tsne[:,2], :group=>plabls(brca_prediction)))
p = AlgebraOfGraphics.data(brca_tsne_df) * mapping(:tsne_1,:tsne_2, color = :group)
draw(p)


folds = split_train_test(Matrix(brca_pca[:,collect(1:25)]), brca_prediction.survt, brca_prediction.surve, vec(brca_pca[:,26]);nfolds =5)



scores = vec(cpu(mdl(X_test)))
median(scores)
groups = ["low_risk" for i in 1:length(scores)]
high_risk = scores .> median(scores)
low_risk = scores .<= median(scores)
median(scores[high_risk])
median(scores[low_risk])


sum(high_risk)
groups[high_risk] .= "high_risk"
p_high, x_high, sc1_high, sc2_high = surv_curve(cpu(Y_t_test)[high_risk], cpu(Y_e_test)[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(cpu(Y_t_test)[low_risk], cpu(Y_e_test)[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low) 


concat_outs = vcat(outs...)
concat_survts = vcat(survts...)
concat_surves = vcat(surves...)
nsamples = length(concat_survts)
cis = []
bootstrapn = 1000
for i in 1:bootstrapn
sampling = rand(1:nsamples, nsamples)
push!(cis, concordance_index(concat_outs[sampling], concat_survts[sampling], concat_surves[sampling]))
end 
sorted_accs = sort(cis)
low_ci, med, upp_ci = sorted_accs[Int(round(bootstrapn * 0.025))], median(sorted_accs), sorted_accs[Int(round(bootstrapn * 0.975))]

concordance_index(concat_outs, concat_survts, concat_surves)

scores = concat_outs
median(scores)
groups = ["low_risk" for i in 1:length(scores)]
high_risk = scores .> median(scores)
low_risk = scores .<= median(scores)
groups[high_risk] .= "high_risk"
p_high, x_high, sc1_high, sc2_high = surv_curve(concat_survts[high_risk], concat_surves[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(concat_survts[low_risk], concat_surves[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low)    
fig = Figure();
ticks = collect(0:250:end_of_study)
Stf_hat_labels = ["$i\n$(label)" for (i,label) in zip(ticks, get_Stf_hat_surv_rates(ticks, sc1_high))] 
ylow = 0.2
lrt_pval = round(log_rank_test(concat_survts, concat_surves, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)

caption = "High risk vs Low risk based on median of risk scores by CPHDNN \n on TCGA BRCA data using 25 PCA from Gene Expression\n Log-rank-test pval = $lrt_pval"
            
ax = Axis(fig[1,1], limits = (0,end_of_study, ylow, 1.05), 
                yminorticksvisible = true, yminorgridvisible = true, yminorticks = IntervalsBetween(2),
                yticks = collect(0:10:100) ./ 100,
                xticks = (ticks, Stf_hat_labels),
                xlabel = "Elapsed time (days)",
                ylabel = "Survival (fraction still alive)",
                titlesize = 14, 
                xticklabelsize =11, 
                yticklabelsize =11, 
                ylabelsize = 14, 
                xlabelsize = 14,
                title = caption)
# plot lines
lines!(ax, Array{Int64}(sc1_low[sc1_low.e .== 1,:tf]), sc1_low[sc1_low.e .== 1, :Stf_hat], color = "blue", label = "low risk (scores < median)") 
conf_tf, lower_95, upper_95 = get_95_conf_interval(sc2_low.tf, sc2_low.nf, sc2_low.Stf_hat, end_of_study)
lines!(ax, Array{Int64}(conf_tf), upper_95, linestyle = :dot, color = "blue")
lines!(ax, Array{Int64}(conf_tf), lower_95, linestyle = :dot, color = "blue")
fill_between!(ax, conf_tf, lower_95, upper_95, color = ("blue", 0.1))
# plot censored
# scatter!(ax, sc1_low[sc1_low.e .== 0,:tf], sc1_low[sc1_low.e .== 0, :Stf_hat], marker = [:vline for i in 1:sum(sc1_low.e .== 0)], color = "black")
lines!(ax, sc1_high[sc1_high.e .== 1,:tf], sc1_high[sc1_high.e .== 1, :Stf_hat], color = "red", label = "high risk (scores > median)") 
conf_tf, lower_95, upper_95 = get_95_conf_interval(sc2_high.tf, sc2_high.nf, sc2_high.Stf_hat, end_of_study)
lines!(ax, conf_tf, upper_95, linestyle = :dot, color = "red")
lines!(ax, conf_tf, lower_95, linestyle = :dot, color = "red")
fill_between!(ax, conf_tf, lower_95, upper_95, color = ("red", 0.1))
axislegend(ax, position = :rb, labelsize = 11, framewidth = 0)
                
fig
CairoMakie.save("$outpath/aggregated_scores_high_vs_low_CPHDNN_TCGA_BRCA.pdf",fig)
fig = Figure();
scatter(fig[1,1], ones(sum(high_risk)), log10.(scores[high_risk]), color = "red")
scatter!(fig[1,1], ones(sum(low_risk)), log10.(scores[low_risk]), color = "blue")
fig

fig = Figure();
median(scores)
hist(fig[1,1], scores[high_risk], color = "red")
hist(fig[1,1], scores[low_risk], color = "blue")
hist(fig[1,1], concat_outs)

fig
mean(scores[low_risk])
mean(scores[high_risk])
scores = vec(cpu(mdl(X)))
median(scores)
high_risk = scores .> median(scores)
low_risk = scores .<= median(scores)
cpu(Y_e)[high_risk]
cpu(Y_t)[high_risk]
cpu(Y_e)[low_risk]
p_high, x_high, sc1_high, sc2_high = surv_curve(cpu(Y_t)[high_risk], cpu(Y_e)[high_risk]; color = "red")
p_low, x_low, sc1_low, sc2_low = surv_curve(cpu(Y_t)[low_risk], cpu(Y_e)[low_risk]; color = "blue")
draw(p_high + x_high + p_low + x_low)            

##### ST-CPH-ridge on clinical (BENCHMARK)
brca_cphclinf_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "cphclinf", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes), 
 "nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50,"wd" => 1e-3, "enc_nb_hl" =>ae_nb_hls, "enc_hl_size" => 128, "dim_redux"=> 1, 
"nb_clinf" => 5,"cph_lr" => 1e-3, "cph_nb_hl" => 2, "cph_hl_size" => 128)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
#validate_cphdnn_clinf!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)
validate_cphdnn_clinf!(brca_cphclinf_params, brca_prediction, dummy_dump_cb, clinf)



##### ST-CPHDNN on clinical (BENCHMARK)
noise_sizes = [1,2,3,4,5,10,15,20,30,40,50,75,100,200,300,400,500,1000,2000]
c_inds = []
nepochs = 5_000
for (i, noise_size) in enumerate(noise_sizes)
    brca_prediction_NOISE = BRCA_data(reshape(rand(1050 *noise_size), (1050,noise_size)),brca_prediction.samples, brca_prediction.genes, brca_prediction.survt,brca_prediction.surve,brca_prediction.age, brca_prediction.stage, brca_prediction.ethnicity)
    brca_cphdnnclinf_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
    "model_type" => "cphdnnclinf", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
    "nsamples" => length(brca_prediction.samples) , "insize" => noise_size, 
    "nfolds" => 5,  "nepochs" =>nepochs, "wd" =>  2e-2,  
    "nb_clinf" => 5,"cph_lr" => 1e-4, "cph_nb_hl" => 1, "cph_hl_size" => 16)
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
        CairoMakie.save("$outpath/CPHDNNCLINF_BRCA_BY_NB_DIM.pdf", fig)
    end 
end 

noise_size = 2
brca_prediction_NOISE = BRCA_data(reshape(rand(1050 *noise_size), (1050,noise_size)),brca_prediction.samples, brca_prediction.genes, brca_prediction.survt,brca_prediction.surve,brca_prediction.age, brca_prediction.stage, brca_prediction.ethnicity)
brca_cphdnnclinf_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "cphdnnclinf", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => noise_size, 
"nfolds" => 5,  "nepochs" =>nepochs, "mb_size" => 50,"wd" =>  2e-2, "enc_nb_hl" =>ae_nb_hls, "enc_hl_size" => 128, "dim_redux"=> 1, 
"nb_clinf" => 5,"cph_lr" => 1e-4, "cph_nb_hl" => 2, "cph_hl_size" => 32)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
#validate_cphdnn_clinf!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)
c_ind = validate_cphdnn_clinf!(brca_cphdnnclinf_params, brca_prediction_NOISE, dummy_dump_cb, clinf)
push!(c_inds, c_ind)
println(c_inds)
### Update benchmark figure
i = 2
fig = Figure(resolution = (1024, 512));
ax = Axis(fig[1,1];xticks=(log10.(noise_sizes[1:i]), ["$x" for x in noise_sizes[1:i]]), xlabel = "Nb of added noisy features (log10 scale)",ylabel = "C-index (test)", title = "Performance of CPHDNN on BRCA clinical features by number of extra noisy dimension")
log10.(noise_sizes[1:i])
c_inds
test_points = zeros(2)
test_points[1:i] .= c_inds 
scatter!(fig[1,1], log10.(noise_sizes[1:i]), test_points, color = "blue", label = "test")
lines!(fig[1,1], log10.(noise_sizes[1:i]), test_points, color = "blue", linestyle = "--")
Label(fig[2,1], "𝗣𝗮𝗿𝗮𝗺𝗲𝘁𝗲𝗿𝘀 $(stringify(brca_cphdnnclinf_params))")
axislegend(ax, position = :rb)
CairoMakie.save("$outpath/CPHDNNCLINF_BRCA_BY_NB_DIM.png", fig)

##### ST-CPHDNN on clinical + gene expressions
brca_cphdnn_params = Dict("modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])", "dataset" => "brca_prediction", 
"model_type" => "enccphdnn", "session_id" => session_id, "nsamples_train" => length(brca_prediction.samples) - Int(round(length(brca_prediction.samples) / nfolds)), "nsamples_test" => Int(round(length(brca_prediction.samples) / nfolds)),
"nsamples" => length(brca_prediction.samples) , "insize" => length(brca_prediction.genes), "ngenes" => length(brca_prediction.genes), 
 "nfolds" => 5,  "nepochs" => 10_000, "mb_size" => 50,"wd" => 1e-3, "enc_nb_hl" =>ae_nb_hls, "enc_hl_size" => 128, "dim_redux"=> 1, 
"nb_clinf" => 5,"cph_lr" => 1e-3, "cph_nb_hl" => 2, "cph_hl_size" => 128)
dump_cb_brca = dump_model_cb(1000, labs_appdf(brca_prediction.stage), export_type = "pdf")
#validate_cphdnn_clinf!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)
validate_enccphdnn!(brca_cphdnn_params, brca_prediction, dump_cb_brca, clinf)

model = build(brca_cphdnn_params)

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

train_y_t = gpu(Matrix(fold["Y_t_train"][ordering,:]'));
train_y_e = gpu(Matrix(fold["Y_e_train"][ordering,:]'));
NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0
train_x
train_x_c
folds[1]["train_x_c"]
model.cph.encoder(train_x)
model.cph.lossf(model.cph, train_x, train_x_c, train_y_e, NE_frac_tr, brca_mtcphae_params["wd"])
model.encoder(train_x)
model.lossf(model, train_x, train_x_c, train_y_e, NE_frac_tr, brca_cphdnn_params["wd"])
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

