using DataFrames
using Statistics
# using RedefStructs
using CSV
# using CUDA
# using Flux 
# using RedefStructs
# using Random


struct DataFE
    name::String
    data::Array
    factor_1::Array
    factor_2::Array
end

Base.getindex(d::DataFE, i::Int) = d.data[i,:]
function Base.getindex(d::DataFE, ids::Vector)
    return DataFE(d.name, d.data[ids,:], d.factor_1[ids], d.factor_2)
end 

struct FoldDataFE
    train::DataFE
    train_ids::Array
    test::DataFE
    test_ids::Array
end

struct FoldDataGen
    train::Array
    train_ids::Array
    test::Array
    test_ids::Array
end

function split_train_test(data::DataFrame; nfolds::Int=10)
    folds = Array{FoldDataGen, 1}(undef, nfolds)
    nfeatures = size(data)[2]
    nsamples = size(data)[1]
    fold_size = Int(floor(nsamples / nfolds))
    ids = collect(1:nsamples) # or data.factor_1
    shuffled_ids = shuffle(ids)
    for i in 1:nfolds 
        tst_ids = shuffled_ids[(i - 1) * fold_size + 1: min(nsamples, i * fold_size )]
        tr_ids = setdiff(ids,tst_ids)

        train = Matrix(data[tr_ids,:])
        test = Matrix(data[tst_ids,:])

        folds[i] = FoldDataGen(train, tr_ids, test, tst_ids)
    end 
    return folds, fold_size
end
function split_train_test(data::DataFE; nfolds::Int=10)
    nfolds = 10
    folds = Array{FoldDataFE, 1}(undef, nfolds)
    nsamples = length(data.factor_1)
    fold_size = Int(floor(nsamples / nfolds))
    ids = collect(1:nsamples) # or data.factor_1
    shuffled_ids = shuffle(ids)
    for i in 1:nfolds 
        tst_ids = shuffled_ids[(i - 1) * fold_size + 1: min(nsamples, i * fold_size )]
        tr_ids = setdiff(ids,tst_ids)

        train = DataFE("train", data.data[tr_ids,:], data.factor_1[tr_ids], data.factor_2)
        test = DataFE("test", data.data[tst_ids,:], data.factor_1[tst_ids], data.factor_2)

        folds[i] = FoldDataFE(train, tr_ids, test, tst_ids)
    end 
    return folds 
end


function split_train_test_interest_groups(data, cf_df::DataFrame; n::Int=10)
    ## extract n samples from training set.
    inv16 = findall(x-> x == "inv_16", cf_df.interest_groups)
    t8_21 = findall(x-> x == "t8_21", cf_df.interest_groups)
    mll_t = findall(x-> x == "MLL_t", cf_df.interest_groups)
    interest = shuffle(vcat([inv16 , t8_21, mll_t]...))
    tst_ids = interest[1:(min(n, length(interest)))]
    tr_ids = setdiff(collect(1:size(data.data)[1]), tst_ids)

    train = DataFE("train", data.data[tr_ids,:], data.factor_1[tr_ids], data.factor_2)
    test = DataFE("test", data.data[tst_ids,:], data.factor_1[tst_ids], data.factor_2)

    fd = FoldDataFE(train, tr_ids, test, tst_ids)
    return fd
end

function params_list_to_df(pl)
    df = DataFrame(Dict([
    ("modelid", [p.modelid for p in pl]), 
    ("emb_size_1", [p.emb_size_1 for p in pl]),
    ("emb_size_2", [p.emb_size_2 for p in pl]),
    ("tr", [p.tr for p in pl]),
    ("wd", [p.wd for p in pl]),
    ("hl1_size", [p.hl1_size for p in pl]),
    ("hl2_size", [p.hl2_size for p in pl]),
    ("nepochs", [p.nepochs for p in pl]),
    ("insize", [p.insize for p in pl]),
    ("nsamples", [p.nsamples for p in pl]),
    ("set", [p.set for p in pl]),
    ("gr_clip", [p.clip for p in pl])
    ]))
    return df
end
function params_list_to_df_3_factors(pl)
    df = DataFrame(Dict([
    ("modelid", [p.modelid for p in pl]), 
    ("emb_size_1", [p.emb_size_1 for p in pl]),
    ("emb_size_2", [p.emb_size_2 for p in pl]),
    ("emb_size_3", [p.emb_size_3 for p in pl]),
    ("tr", [p.tr for p in pl]),
    ("wd", [p.wd for p in pl]),
    ("hl1_size", [p.hl1_size for p in pl]),
    ("hl2_size", [p.hl2_size for p in pl]),
    ("nepochs", [p.nepochs for p in pl]),
    ("insize", [p.insize for p in pl]),
    ("nsamples", [p.nsamples for p in pl]),
    ("set", [p.set for p in pl]),
    ("gr_clip", [p.clip for p in pl])
    ]))
    return df
end


function get_interest_groups(target)
    if  occursin( "inv(16)", target)
        return "inv(16)"
    
    elseif occursin("t(8;21)", target)
        return "t(8;21)"
    
    elseif occursin("t(9;11)", target)
        return "t(9;11)"

    elseif occursin("t(15;17)", target)
        return "t(15;17)"

    elseif occursin("without maturation", target)
        return "without_maturation"
    
    elseif occursin("with maturation", target)
        return "with_maturation"
    
    else return "other"

    end
end

function load_data(basepath::String; frac_genes=0.5, avg_norm = false)
    clinical_fname = "$(basepath)/Data/LEUCEGENE/lgn_pronostic_CF"
    ge_cds_fname = "$(basepath)/Data/LEUCEGENE/lgn_pronostic_GE_CDS_TPM.csv"
    ge_lsc17_fname = "$(basepath)/Data/SIGNATURES/LSC17_lgn_pronostic_expressions.csv"

    cf = CSV.read(clinical_fname, DataFrame)
    interest_groups = [get_interest_groups(g) for g  in cf[:, "WHO classification"]]
    cf.interest_groups = interest_groups
    ge_cds_raw_data = CSV.read(ge_cds_fname, DataFrame)
    lsc17 = CSV.read(ge_lsc17_fname, DataFrame)
    ge_cds_all = log_transf_high_variance(ge_cds_raw_data, frac_genes=frac_genes, avg_norm = avg_norm)
    # print(ge_cds_all)
    # ge_cds_split = split_train_test(ge_cds_all)
    return (cf, ge_cds_all, lsc17)
end
struct MetaData 
    data::Matrix{String}
    cols::Array{String}
    rows::Array{String}
end 


function Base.getindex(factors::MetaData, col::String)
    return factors.data[:, factors.cols .== col]
end
function Base.getindex(factors::MetaData, cols::Array{String})
    return factors.data[:,findall(map(x -> in(x, cols), md.cols))]
end
function MetaData(fpath)
    cf = CSV.read(fpath, DataFrame)
    columns = cf[:,1]
    cf = cf[:,2:end-2]
    sample_ids = names(cf)
    m = Array{String, 2}(undef, reverse(size(cf)))
    for i in 1:size(cf)[1]
        println(i)
        correct = findall(skipmissing(Array(cf[i,:]) .== Array(cf[i,:])))
        miss_vals = setdiff(collect(1:size(cf)[2]), correct)
        m[correct,i] = Array(cf[i,correct])
        m[miss_vals,i] = ["nan" for _ in 1:length(miss_vals)]
    end 
    
    md = MetaData(m, columns, sample_ids)
    return md
end 
struct GenesTPM
    data::Matrix{Float64}
    rows::Array{String}
    cols::Array{String}
    # add sample_ids attribute 
    gene_names::Array{String}
    transcript_ids::Array{String}
    gene_category::Array{String}
    gene_loc::Array{String}
    gene_ID::Array{String}
end
function GenesTPM(filename)
    genes_tpm = CSV.read(filename, DataFrame)
    data = log10.(Matrix{Float64}(genes_tpm[:,3:end-4]) .+ 1)
    # remove unexpressed, keep 75% most variant
    vars = sort(var(data, dims = 2), dims = 1)
    high_v = findall(vec(vars .>= vars[Int(floor(0.75 * length(vars)))]))

    gene_names = Array{String}(genes_tpm[high_v, "Name"])
    gene_ids = Array{String}(genes_tpm[high_v,"gene_id"])
    transcr_ids = genes_tpm[high_v, "transcript_id(s)"]
    gene_cat = Array{String}(genes_tpm[high_v, "Category"])
    gene_loc = Array{String}(genes_tpm[high_v, "Location"])
    gene_ID = Array{String}(genes_tpm[high_v, "ID"])
    sample_ids = names(genes_tpm[high_v,3:end-4])
    data = data[high_v,:]

    GTPM = GenesTPM(data', sample_ids, gene_names, gene_names, transcr_ids, gene_cat, gene_loc, gene_ID)

    return GTPM 
end
function log_transf_high_variance(df::DataFrame; frac_genes = 0.1, avg_norm=false)
    index = df[:,1]
    data_full = df[:,2:end] 
    cols = names(data_full)
    # log transforming
    data_full = log10.(Array(data_full) .+ 1)
    # remove least varying genes
    ge_var = var(data_full, dims = 1)[1,:]
    ge_var_bp = quantile(ge_var, 1.0 - frac_genes)
    # high variance only 
    hvg = findall(ge_var .> ge_var_bp)
    data_full = data_full[:,hvg]
    if avg_norm 
        data_full = data_full .- mean(data_full, dims = 1)
    end
    cols = cols[hvg]
    new_data = DataFE("full", data_full, index, cols)
    return new_data
end