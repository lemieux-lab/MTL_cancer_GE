function get_GDC_CLIN_data_init_paths()

    # loading data
    CLIN_FULL = CSV.read("Data/GDC_clinical_raw.tsv", DataFrame)
    # MANIFEST = CSV.read("data/gdc_manifest_GE_2023-02-02.txt", DataFrame)
    # IDS = CSV.read("data/sample.tsv", DataFrame)
    baseurl = "https://api.gdc.cancer.gov/data"
    basepath = "Data/DATA/GDC_processed"
    FILES = "$basepath/GDC_files.json"
    J = JSON.parsefile(FILES)
    features = ["case_id", "case_submitter_id", "project_id", "gender", "age_at_index","age_at_diagnosis", "days_to_death", "days_to_last_follow_up", "primary_diagnosis", "treatment_type"]
    CLIN = CLIN_FULL[:, features]
    return J, CLIN_FULL, CLIN, baseurl, basepath
end 

function generate_fetch_data_file(J, baseurl, basepath)
    outputfile = "$basepath/fetch_data.sh"
    f = open(outputfile, "w")
    ## FECTHING DATA 
    for i::Int in ProgressBar(1:length(J))
        file_id = J[i]["file_id"]
        # println(file_id)
        case_id = J[i]["cases"][1]["case_id"]
        # println(case_id)
        cmd = "curl $baseurl/$file_id -o $basepath/GDC/$case_id\n"
        write(f, cmd)
        # cmd = `curl $baseurl/$file_id -o $basepath/$case_id`
        #run(cmd)
    end 
    close(f)
end 

struct GDC_data
    data::Matrix
    rows::Array 
    cols::Array
    targets::Array
end 

function load_GDC_data(infile; log_transform = false, shuffled= true)
    inf = h5open(infile, "r")
    tpm_data = inf["data"][:,:]
    case_ids = inf["rows"][:]
    gene_names = inf["cols"][:] 
    if in("labels", keys(inf))
        labels = inf["labels"][:]
    else 
        labels = zeros(length(case_ids))
    end 

    close(inf)
    if log_transform
        tpm_data = log10.(tpm_data .+1 )
    end 
    
    ids = collect(1:length(case_ids))
    shuffled_ids = shuffle(ids)
    if shuffled
        ids = shuffled_ids
    end 
    return tpm_data[ids,:], case_ids[ids], gene_names, labels[ids]
end   

function GDC_data(inputfile::String; log_transform=false, shuffled = true)
    tpm, cases, gnames, labels = load_GDC_data(inputfile;log_transform=log_transform, shuffled = shuffled)
    return GDC_data(tpm, cases, gnames, labels)
end
struct GDC_data_surv 
    data::Matrix
    rows::Array 
    cols::Array
    subgroups::Array
    survt::Array
    surve::Array
end 
function  GDC_data_surv(inf::String;log_transf = false)
    f = h5open(inf, "r")
    TPM_data = f["data"][:,:]
    if log_transf
        TPM_data = log10.(TPM_data .+ 1)
    end 

    case_ids = f["rows"][:]
    gene_names = f["cols"][:]
    survt = f["survt"][:]
    surve = f["surve"][:]
    subgroups = f["subgroups"][:]
    close(f)
    return GDC_data_surv(TPM_data, case_ids, gene_names, subgroups, survt, surve)
end

function write_h5(dat::GDC_data_surv, outfile)
    # HDF5
    # writing to hdf5 
    f = h5open(outfile, "w")
    f["data"] = dat.data
    f["rows"] = dat.rows
    f["cols"] = dat.cols
    f["subgroups"] = dat.subgroups
    f["survt"] = dat.survt
    f["surve"] = dat.surve    
    close(f)
end 

function write_h5(dat::GDC_data, labels, outfile)
    # HDF5
    # writing to hdf5 
    f = h5open(outfile, "w")
    f["data"] = dat.data
    f["rows"] = dat.rows
    f["cols"] = dat.cols
    f["labels"] = labels 
    close(f)
end 
function label_binarizer(labels)
    nb = length(labels)
    levels = unique(labels)
    numerised = [findall(levels .== x)[1] for x in labels]
    mat = reshape(Array{Int}(zeros(length(levels)*nb)), (nb, length(levels)))
    [mat[i,numerised[i]] = 1 for i in 1:length(numerised)]
    return mat
end 
function merge_GDC_data(basepath, outfile)
    files = readdir(basepath)
    sample_data = CSV.read("$basepath/$(files[1])", DataFrame, delim = "\t", header = 2)
    sample_data = sample_data[5:end, ["gene_name", "tpm_unstranded"]]
    nsamples = length(files)
    ngenes = size(sample_data)[1]
    m=Array{Float32, 2}(undef, (nsamples, ngenes))

    for fid::Int in ProgressBar(1:length(files))
        file = files[fid]
        dat = CSV.read("$basepath/$(file)", DataFrame, delim = "\t", header = 2)
        dat = dat[5:end, ["gene_name", "tpm_unstranded"]]
        m[fid, :] = dat.tpm_unstranded
    end
    output_data = GDC_data(m, files, Array{String}(sample_data.gene_name)) 
    write_h5(output_data, outfile)
    return output_data 
end 

function tcga_abbrv()
    abbrv = CSV.read("Data/GDC_processed/TCGA_abbrev.txt", DataFrame, delim = ",")
    abbrvDict = Dict([("TCGA-$(String(strip(abbrv[i,1])))", abbrv[i,2]) for i in 1:size(abbrv)[1]])
    return abbrvDict
end
function tcga_abbrv(targets::Array)
    abbrv = CSV.read("Data/GDC_processed/TCGA_abbrev.txt", DataFrame, delim = ",")
    abbrvDict = Dict([("TCGA-$(String(strip(abbrv[i,1])))", abbrv[i,2]) for i in 1:size(abbrv)[1]])
    return [abbrvDict[l] for l in targets]
end

function preprocess_data(GDCd, CLIN, outfilename; var_frac = 0.75)
    cases = GDCd.rows
    ngenes = length(GDCd.cols)
    # intersect with clin data 
    uniq_case_id = unique(CLIN.case_id)
    keep = [in(c,uniq_case_id ) for c in cases]
    GDC = GDC_data(GDCd.data[keep,:], GDCd.rows[keep], GDCd.cols)

    # map to tissues 
    cid_pid = Dict([(cid, pid) for (cid, pid) in zip(CLIN.case_id, Array{String}(CLIN.project_id))])
    tissues = [cid_pid[c] for c in GDC.rows]
    # filter on variance
    vars = vec(var(GDC.data, dims = 1))  
    hv = vec(var(GDC.data, dims =1 )) .> sort(vars)[Int(round(var_frac * ngenes))]

    GDC_hv = GDC_data(GDC.data[:,hv], GDC.rows, GDC.cols[hv])

    f = h5open(outfilename, "w")
    f["data"] = GDC_hv.data
    f["rows"] = GDC_hv.rows
    f["cols"] = GDC_hv.cols
    f["tissues"] = tissues
    close(f)
    return GDC_hv
end 

function run_tsne_on_GDC(GDC_data, tissues)
    @time TCGA_tsne = tsne(GDC_data, 2, 50, 1000, 30.0;verbose=true,progress=true)

    TSNE_df = DataFrame(Dict("dim_1" => TCGA_tsne[:,1], "dim_2" => TCGA_tsne[:,2], "tissue" => tissues))

    q = AlgebraOfGraphics.data(TSNE_df) * mapping(:dim_1, :dim_2, color = :tissue, marker = :tissue) * visual(markersize = 15,strokewidth = 0.5, strokecolor =:black)

    main_fig = draw(q ; axis=(width=1024, height=1024,
                    title = "2D TSNE by tissue type on GDC data, number of input genes: $(size(tcga_hv.data)[2]), nb. samples: $(size(tcga_hv.data)[1])",
                    xlabel = "TSNE 1",
                    ylabel = "TSNE 2"))


    save("RES/GDC_$(nsamples)_samples.svg", main_fig, pt_per_unit = 2)

    save("RES/GDC_$(nsamples)_samples.png", main_fig, pt_per_unit = 2)

end 

function list_ages(case_ids, ages)
    out = Dict()
    for (case_id,age) in zip(case_ids,ages)
        if age != "'--"
            dat = parse(Int, age)
        else 
            dat = 60
        end 
        out[case_id] = dat
    end 
    return out
end 
function list_stages(case_ids, stages)
    stage_i = Dict()
    stage_ii = Dict()
    stage_iii = Dict()
    stage_iv = Dict()
    for (case_id, stage) in zip(case_ids, stages)
        if stage in ["Stage I", "Stage IA", "Stage IB", "Stage IC"]
            stage_i[case_id] = 1
            stage_ii[case_id] = 0
            stage_iii[case_id] = 0
            stage_iv[case_id] = 0
        elseif stage in ["Stage II", "Stage IIA", "Stage IIB", "Stage IIC"]
            stage_i[case_id] = 0
            stage_ii[case_id] = 1
            stage_iii[case_id] = 0
            stage_iv[case_id] = 0
        elseif stage in ["Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC"]
            stage_i[case_id] = 0
            stage_ii[case_id] = 0
            stage_iii[case_id] = 1
            stage_iv[case_id] = 0
        elseif stage == "Stage IV"
            stage_i[case_id] = 0
            stage_ii[case_id] = 0
            stage_iii[case_id] = 0
            stage_iv[case_id] = 1   
        else 
            stage_i[case_id] = 0
            stage_ii[case_id] = 0
            stage_iii[case_id] = 0
            stage_iv[case_id] = 0   
        end        
    end
    return stage_i, stage_ii, stage_iii, stage_iv        
end 


function assemble_clinf(brca_prediction)
    clinf = DataFrame(["samples"=> brca_prediction.samples, 
        "age"=>log2.(brca_prediction.age),
        "stage_i"=> Array{Int}([x in ["Stage I", "Stage IA", "Stage IB", "Stage IC"] for x in brca_prediction.stage]),
        "stage_ii"=>  Array{Int}([x in ["Stage II", "Stage IIA", "Stage IIB", "Stage IIC"] for x in brca_prediction.stage]),
        "stage_iii"=>  Array{Int}([x in ["Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC"] for x in brca_prediction.stage]),
        "stage_iv"=>  Array{Int}([x in ["Stage IV"] for x in brca_prediction.stage]) ])
    return clinf
end
struct BRCA_data
    data::Matrix # gene expression data
    samples::Array # sample ids (case_ids)
    genes::Array # gene names 
    survt::Array # survival times
    surve::Array # censorship
    age::Array # patient age 
    stage::Array # cancer stage 
    ethnicity::Array # patient ethnicity 
end

struct LGN_data
    data::Matrix # gene expression data
    samples::Array # sample ids (case_ids)
    genes::Array # gene names 
    #survt::Array # survival times
    #surve::Array # censorship
    #age::Array # patient age 
    cyto_group::Array
    #stage::Array # cancer stage 
    #ethnicity::Array # patient ethnicity 
end


function write_h5(dat::BRCA_data, outfile)
    # HDF5
    # writing to hdf5 
    f = h5open(outfile, "w")
    f["data"] = dat.data
    f["samples"] = dat.samples
    f["genes"] = dat.genes
    f["survt"] = dat.survt 
    f["surve"] = dat.surve
    f["age"] = dat.age
    f["stage"] = dat.stage
    f["ethnicity"] = dat.ethnicity
    
    close(f)
end 
function minmaxnorm(data, genes)
    # remove unexpressed
    genes = genes[vec(sum(data, dims = 1) .!= 0)]
    data = data[:, vec(sum(data, dims = 1) .!= 0)]
    # normalize
    vmax = maximum(data, dims = 1)
    vmin = minimum(data, dims = 1)
    newdata = (data .- vmin) ./ (vmax .- vmin)
    genes = genes[vec(var(newdata, dims = 1) .> 0.02)]
    newdata = newdata[:, vec(var(newdata, dims = 1) .> 0.02)]
    return newdata, genes 
end 

function BRCA_data(infile::String; minmax_norm = false, remove_unexpressed=false)
    inf = h5open(infile, "r")
    data, samples, genes, survt, surve,age, stage, ethnicity = inf["data"][:,:], inf["samples"][:], inf["genes"][:], inf["survt"][:], inf["surve"][:], inf["age"][:], inf["stage"][:], inf["ethnicity"][:]
    if remove_unexpressed
        expressed = vec(sum(data, dims = 1) .!= 0)
        data = data[:,findall(expressed)]
        genes = genes[findall(expressed)]
    end 
    if minmax_norm
        data, genes = minmaxnorm(data, genes)
    end 
    brca_prediction = BRCA_data(data, samples, genes, survt, surve,age, stage, ethnicity)
    close(inf)
    return brca_prediction 
end 

function compute_t_e_on_clinical_data(CLIN_FULL, brca_submitter_ids)
    features = ["case_id", "case_submitter_id", "project_id", "gender", "age_at_index","age_at_diagnosis","ajcc_pathologic_stage", "ann_arbor_pathologic_stage", "days_to_death", "days_to_last_follow_up","ethnicity","primary_diagnosis", "treatment_type"]
    BRCA_CLIN = CLIN_FULL[findall([x in brca_submitter_ids for x in CLIN_FULL[:,"case_submitter_id"]]),features]
    BRCA_CLIN = BRCA_CLIN[findall(nonunique(BRCA_CLIN[:,1:end-1])),:]
    BRCA_CLIN[findall(BRCA_CLIN[:,"days_to_death"] .== "'--"), "days_to_death"] .= "NA"
    BRCA_CLIN[findall(BRCA_CLIN[:,"days_to_last_follow_up"] .== "'--"), "days_to_last_follow_up"] .= "NA"
    BRCA_CLIN = BRCA_CLIN[findall(BRCA_CLIN[:,"days_to_death"] .!= "NA" .|| BRCA_CLIN[:,"days_to_last_follow_up"] .!= "NA"),:]
    BRCA_CLIN[:, "age_years"] .= [parse(Int, x) for x in BRCA_CLIN[:,"age_at_index"]]
    # BRCA_CLIN[:, "age_days"] .= [parse(Int, x) for x in BRCA_CLIN[:,"age_at_diagnosis"]]
    
    typeof(BRCA_CLIN[:, "age_at_index"])
    survt = Array{String}(BRCA_CLIN[:,"days_to_death"])
    surve = ones(length(survt))
    surve[survt .== "NA"] .= 0
    survt[survt .== "NA"] .= BRCA_CLIN[survt .== "NA","days_to_last_follow_up"]
    survt = [Int(parse(Float32, x)) for x in survt]
    BRCA_CLIN[:,"survt"] .= survt
    BRCA_CLIN[:,"surve"] .= surve    
    keep = BRCA_CLIN[:,"survt"] .> 0 # keep only positive valued survt patients 
    features = ["case_id", "case_submitter_id", "project_id", "gender", "age_years", "ajcc_pathologic_stage", "ann_arbor_pathologic_stage", "days_to_death", "days_to_last_follow_up","ethnicity","primary_diagnosis", "treatment_type", "survt", "surve"]
    
    return BRCA_CLIN[keep,features]
end
function assemble_BRCA_data(CLIN_FULL, brca_fpkm_df)
    ids = names(brca_fpkm_df)
    brca_submitter_ids = [join(split(x,"-")[1:3],"-") for x in ids[2:end]]
    keep = [split(x,"-")[4] == "01A" for x in ids[2:end]] # keep only the 01A tagged samples
    brca_submitter_ids = brca_submitter_ids[findall(keep)]
    gene_names = brca_fpkm_df[:,1]
    brca_fpkm_df = brca_fpkm_df[:,findall(keep) .+ 1]
    BRCA_CF = compute_t_e_on_clinical_data(CLIN_FULL, brca_submitter_ids)
    sample_ids = intersect(BRCA_CF[:,"case_submitter_id"], brca_submitter_ids)
    # select in fpkm matrix 
    brca_fpkm_df = brca_fpkm_df[:,findall([x in BRCA_CF[:,"case_submitter_id"] for x in brca_submitter_ids])]
    data = Matrix(brca_fpkm_df[:,sort(names(brca_fpkm_df))])
    # select in clinical file matrix
    sample_ids = names(brca_fpkm_df)
    samples = sort(sample_ids)
    survt = BRCA_CF[sortperm(sample_ids),"survt"]
    surve = BRCA_CF[sortperm(sample_ids),"surve"]
    age = BRCA_CF[sortperm(sample_ids),"age_years"]
    ethnicity = Array{String}(BRCA_CF[sortperm(sample_ids),"ethnicity"])
    stage = Array{String}(BRCA_CF[sortperm(sample_ids),"ajcc_pathologic_stage"])
    brca_prediction = BRCA_data(Matrix(data'), samples, gene_names,  survt, surve, age, stage, ethnicity)
    infile = "Data/GDC_processed/TCGA_BRCA_surv_cf_fpkm.h5"
    write_h5(brca_prediction, infile)

    return brca_prediction, infile
end 