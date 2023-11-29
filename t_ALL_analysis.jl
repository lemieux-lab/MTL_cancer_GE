using Pkg 
using JSON
using CSV 
using DataFrames
using ProgressBars
using Dates
using XLSX

Pkg.activate(".")
include("init.jl")
include("data_processing.jl")

baseurl = "https://api.gdc.cancer.gov/data"
basepath = "Data/GDC_raw"
outputfile = "$basepath/TARGET_ALL_fetch_data.sh"
FILES = "$basepath/TARGET_ALL_files.json"
J = JSON.parsefile(FILES)
f = open(outputfile, "w")
for file_id::Int in ProgressBar(1:length(J))
    case_id = J[file_id]["cases"][1]["case_id"]
    filename = J[file_id]["file_name"]
    MANIFEST = CSV.read("$basepath/TARGET_ALL_manifest.txt", DataFrame)
    UUID = MANIFEST[MANIFEST[:,"filename"] .== filename,"id"][1]
    cmd = "curl --remote-header-name $baseurl/$UUID -o $basepath/TARGET_ALL/$case_id\n"
    write(f, cmd)
end
close(f)
FILES_id = DataFrame(:UUID=>UUIDs, :case_id => case_ids, :fname=>filenames)
CSV.write("")


struct ALL_data
    data::Matrix
    samples::Array 
    genes::Array
end 

function merge_GDC_data(basepath)
    files = readdir(basepath)
    sample_data = CSV.read("$basepath/$(files[1])", DataFrame, delim = "\t", header = 2)
    sample_data = sample_data[5:end, ["gene_name", "stranded_second"]]
    genes = sample_data.gene_name
    nsamples = length(files)
    ngenes = size(sample_data)[1]
    m=Array{Float32, 2}(undef, (nsamples, ngenes))

    for fid::Int in ProgressBar(1:length(files))
        file = files[fid]
        dat = CSV.read("$basepath/$(file)", DataFrame, delim = "\t", header = 2)
        dat = dat[5:end, ["gene_name", "stranded_second"]]
        m[fid, :] = dat.stranded_second
    end
    return m, files, genes
end 
m, samples, genes = merge_GDC_data("Data/GDC_raw/TARGET_ALL")


FILES = "$basepath/TARGET_ALL_clinical.json"
J = JSON.parsefile(FILES)
submitids, case_ids = [], []
for F in J
    case_id = F["case_id"]
    # surv_t 
    # surv_e 
    # primary diagnosis 
    subtype = F["diagnoses"][1]["primary_diagnosis"] 
    ethnicity = F["demographic"]["race"]
    crea_dtime = F["demographic"]["created_datetime"]
    upd_dtime =  F["demographic"]["updated_datetime"]
    submitter_ID = split(F["demographic"]["submitter_id"],"-")[3][1:6]
    push!(case_ids, case_id)
    push!(submitids, submitter_ID)
    elapsed_time = Day(DateTime(split(upd_dtime, "T")[1], "yyyy-mm-dd") - DateTime(split(crea_dtime, "T")[1], "yyyy-mm-dd")).value
    surv_status = F["demographic"]["vital_status"] == "Dead" ? 1 : 0
    surv_t = surv_status == 1 ? F["demographic"]["days_to_death"] : elapsed_time
    println("$(case_id) $ethnicity $submitter_ID $surv_t $surv_status")
end 
CLIN_df = DataFrame(:USI=>submitids, :case_id=>case_ids)

fpath = "/u/sauves/MTL_cancer_GE/Data/GDC_raw/TARGET_phase2_SampleID_tSNE-perplexity20"
#fpath = "/u/sauves/MTL_cancer_GE/Data/GDC_raw/TARGET_Phase2_T_ALL_Mullighan"
#ALL_subtypes = XLSX.readxlsx("$fpath.xlsx")["Sample ID"][:]
ALL_subtypes = XLSX.readxlsx("$fpath.xlsx")["Sheet1"][:]

ALL_df = DataFrame(:X1=>ALL_subtypes[2:end,2],:X2=>ALL_subtypes[2:end,3],:USI =>  ALL_subtypes[2:end,6], :sampleID =>   ALL_subtypes[2:end,7], 
    :subtype =>   ALL_subtypes[2:end,8], :ETP_classification => ALL_subtypes[2:end,4])
#ALL_df = DataFrame(:USI => ALL_subtypes[3:end, 5], :subtype =>   ALL_subtypes[3:end,6])
CSV.write("$fpath.csv", ALL_df)
p = data(ALL_df) * mapping(:X1,:X2,color=:subtype, marker = :subtype)
draw(p)

# pruning samples
# interesect

# join ALL_df to USI_rnaseq_filename
samples_df = DataFrame(:case_id=>samples, :II=>collect(1:length(samples)))
samples_df = sort(innerjoin(samples_df, CLIN_df, on = :case_id), :II)
# mm = m[innerjoin(samples_df, CLIN_df, on = :case_id)[:,"II"],:]
FULL_CLIN_DF = innerjoin(ALL_df, samples_df, on = :USI)
labels = FULL_CLIN_DF.subtype # already sorted

# pruning genes OPTIONAL
vars = vec(var(tpm_data, dims = 1))
new_tpm_data, new_genes = minmaxnorm(log10.(tpm_data .+ 1), genes)

outfile = h5open("Data/GDC_processed/TARGET_ALL_264_norm_tpm_lab.h5", "w")
outfile["data"] = log10.(m[FULL_CLIN_DF.II,:] .+ 1) 
outfile["samples"] = Array{String}(samples[FULL_CLIN_DF.II]) 
outfile["labels"] = Array{String}(labels) 
outfile["genes"] = Array{String}(genes) 
close(outfile)

using TSne
size(tpm_data)
ALL_tsne = tsne(log10.(tpm_data .+ 1) ,2, 0, 1000, 20; pca_init = false, verbose = true, progress = true)
X = ALL_tsne[:,1]
Y = ALL_tsne[:,2]
TSNE_df = DataFrame(:TSNE1=>X, :TSNE2=>Y, :group=>labels)
g = data(TSNE_df) * mapping(:TSNE1,:TSNE2,color=:group, marker = :group)
draw(g)


TPM_data_Pat = CSV.read("Data/GDC_raw/target-phase2.reverse.stranded.tsv", DataFrame)


#TPM_data = parse(Float32, TPM_data_Pat[6:end,2:end])
II = size(TPM_data_Pat)[1] - 5
JJ =  size(TPM_data_Pat)[2] - 1
TPM_data = zeros(JJ,II)
sample_names = [split(x,"-")[3] for x in names(TPM_data_Pat)[2:end]]

[TPM_data[j,i] = parse(Float32, TPM_data_Pat[i + 5 ,j + 1])  for i in 1:II, j in 1:JJ]
TPM_data

ALL_tsne = tsne(log10.(m[FULL_CLIN_DF.II,:] .+ 1),2, 0, 1000, 20; pca_init = false, verbose = true, progress = true)
X = ALL_tsne[:,1]
Y = ALL_tsne[:,2]
TSNE_df = DataFrame(:TSNE1=>X, :TSNE2=>Y, :group=>FULL_CLIN_DF.subtype)
h = data(TSNE_df) * mapping(:TSNE1,:TSNE2,color=:group, marker = :group)
draw(h)
