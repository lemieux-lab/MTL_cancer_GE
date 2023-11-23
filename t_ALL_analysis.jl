using Pkg 
using JSON
using CSV 
using DataFrames
using ProgressBars
Pkg.activate(".")
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

struct ALL_data
    data::Matrix
    samples::Array 
    genes::Array
end 

function merge_GDC_data(basepath)
    files = readdir(basepath)
    sample_data = CSV.read("$basepath/$(files[1])", DataFrame, delim = "\t", header = 2)
    sample_data = sample_data[5:end, ["gene_name", "tpm_unstranded"]]
    genes = sample_data.gene_name
    nsamples = length(files)
    ngenes = size(sample_data)[1]
    m=Array{Float32, 2}(undef, (nsamples, ngenes))

    for fid::Int in ProgressBar(1:length(files))
        file = files[fid]
        dat = CSV.read("$basepath/$(file)", DataFrame, delim = "\t", header = 2)
        dat = dat[5:end, ["gene_name", "tpm_unstranded"]]
        m[fid, :] = dat.tpm_unstranded
    end
    return m, files, genes
end 
m, samples, genes = merge_GDC_data("Data/GDC_raw/TARGET_ALL")
samples
genes

FILES = "$basepath/TARGET_ALL_clinical.json"
J = JSON.parsefile(FILES)
J[1]["case_id"]
J[1]["diagnoses"][1]
upd_dtime =  J[1]["demographic"]["updated_datetime"]
using Dates

a = DateTime(split(upd_dtime, "T")[1], "yyyy-mm-dd")
b = DateTime(split(J[1]["demographic"]["created_datetime"], "T")[1], "yyyy-mm-dd")
case_ids =
for F in J
    case_id = F["case_id"]
    # surv_t 
    # surv_e 
    # primary diagnosis 
    subtype = F["diagnoses"][1]["primary_diagnosis"] 
    ethnicity = F["demographic"]["race"]
    crea_dtime = F["demographic"]["created_datetime"]
    upd_dtime =  F["demographic"]["updated_datetime"]
    elapsed_time = Day(DateTime(split(upd_dtime, "T")[1], "yyyy-mm-dd") - DateTime(split(crea_dtime, "T")[1], "yyyy-mm-dd")).value
    surv_status = F["demographic"]["vital_status"] == "Dead" ? 1 : 0
    surv_t = surv_status == 1 ? F["demographic"]["days_to_death"] : elapsed_time
    println("$(case_id) $ethnicity $surv_t $surv_status")
end 