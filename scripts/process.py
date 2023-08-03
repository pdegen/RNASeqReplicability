import os
import glob
import logging
import numpy as np
import pandas as pd
import json
from itertools import combinations
from pathlib import Path
from misc import Timer, pickler, open_table
from numba import njit
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
import pickle
from copy import deepcopy
from itertools import combinations, product
from statsmodels.stats.multitest import multipletests

def signal_to_noise(counts_df):
    N = len(counts_df.columns)//2
    control_mean = counts_df.iloc[:,:N].mean(axis=1)
    case_mean = counts_df.iloc[:,N:].mean(axis=1)
    control_std = counts_df.iloc[:,:N].std(axis=1)
    case_std = counts_df.iloc[:,N:].std(axis=1)
    return (control_mean-case_mean)/(control_std+case_std)

def sklearn_metrics(true, pred, average="binary", zero_division=0, pos_label=1):
    prec, rec, _, _ = precision_recall_fscore_support(true, pred, average=average, zero_division=zero_division, pos_label=pos_label)
    mcc = matthews_corrcoef(true, pred)
    return mcc, prec, rec

@njit()
def get_array_metrics_numba(truth, boolarr):
    """
    boolarr: boolean np.array of shape (nrow = n_genes, ncols = n_cohorts), true if gene is DEG in cohort
    truth: ground truth boolean array of shape (n_genes, 1)
    returns lists of mcc, precision, recall values for each cohort
    """    
    mcc, prec, rec = [], [], []
    a = boolarr + 10*np.expand_dims(truth, -1)
    n = len(truth)
    
    for col in a.T:
        TP = np.count_nonzero(col == 11)
        FN = np.count_nonzero(col == 10)
        FP = np.count_nonzero(col == 1)
        TN = n-TP-FN-FP
        prec.append(TP/(TP+FP) if TP+FP else np.nan)
        rec.append(TP/(TP+FN) if TP+FN else np.nan)
        squared = float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        mcc.append((TP*TN-FP*FN)/(np.sqrt(squared)) if squared else np.nan)
        
    return mcc, prec, rec

@njit()
def get_replicability_numba(boolarr):
    """
    boolarr: boolean np.array of shape (nrow = n_genes, ncols = n_cohorts), true if gene is DEG in cohort
    returns list of jaccard indices of all pairwise cohorts
    """
    n = boolarr.shape[1] # n_cohorts
    #ncomb = (n**2-n)//2
    rep = []
    col_sum = np.sum(boolarr,axis=0)
    for i in range(n):
        for j in range(i+1,n):
            intersect = np.sum(boolarr[:,i] & boolarr[:,j])
            rep.append(intersect / max(0.1, col_sum[i]+col_sum[j]-intersect))
    return rep


def get_replicability(list_list, method="jaccard", top=-1, min_genes=0):
    """
    Given a list of list of elements (e.g. DEGs), calculate replicability of top elements of all pairwise lists
    For top_genes > 0, assumes that the DEGs are sorted by FDR (most signficant to least)
    top_genes = 'min:500' uses a cutoff of 500 only when both cohorts have at least 500 DEG, else use the smaller of the two sets as cutoff
    """
    
    if top < 0:
        return get_replicability_notop(list_list, method=method, min_genes=min_genes)
    
    rep = []
    use_min_top = str(top).startswith("min")
    
    for c1, c2 in combinations(list_list, 2):
        if use_min_top: top = min(len(c1),len(c2), int(top.split(":")[1]))
        c1, c2 = set(c1[:top]), set(c2[:top])
        if (len(c1) < min_genes) or (len(c2) < min_genes): continue
        intersect = len(c1 & c2)
        if method == "max":
            rep.append(intersect / max(0.1, len(c1), len(c2)))
        elif method == "min":
            rep.append(intersect / max(0.1, min(len(c1), len(c2))))
        elif method == "jaccard":
            rep.append(intersect / max(0.1, len(c1)+len(c2)-intersect))
        else:
            raise Exception("Not a valid method")
    return rep


def get_replicability_notop(list_list, method="jaccard", min_genes=0):
    """
    Given a list of list of elements (e.g. DEGs), calculate replicability of all pairwise lists
    """
    
    list_set = [set(l) for l in list_list]
        
    rep = []
    for c1, c2 in combinations(list_set, 2):
        if (len(c1) < min_genes) or (len(c2) < min_genes): continue
        intersect = len(c1 & c2)
        if method == "max":
            rep.append(intersect / max(0.1, len(c1), len(c2)))
        elif method == "min":
            rep.append(intersect / max(0.1, min(len(c1), len(c2))))
        elif method == "jaccard":
            rep.append(intersect / max(0.1, len(c1)+len(c2)-intersect))
        else:
            raise Exception("Not a valid method")
    return rep

def delete_redundant_slurmfiles(outpath, outname, all_N, gsea=False) -> None:
    """
    Remove old slurm files of jobs that failed and have been redone
    """
 
    slurm_folder = "gsea/slurm" if gsea else "slurm"
    tot_removed = 0
    for N in all_N:
        outpath_N = f"{outpath}/{outname}_N{N}"   
        cohorts = [f.name for f in os.scandir(outpath_N) if f.is_dir()]
        
        for cohort in cohorts:
            cohort_id = int(cohort.split("_")[-1])
            slurmpath = f"{outpath_N}/{cohort}/{slurm_folder}"
            slurmfiles = glob.glob(f"{slurmpath}/slurm-*.out")
            slurmfiles = sorted(slurmfiles, reverse=True)
            
            newest = []
            for sl in slurmfiles:
                suffix = sl.split(".")[1:-1]
                suffix = ".".join(suffix)
                if suffix in newest:
                    logging.info(f"Removing {cohort} {suffix}")
                    tot_removed += 1
                    os.system(f"rm {sl}")
                else:
                    newest.append(suffix)
            
        logging.info(f"N{N}: Removed {tot_removed} slurm files")
        
def delete_redundant_gsea_tmp_folders(outpath, outname, all_N) -> None:
    """
    Remove old tmp folders of jobs that failed and have been redone
    """

    tot_removed = 0
    for N in all_N:
        outpath_N = f"{outpath}/{outname}_N{N}"   
        cohorts = [f.name for f in os.scandir(outpath_N) if f.is_dir()]
        
        for cohort in cohorts:
            cohort_id = int(cohort.split("_")[-1])
            g=glob.glob(f"{outpath_N}/{cohort}/gsea/tmp*")
            if len(g) > 0:
                os.system(f"rm -r {outpath_N}/{cohort}/gsea/tmp*")
                tot_removed += len(g)
        
    logging.info(f"Removed {tot_removed} redundant tmp folders")

def get_init_samples_from_cohort(configfile):
    """From the config.json read the subsamples from before outlier removal"""
    with open(configfile, "r") as f:
        configdict = json.load(f)    
    return configdict["samples_i"] if "samples_i" in configdict else []

def update_results_dict(results, all_N, DEAs, outlier_methods, FDRs, logFCs, overwrite=False) -> dict:
    """
    Add missing keys to results dict or create it from scratch
    """
    
    inner = {fdr: {logfc: None for logfc in logFCs} for fdr in FDRs}
    outer = {"median_rep": deepcopy(inner), 
             "median_deg": deepcopy(inner), 
             "median_rep_adj": deepcopy(inner), 
             "median_deg_adj": deepcopy(inner), 
             "median_mcc": deepcopy(inner), 
             "median_prec": deepcopy(inner), 
             "median_rec": deepcopy(inner),
             "median_mcc_adj": deepcopy(inner), 
             "median_prec_adj": deepcopy(inner), 
             "median_rec_adj": deepcopy(inner),
             "gene_rep": deepcopy(inner)}
    

    if overwrite:
        logging.info("Creating results dict from scratch")
        results = {N: {out: {dea: deepcopy(outer) for dea in DEAs} | {"cohorts": {}} for out in outlier_methods} for N in all_N}
        results["n_patients_df"] = {out: None for out in outlier_methods if out != "none"}
        
    else:
        logging.info("Checking for missing keys")
        missing = []
        
        for N in all_N:
            if N not in results:
                missing.append(N)
                results[N] = {out: {dea: deepcopy(outer) for dea in DEAs} | {"cohorts": {}} for out in outlier_methods}
                
            for out in outlier_methods:
                if out not in results[N].keys():
                    missing.append(out)
                    results[N][out] = {dea: deepcopy(outer) for dea in DEAs} | {"cohorts": {}}
                    
                for dea in DEAs:
                    if dea not in results[N][out].keys():
                        missing.append(dea)
                        results[N][out][dea] = deepcopy(outer)
                        
        for out in outlier_methods:
            if out == "none": continue
            if out not in results["n_patients_df"]:
                results["n_patients_df"][out] = None
                        
        if len(missing): logging.info(f"New keys appended to results dict: {set(missing)}")
        else: logging.info("No new keys appended to results dict")
        
    return results

def check_failed_jobs_due_to_time_and_move(path) -> None:
    """Check if any jobs failed due to exceeding the time limit and if so, move the slurm file to correct folder
    path: path to folder where slurm files are stored before they are moved if job is successful
    """
    slurmfiles = glob.glob(f"{path}/slurm-*.out")
    
    for slurm in slurmfiles:
        with open(slurm, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            if line.startswith("Outpath cohort: "): 
                outpath_c = line.split("Outpath cohort: ")[1][:-1]
            if line.startswith("DEA: "):
                DEA = line.split("DEA: ")[1][:-1]
            if line.startswith("Outlier method: "):
                out = line.split("Outlier method: ")[1][:-1]
            if line.startswith("Parameter set: "):
                p = line.split("Parameter set: ")[1][:-1]
            if line.startswith("GSEA method: "):
                gsea = line.split("GSEA method: ")[1][:-1]
                
        slurmname = slurm.split("/")[-1].split(".out")[0]
        try:
            if "gsea" in locals():
                os.system(f"mv {slurm} {outpath_c}/gsea/slurm/{slurmname}.{out}.{DEA}.{gsea}.{p}.out")
                del outpath_c, DEA, out, p, gsea
            else:
                os.system(f"mv {slurm} {outpath_c}/slurm/{slurmname}.{out}.{DEA}.{p}.out")
                del outpath_c, DEA, out, p
        except UnboundLocalError as e:
            logging.info(f"{slurmname}\n{e}")

def process_slurm(results, outpath, outname, all_N, DEAs, outlier_methods, param_set, n_cohorts = "") -> dict:
    """
    Process the slurm log files: copy outlier patients to results dict; check if any jobs failed
    """

    total_slurmfiles, failed_slurmfiles = 0, 0
    
    for N in all_N:
        outpath_N = f"{outpath}/{outname}_N{N}"   
        cohorts = sorted([f.name for f in os.scandir(outpath_N) if f.is_dir()])
        if n_cohorts == "": n_cohorts = len(cohorts)

        for cohort in cohorts[:n_cohorts]:
            cohort_id = int(cohort.split("_")[-1])
            slurmpath = f"{outpath_N}/{cohort}/slurm"
            slurmfiles = glob.glob(f"{slurmpath}/slurm-*{param_set}.out")
            slurmfiles = sorted(slurmfiles, reverse=True)
            
            for slurm in slurmfiles:
                total_slurmfiles += 1
                with open(slurm, "r") as f:
                    lines = f.readlines()
                    
                # Look for failed jobs
                failed = False
                for line in lines:
                    if line.find('srun: error:') != -1 or line.find('slurmstepd: error:') != -1:
                        failed = True
                        
                    # If this line is found, it means the code finished successfully despite slurm errors
                    if line.find('Elapsed time:') != -1:
                        break
                else:
                    if failed:
                        failed_slurmfiles += 1
                        logging.info(f"{cohort} {slurm.split('/')[-1]} {line}")     

                # Read outliers and store in results file
                for line in lines:
                    if line.startswith("Outlier method: "): 
                        out = line.split("Outlier method: ")[-1][:-1]
                    elif line.startswith("Outliers_f: "):
                        outliers = line.split("Outliers_f: ")[1]
                        outliers = outliers[2:-3].split("', '")
                        if outliers[0] == "": outliers = []
                        outliers = outliers[:len(outliers)//2] # store only control samples
                        try: 
                            results[N][out]["cohorts"][cohort_id]["outliers"] = outliers
                        except KeyError:
                            results[N][out]["cohorts"][cohort_id] = {}
                            results[N][out]["cohorts"][cohort_id]["outliers"] = outliers
                        del out
                            
    logging.info(f"{failed_slurmfiles} jobs failed out of {total_slurmfiles}")        
    return results


def get_n_patients_df(results, all_N, outpath, outlier_method) -> pd.DataFrame:
    """
    Constructs df with three columns:
    Ni = number of initial patients
    Nf = number of final patients after outlier removal
    Cohort = Cohort number
    Stores df in results dict
    """

    df = pd.DataFrame()
    Nf, Ni, cohorts = [], [], []
    for N in all_N:
        cohorts_N = list(results[N][outlier_method]["cohorts"].keys())
        Nf += [N-len(results[N][outlier_method]["cohorts"][cohort]["outliers"]) for cohort in cohorts_N]
        Ni += [N]*len(cohorts_N) # faster than list comprehension
        cohorts += cohorts_N

    df["Ni"] = Ni
    df["Nf"] = Nf
    df["Cohort"] = cohorts
    return df

def process_results(results, outpath, outname, all_N, DEAs, outlier_methods, FDRs, logFCs, lfc_test, param_set, overwrite=False) -> dict:
    """
    Calcualte median replicability, #DEG, MCC, precision, recall for one dataset
    """

    truth_dict = {fdr: {logFC: pd.read_csv(f"{outpath}/truth.fdr{fdr}.post_lfc{logFC}.lfc{lfc_test}.csv", index_col=0) for logFC in logFCs} for fdr in FDRs}
    
    for N in all_N:
        print(f"N{N} ", end=" ")
        outpath_N = f"{outpath}/{outname}_N{N}"   
        
        for out in outlier_methods:
            for dea in DEAs:
             
                # Check which results already exist
                to_process = list(product(*[FDRs,logFCs]))
                for fdr, logFC in list(product(*[FDRs,logFCs])):
                    if (not overwrite and 
                        (results[N][out][dea]["median_rep"][fdr][logFC] is not None) and 
                        (results[N][out][dea]["median_deg"][fdr][logFC] is not None)):
                        to_process.remove((fdr,logFC))
                        
                if len(to_process) < 1: continue
                                
                df_lfc = open_table(f"{outpath_N}/all.logFC.{out}.{dea}.{param_set}.feather")
                df_fdr = open_table(f"{outpath_N}/all.FDR.{out}.{dea}.{param_set}.feather")

                for fdr, logFC in to_process:
                    
                    boolarr_deg = np.where((df_lfc.abs()>logFC) & (df_fdr<fdr), True, False)
                    results[N][out][dea]["median_deg"][fdr][logFC] = np.median(np.sum(boolarr_deg,axis=0))
                    results[N][out][dea]["median_rep"][fdr][logFC] = np.median(get_replicability_numba(boolarr_deg))
                    
                    truth_df = truth_dict[fdr][logFC]
                    truth = df_lfc.index.isin(truth_df.index)
                    mcc, prec, rec = get_array_metrics_numba(truth, boolarr_deg)
                    results[N][out][dea]["median_mcc"][fdr][logFC] = np.nanmedian(mcc)
                    results[N][out][dea]["median_prec"][fdr][logFC] = np.nanmedian(prec)
                    results[N][out][dea]["median_rec"][fdr][logFC] = np.nanmedian(rec)


    return results

def calc_rep_same_cohort_size(results, outpath, outname, all_N, DEAs, outlier_methods, FDRs, logFCs, lfc_test, param_set) -> dict:
      
    truth_dict = {fdr: {logFC: pd.read_csv(f"{outpath}/truth.fdr{fdr}.post_lfc{logFC}.lfc{lfc_test}.csv", index_col=0) for logFC in logFCs} for fdr in FDRs}

        
    for dea in DEAs:
        for out in outlier_methods:            
            if out == "none": continue     
            
            df_dict = {N: {} for N in all_N}
            for N in all_N:
                outpath_N = f"{outpath}/{outname}_N{N}"
                df_fdr = open_table(f"{outpath_N}/all.FDR.{out}.{dea}.{param_set}")
                df_logfc = open_table(f"{outpath_N}/all.logFC.{out}.{dea}.{param_set}")
                df_fdr.columns = df_fdr.columns.astype(str)
                df_logfc.columns = df_logfc.columns.astype(str)
                df_dict[N]["FDR"] = df_fdr#.sort_index()
                df_dict[N]["logFC"] = df_logfc#.sort_index()
                        
                    
            c = results["n_patients_df"][out]
            
            for Nf in all_N:
                cf = c[c["Nf"] == Nf] # Cohorts with Nf final patients
                df_nf_fdr, df_nf_logfc = [], []
                
                for Ni in set(cf["Ni"]):
                    ci = cf[cf["Ni"]==Ni]["Cohort"].astype(str).values # Cohorts with Ni initial and Nf final patients
                    df_nf_fdr.append(df_dict[Ni]["FDR"][ci])
                    df_nf_logfc.append(df_dict[Ni]["logFC"][ci])
                
                if len(df_nf_fdr) < 1:
                    continue
                    
                df_nf_fdr = pd.concat(df_nf_fdr, axis=1)
                df_nf_logfc = pd.concat(df_nf_logfc, axis=1)      
                
                
                for fdr in FDRs:
                    for logFC in logFCs:
                        
                        boolarr_deg = np.where((df_nf_logfc.abs()>logFC) & (df_nf_fdr<fdr), True, False)
                        r = get_replicability_numba(boolarr_deg)
                        results[Nf][out][dea]["median_deg_adj"][fdr][logFC] = np.median(np.sum(boolarr_deg,axis=0))
                        results[Nf][out][dea]["median_rep_adj"][fdr][logFC] = np.median(r) if len(r) > 1 else np.NaN
                        
                        truth_df = truth_dict[fdr][logFC]
                        truth = df_nf_logfc.index.isin(truth_df.index)
                        mcc, prec, rec = get_array_metrics_numba(truth, boolarr_deg)
                        results[Nf][out][dea]["median_mcc_adj"][fdr][logFC] = np.nanmedian(mcc)
                        results[Nf][out][dea]["median_prec_adj"][fdr][logFC] = np.nanmedian(prec)
                        results[Nf][out][dea]["median_rec_adj"][fdr][logFC] = np.nanmedian(rec)
                        
    return results

def merge_tables(outpath, outname, all_N, DEAs, outlier_methods, param_set, n_cohorts) -> None:
    """
    Merge tables from different cohorts
    """
    
    for N in all_N:
        outpath_N = f"{outpath}/{outname}_N{N}"   
        cohorts = sorted([f.name for f in os.scandir(outpath_N) if f.is_dir()])[:n_cohorts]

        for out in outlier_methods:

            for dea in DEAs:
                list_FDRs, list_logFCs, cohort_ids, tabs = [], [], [], []
                
                for cohort in cohorts:
                    cohort_id = str(int(cohort.split("_")[-1]))
                    cohort_ids.append(cohort_id)
                    outpath_c = f"{outpath_N}/{cohort}"
                    tab = open_table(f"{outpath_c}/tab.{out}.{dea}.{param_set}")
                    list_FDRs.append(tab["FDR"]), list_logFCs.append(tab["logFC"])

                FDR_concat = pd.concat(list_FDRs, axis=1, keys=cohort_ids).reset_index(drop=False)
                FDR_concat.to_feather(f"{outpath_N}/all.FDR.{out}.{dea}.{param_set}.feather")
                logFC_concat = pd.concat(list_logFCs, axis=1, keys=cohort_ids).reset_index(drop=False)
                logFC_concat.to_feather(f"{outpath_N}/all.logFC.{out}.{dea}.{param_set}.feather")

@Timer(name="decorator")
def process_pipeline(outpath, outname, all_N, DEAs, outlier_methods, FDRs, logFCs, lfc_test, param_set, overwrite=False, overwrite_merged=False, n_cohorts=""):
    
    #### Create or load the results dict
    
    resultsfile = f"{outpath}/results.{param_set}.txt"
    if not Path(resultsfile).is_file():
        results = {}
        pickler(results,resultsfile)
        overwrite = True
    elif overwrite:
        results = {}
    else:
        with open(resultsfile, "rb") as f:
            results = pickle.load(f)
            
    results = update_results_dict(results, all_N, DEAs, outlier_methods, FDRs, logFCs, overwrite=overwrite)
    
    #### Process
    
    # Read outliers from slurm file
    logging.info("Processing slurm files...")
    check_failed_jobs_due_to_time_and_move("../notebooks")
    results = process_slurm(results, outpath, outname, all_N, DEAs, outlier_methods, param_set, n_cohorts)
    
    # Construct n_patients_df
    for out in outlier_methods:
        if out == "none": continue
        results["n_patients_df"][out] = get_n_patients_df(results, all_N, outpath, out)

    # Merge tables from different cohorts
    if overwrite_merged:
        logging.info("Merging tables...")
        merge_tables(outpath, outname, all_N, DEAs, outlier_methods, param_set, n_cohorts)
    
    # Calculate median replicability, median #DEG, etc.
    logging.info("Processing results...")
    results = process_results(results, outpath, outname, all_N, DEAs, outlier_methods, FDRs, logFCs, lfc_test, param_set, overwrite)
    
    logging.info("\nProcessing adjusted results...")
    # Calculate median replicability, median #DEG, etc. adjusted for outlier removal
    results = calc_rep_same_cohort_size(results, outpath, outname, all_N, DEAs, outlier_methods, FDRs, logFCs, lfc_test, param_set)
    
    #### Save results
    
    pickler(results, resultsfile)
    logging.info("Done!")
    
    
def gene_rep(FDR_tab, logFC_tab = [], FDR=0.01, logFC=0, normalize=False):
    """
    For each gene, calculate number of times found DE in each cohort
    FDR_tab: pd.DataFrame of FDR scores for each cohort
    """
    
    if logFC > 0 and len(logFC_tab) < 1: raise Exception("Need logFC tab when logFC > 0")
    
    score = FDR_tab.where(FDR_tab < FDR, 0).where(FDR_tab >= FDR, 1).astype(bool)
    
    if logFC > 0:
        scorelfc = logFC_tab.where(logFC_tab.abs() <= logFC, 1).where(logFC_tab.abs() > logFC, 0).astype(bool)
        score = score & scorelfc
    
    score = score.sum(axis=1)

    if normalize:
        n_cohorts = len(set(FDR_tab.columns.get_level_values(level=0)))
        return score / n_cohorts
    else:
        return score


def find_ground_truth(datasets, DEAs, FDRs, logFCs, lfc_test):
    """Find ground truth of differentially expressed genes
    
    Creates two truth dataframes:
    lfcs_df: mean logFC estimate of all DEA methods for all genes; shape = n_genes x 1
    deg_df: for a given fdr, lfc cutoff, store lfcs estimates of all DEA methods for intersection of passing genes; shape = n_deg x n_methods
    
    Additionally, creates a stats dictionary that stores jaccard, intersection, union, n_deg of the different methods
    
    lfc_test: the threshold at which differential expression was tested by edegR or deseq2
    logFCs: post hoc lfc thresholds
    """

    stats_dict_file = "../data/multi/stats_dict.txt"
    os.system(f"mkdir -p ../data/multi")
        
    if Path(stats_dict_file).is_file():
        with open(stats_dict_file, "rb") as f:
            stats_dict = pickle.load(f)
    
        for data in datasets:
            if "truth_stats" not in datasets[data]: datasets[data]["truth_stats"] = {}
            datasets[data]["truth_stats"][lfc_test] = stats_dict[data]
        return datasets
    
    stats_dict = {data: {fdr: {logFC: {} for logFC in logFCs} for fdr in FDRs} for data in datasets}
    
    for data in datasets:  

        dea_dict = {dea: {fdr: {logfc: {"DEGs":None} for logfc in logFCs} for fdr in FDRs} for dea in DEAs}
        lfcs = []
        for dea in DEAs:
            d = datasets[data]["datapath"]
            p = Path(d.parent, datasets[data]["datapath"].stem + f".{dea}.lfc{lfc_test}.csv")
            tab = pd.read_csv(p, index_col=0)
            lfcs.append(tab["logFC"])
    
            for fdr in FDRs:
                for logFC in logFCs:
                    dea_dict[dea][fdr][logFC]["DEGs"] = tab[(tab["FDR"]<fdr) & (tab["logFC"].abs() > logFC)]["logFC"]
             
        # Store logFC truth df of all genes
        savepath = Path(datasets[data]["outpath"], "truth_lfc.csv")
        if overwrite or not Path(savepath).is_file():
            common = set.intersection(*[set(l.index) for l in lfcs])
            lfcs = [l.loc[common] for l in lfcs]
            lfcs_df = pd.DataFrame(np.mean(np.array(lfcs).T, axis=1), index=common, columns=["logFC"])
            lfcs_df.to_csv(savepath)
        
        
        # Store DEG truth for all fdr, lfc cutoffs
        for fdr in FDRs:
            for logFC in logFCs:

                sets = []
                for dea in DEAs:
                    sets.append(set(dea_dict[dea][fdr][logFC]["DEGs"].index))
                    stats_dict[data][fdr][logFC][dea] = len(sets[-1])

                inter, union = set.intersection(*sets), set.union(*sets)

                stats_dict[data][fdr][logFC]["jaccard"] = len(inter)/len(union)
                stats_dict[data][fdr][logFC]["union"] = len(union)
                stats_dict[data][fdr][logFC]["inter"] = len(inter)
                
                savepath = Path(datasets[data]["outpath"], f"truth.fdr{fdr}.post_lfc{logFC}.lfc{lfc_test}.csv")
                if not overwrite and Path(savepath).is_file(): continue
                
                deg_df = pd.DataFrame(index=inter,columns=DEAs)
                for dea in DEAs:
                    lfc = dea_dict[dea][fdr][logFC]["DEGs"]
                    deg_df[dea] = lfc.loc[lfc.index.intersection(inter)]
                #datasets[data]["fulldata"]["combined"][fdr][logFC]["truth"] = logFC_df
                deg_df.to_csv(savepath)
                
        if "truth_stats" not in datasets[data]: datasets[data]["truth_stats"] = {}
        datasets[data]["truth_stats"][lfc_test] = stats_dict[data]      
    
    pickler(stats_dict, stats_dict_file)
    return datasets


def get_init_samples_from_cohort(configfile):
    """From the config.json read the subsamples from before outlier removal"""
    with open(configfile, "r") as f:
        configdict = json.load(f)    
    return configdict["samples_i"] if "samples_i" in configdict else []


###############################################################
###################### Enrichment #############################
###############################################################

@Timer(name="decorator")
def gsea_process_pipeline(outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, 
                          FDRs, gsea_param_set, overwrite=False, overwrite_merged=False, n_cohorts="",calculate_common=True):
  
    #### Create or load the results dict
    
    resultsfile = f"{outpath}/gsea_results.txt"
    if not Path(resultsfile).is_file():
        results = {}
        pickler(results,resultsfile)
        overwrite = True
    elif overwrite:
        results = {}
    else:
        with open(resultsfile, "rb") as f:
            results = pickle.load(f)
            
    check_failed_jobs_due_to_time_and_move("../notebooks")
    results = update_gsea_results_dict(results, all_N, DEAs, outlier_methods, gsea_methods, libraries, FDRs, overwrite=overwrite)
    
    # Process slurm files, check for failed jobs
    logging.info("Processing slurm files...")
    #process_gsea_slurm(results, outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, n_cohorts=n_cohorts)
    
    if calculate_common:
        logging.info("Recalculating common terms")
        calculate_common_fdr(outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, gsea_param_set, n_cohorts)
    
    # Merge tables from different cohorts
    if overwrite_merged:
        logging.info("Merging tables...")
        merge_gsea_tables(outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, gsea_param_set, n_cohorts=n_cohorts)
        merge_gsea_tables(outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, gsea_param_set, n_cohorts=n_cohorts, mode="FDR.common")
    
    # Calculate replicability, metrics
    logging.info("Processing results...")
    results = process_gsea_results(results, outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, FDRs, gsea_param_set, overwrite=overwrite)
    
    logging.info("\nFinish all N jobs before calculating adjusted results...")
#     # Calculate adjusted replicability, metrics
#     logging.info("\nProcessing adjusted results...")
    
#     # Get n_patients_df
#     dea_resultsfile = f"{outpath}/results.txt"
#     with open(dea_resultsfile,"rb") as f:
#         dea_results = pickle.load(f)

#     for out in outlier_methods:
#         if out == "none": continue
#         results["n_patients_df"][out] = dea_results["n_patients_df"][out]

#     results = calc_gsea_rep_same_cohort_size(results, outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, FDRs, gsea_param_set)
    
    # Delete redundant slurm files, tmp folders
    delete_redundant_slurmfiles(outpath, outname, all_N, gsea=True)

    #### Save results
    
    pickler(results, resultsfile)
    logging.info(f"Saved results in {resultsfile}")


def update_gsea_results_dict(results, all_N, DEAs, outlier_methods, gsea_methods, libraries, FDRs, overwrite=False) -> dict:
    """
    Add missing keys to results dict or create it from scratch
    """
    
    inner = {fdr: None for fdr in FDRs}
    outer = {"median_rep": deepcopy(inner), 
             "median_terms": deepcopy(inner), 
             "median_mcc": deepcopy(inner), 
             "median_prec": deepcopy(inner), 
             "median_rec": deepcopy(inner),
             "median_rep_adj": deepcopy(inner), 
             "median_terms_adj": deepcopy(inner), 
             "median_mcc_adj": deepcopy(inner), 
             "median_prec_adj": deepcopy(inner), 
             "median_rec_adj": deepcopy(inner),
             "median_rep_common": deepcopy(inner), 
             "median_terms_common": deepcopy(inner), 
             "median_mcc_common": deepcopy(inner), 
             "median_prec_common": deepcopy(inner), 
             "median_rec_common": deepcopy(inner),
             "median_rep_adj_common": deepcopy(inner), 
             "median_terms_adj_common": deepcopy(inner), 
             "median_mcc_adj_common": deepcopy(inner), 
             "median_prec_adj_common": deepcopy(inner), 
             "median_rec_adj_common": deepcopy(inner)}
    
    if overwrite:
        logging.info("Creating gsea results dict from scratch")
        results = {N: {out: {dea: {gsea: {library: deepcopy(outer) for library in libraries} for gsea in gsea_methods}  for dea in DEAs} for out in outlier_methods} for N in all_N}
        results["n_patients_df"] = {out: None for out in outlier_methods if out != "none"}
        
    else:
        logging.info("Checking for missing keys")
        missing = []
        
        for N in all_N:
            if N not in results:
                missing.append(N)
                results[N] = {out: {dea: {gsea: {library: deepcopy(outer) for library in libraries} for gsea in gsea_methods}  for dea in DEAs} for out in outlier_methods}
                
            for out in outlier_methods:
                if out not in results[N].keys():
                    missing.append(out)
                    results[N][out] = {dea: {gsea: {library: deepcopy(outer) for library in libraries} for gsea in gsea_methods}  for dea in DEAs}
                    
                for dea in DEAs:
                    if dea not in results[N][out].keys():
                        missing.append(dea)
                        results[N][out][dea] = {gsea: {library: deepcopy(outer) for library in libraries} for gsea in gsea_methods}
                        
                    for gsea in gsea_methods:
                        if gsea not in results[N][out][dea].keys():
                            missiing.append(gsea)
                            results[N][out][dea][gsea] = {library: deepcopy(outer) for library in libraries}
                            
                        for library in libraries:
                            if library not in results[N][out][dea][gsea].keys():
                                missing.append(library)
                                results[N][out][dea][gsea][library] = deepcopy(outer)
                        
        if "n_patients_df" not in results:
            results["n_patients_df"] = {out: None for out in outlier_methods if out != "none"}
            
        for out in outlier_methods:
            if out == "none": continue
            if out not in results["n_patients_df"]:
                results["n_patients_df"][out] = None
                        
        if len(missing): logging.info(f"New keys appended to results dict: {set(missing)}")
        else: logging.info("No new keys appended to results dict")
        
    return results

def process_gsea_slurm(results, outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, n_cohorts = "") -> dict:
    """
    Process the slurm log files: check if any jobs failed
    """

    total_slurmfiles, failed_slurmfiles = 0, 0
    
    for N in all_N:
        outpath_N = f"{outpath}/{outname}_N{N}"   
        cohorts = sorted([f.name for f in os.scandir(outpath_N) if f.is_dir()])
        if n_cohorts == "": n_cohorts = len(cohorts)

        for cohort in cohorts[:n_cohorts]:
            cohort_id = int(cohort.split("_")[-1])
            slurmpath = f"{outpath_N}/{cohort}/gsea/slurm"
            slurmfiles = glob.glob(f"{slurmpath}/slurm-*.out")
            slurmfiles = sorted(slurmfiles, reverse=True)
            
            for slurm in slurmfiles:
                total_slurmfiles += 1
                with open(slurm, "r") as f:
                    lines = f.readlines()
                    
                # Look for failed jobs
                failed = False
                for line in lines:
                    if line.find('srun: error:') != -1 or line.find('slurmstepd: error:') != -1:
                        failed = True
                        
                    # If this line is found, it means the code finished successfully despite slurm errors
                    if line.find('Elapsed time:') != -1:
                        break
                else:
                    if failed:
                        failed_slurmfiles += 1
                        logging.info(f"{cohort} {slurm.split('/')[-1]} {line}")     

    logging.info(f"{failed_slurmfiles} jobs failed out of {total_slurmfiles}")        
    return results


def calculate_common_fdr(outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, gsea_param_set, n_cohorts=0) -> None:
    """
    """
    
    file_gobp = Path("../data/multi/common_gobp.txt")
    file_kegg = Path("../data/multi/common_kegg.txt")
    with open(file_gobp, "rb") as f:
        common_gobp = pickle.load(f)
    with open(file_kegg, "rb") as f:
        common_kegg = pickle.load(f)
    
    for N in all_N:
        outpath_N = f"{outpath}/{outname}_N{N}"   
        cohorts = sorted([f.name for f in os.scandir(outpath_N) if f.is_dir()])
        
        if n_cohorts < 1: n_cohorts == len(cohorts)
        
        for out in outlier_methods:
            for dea in DEAs:
                for gsea in gsea_methods:                    
                    if "s2n" in gsea and dea != "edgerqlf": continue
                    if "gsea" in gsea: pvalue = "NOM p-val"
                    else: pvalue = "pvalue"
                    
                    for library in libraries:
                        if library == "GO_Biological_Process_2021": common = common_gobp
                        elif library == "KEGG_2021_Human": common = common_kegg
                        else: raise Exception(f"Invalid library: {library}")

                        for cohort in cohorts[:n_cohorts]:
                            cohort_id = str(int(cohort.split("_")[-1]))
                            outpath_c = f"{outpath_N}/{cohort}/gsea"
                            file = f"{outpath_c}/{gsea}.{library}.{dea}.{out}.{gsea_param_set}"
                            terms = open_table(file)
                            
                            if library == "GO_Biological_Process_2021":
                                ix = common.intersection(terms.index)
                            elif library == "KEGG_2021_Human":
                                try:
                                    ix = terms[terms["Term"].isin(common)].index
                                except KeyError:
                                    ix = common.intersection(terms.index)
                                
                            #print(len(ix),N,out,dea,gsea,library,cohort)
                            try:
                                terms.loc[ix, "FDR.common"] = multipletests(terms.loc[ix, pvalue], method="fdr_bh")[1]
                            except ZeroDivisionError:
                                if len(terms) == 1:
                                    terms.loc[ix, "FDR.common"] = terms.loc[ix, "FDR"]
                                else:
                                    display(terms)
                                    display(ix)
                                    print(N,out,dea,gsea,library)
                                    assert 0
                            terms.reset_index().to_feather(file+".feather")
                            terms = open_table(file+".feather")
                            display(file)
                            
def merge_gsea_tables(outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, gsea_param_set, n_cohorts=0, mode="FDR") -> None:
    """
    Merge tables from different cohorts
    mode: either "FDR" for pipeline-specific ground truth or "FDR.common" for intersection of ground truths
    """
    
    for N in all_N:
        outpath_N = f"{outpath}/{outname}_N{N}"   
        cohorts = sorted([f.name for f in os.scandir(outpath_N) if f.is_dir()])
        
        if n_cohorts < 1: n_cohorts == len(cohorts)
        
        for out in outlier_methods:
            for dea in DEAs:
                for gsea in gsea_methods:
                    if "s2n" in gsea and dea != "edgerqlf": continue
                    for library in libraries:
                        list_FDRs, cohort_ids = [], []
                
                        for cohort in cohorts[:n_cohorts]:
                            cohort_id = str(int(cohort.split("_")[-1]))
                            cohort_ids.append(cohort_id)
                            outpath_c = f"{outpath_N}/{cohort}/gsea"
                            tab = open_table(f"{outpath_c}/{gsea}.{library}.{dea}.{out}.{gsea_param_set}")
                            
                            ## for ORA KEGG: switch term ID and description for index
                            if tab.index[0].startswith("hsa") and "Term" in tab:
                                tab = tab.set_index("Term")
                            
                            list_FDRs.append(tab[mode])

                        FDR_concat = pd.concat(list_FDRs, axis=1, keys=cohort_ids).reset_index(drop=False)
                        FDR_concat.to_feather(f"{outpath_N}/all.{mode}.{gsea}.{library}.{out}.{dea}.{gsea_param_set}.feather")
                        
                        
                        
def process_gsea_results(results, outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, FDRs, gsea_param_set, overwrite=False) -> dict:
    """
    Calcualte median replicability, #DEG, MCC, precision, recall for one dataset
    """
    modes = ["FDR", "FDR.common"]
    truth_dict = {gsea: {library: {mode: {fdr: open_table(f"{outpath}/gsea/{gsea}.{library}.feather")[mode] for fdr in FDRs} for mode in modes} for library in libraries} for gsea in gsea_methods}
    file_gobp = Path("../data/multi/common_gobp.txt")
    file_kegg = Path("../data/multi/common_kegg.txt")
    with open(file_gobp, "rb") as f:
        common_gobp = pickle.load(f)
    with open(file_kegg, "rb") as f:
        common_kegg = pickle.load(f)
    
    for N in all_N:
        print(f"N{N} ", end=" ")
        outpath_N = f"{outpath}/{outname}_N{N}"   
        
        for out in outlier_methods:
            for dea in DEAs:
                for gsea in gsea_methods:
                    if "s2n" in gsea and dea != "edgerqlf": continue
                    
                    for library in libraries:
                        if library == "GO_Biological_Process_2021": len_truth = len(common_gobp)
                        elif library == "KEGG_2021_Human": len_truth = len(common_kegg)
                        else: raise Exception(f"Invalid library: {library}")
                        
                        for mode in modes:
                            mode_suffix = "" if mode == "FDR" else "_common"
             
                            # Check which results already exist
                            to_process = FDRs
                            for fdr in FDRs:
                                if (not overwrite and 
                                    (results[N][out][dea][gsea][library]["median_rep"+mode_suffix][fdr] is not None) and 
                                    (results[N][out][dea][gsea][library]["median_terms"+mode_suffix][fdr] is not None)):
                                    to_process.remove(fdr)

                            if len(to_process) < 1: continue

                            df_fdr = open_table(f"{outpath_N}/all.{mode}.{gsea}.{library}.{out}.{dea}.{gsea_param_set}.feather")

                            for fdr in to_process:
                                
                                truth_df = truth_dict[gsea][library][mode][fdr]
                                
                                if mode == "FDR.common" and library=="GO_Biological_Process_2021":
                                    common = common_gobp
                                elif mode == "FDR.common" and library == "KEGG_2021_Human":
                                    common = common_kegg
                                elif mode != "FDR.common":
                                    common = truth_df.index
                                    
                                truth_df = truth_df.loc[common]
                                
                                # if terms missing in df, append with FDR = 1
                                if mode == "FDR.common":
                                    ix=truth_df.index.difference(df_fdr.index)
                                    if len(ix) > 0:
                                        df_fdr = pd.concat([df_fdr,pd.DataFrame(np.ones(shape=(len(ix),len(df_fdr.columns))), index=ix, columns=df_fdr.columns)])
                                                                 
                                df_fdr = df_fdr.loc[common.intersection(df_fdr.index)]
                                truth_df = truth_df.loc[df_fdr.index]
                                
                                boolarr = np.where(df_fdr<fdr, True, False)

                                results[N][out][dea][gsea][library]["median_terms"+mode_suffix][fdr] = np.median(np.sum(boolarr,axis=0))
                                results[N][out][dea][gsea][library]["median_rep"+mode_suffix][fdr] = np.median(get_replicability_numba(boolarr))
                                    
                                truth = (truth_df < fdr).values
                                
                                # if (results[N][out][dea][gsea][library]["median_rep"+mode_suffix][fdr] == 0):
                                #     display(truth_df)
                                #     display(common)
                                #     display(truth.shape)
                                #     display(boolarr.shape)
                                #     display(truth.sum())
                                #     display(boolarr.sum(axis=1))
                                #     display(df_fdr)
                                #     print(N,dea,gsea,out,library,mode,fdr)
                                #     assert 0
                                    
                                try:
                                    mcc, prec, rec = get_array_metrics_numba(truth, boolarr)
                                except ValueError:
                                    display(truth.shape)
                                    display(boolarr.shape)
                                    display(truth_df)
                                    print(mode,library,gsea)
                                    assert 0
                                results[N][out][dea][gsea][library]["median_mcc"+mode_suffix][fdr] = np.nanmedian(mcc)
                                results[N][out][dea][gsea][library]["median_prec"+mode_suffix][fdr] = np.nanmedian(prec)
                                results[N][out][dea][gsea][library]["median_rec"+mode_suffix][fdr] = np.nanmedian(rec)
    return results

def get_n_gsea_truth(results, outpath, outname, all_N, DEAs, outlier_methods, gsea_methods, libraries, FDRs, gsea_param_set, overwrite=False) -> dict:
    """
    
    """
    modes = ["FDR", "FDR.common"]
    truth_dict = {gsea: {library: {mode: {fdr: open_table(f"{outpath}/gsea/{gsea}.{library}.feather")[mode] for fdr in FDRs} for mode in modes} for library in libraries} for gsea in gsea_methods}
    file_gobp = Path("../data/multi/common_gobp.txt")
    file_kegg = Path("../data/multi/common_kegg.txt")
    with open(file_gobp, "rb") as f:
        common_gobp = pickle.load(f)
    with open(file_kegg, "rb") as f:
        common_kegg = pickle.load(f)
    
    outpath_N = f"{outpath}/{outname}_N15"   

    for out in outlier_methods:
        for dea in DEAs:
            for gsea in gsea_methods:
                if "s2n" in gsea and dea != "edgerqlf": continue

                for library in libraries:
                    if library == "GO_Biological_Process_2021": len_truth = len(common_gobp)
                    elif library == "KEGG_2021_Human": len_truth = len(common_kegg)
                    else: raise Exception(f"Invalid library: {library}")

                    for mode in modes:
                        mode_suffix = "" if mode == "FDR" else "_common"

                        df_fdr = open_table(f"{outpath_N}/all.{mode}.{gsea}.{library}.{out}.{dea}.{gsea_param_set}.feather")

                        for fdr in FDRs:

                            truth_df = truth_dict[gsea][library][mode][fdr]

                            if mode == "FDR.common" and library=="GO_Biological_Process_2021":
                                common = common_gobp
                            elif mode == "FDR.common" and library == "KEGG_2021_Human":
                                common = common_kegg
                            elif mode != "FDR.common":
                                common = truth_df.index

                            truth_df = truth_df.loc[common]

                            # if terms missing in df, append with FDR = 1
                            if mode == "FDR.common":
                                ix=truth_df.index.difference(df_fdr.index)
                                if len(ix) > 0:
                                    df_fdr = pd.concat([df_fdr,pd.DataFrame(np.ones(shape=(len(ix),len(df_fdr.columns))), index=ix, columns=df_fdr.columns)])

                            df_fdr = df_fdr.loc[common.intersection(df_fdr.index)]
                            truth_df = truth_df.loc[df_fdr.index]
                            truth = (truth_df < fdr).values

                            results[out][dea][gsea][library]["truth"+mode_suffix][fdr] = np.sum(truth)

    return results