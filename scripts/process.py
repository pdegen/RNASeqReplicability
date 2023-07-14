import os
import glob
import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from copy import deepcopy
from itertools import combinations, product
from statsmodels.stats.multitest import multipletests
from misc import Timer, pickler, open_table

def find_ground_truth(datasets, DEAs, FDRs, logFCs, lfc_test, overwrite = False):
    """Find ground truth of differentially expressed genes
    
    Creates two truth dataframes:
    lfcs_df: mean logFC estimate of all DEA methods for all genes; shape = n_genes x 1
    deg_df: for a given fdr, lfc cutoff, store lfcs estimates of all DEA methods for intersection of passing genes; shape = n_deg x n_methods
    
    Additionally, creates a stats dictionary that stores jaccard, intersection, union, n_deg of the different methods
    
    lfc_test: the threshold at which differential expression was tested by edegR or deseq2
    logFCs: post hoc lfc thresholds
    """

    stats_dict_file = "/storage/homefs/pd21v747/datanew/multi/stats_dict.txt"
        
    if not overwrite and Path(stats_dict_file).is_file():
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
            p = datasets[data]["datapath"].split(".csv")[0] + f".{dea}.lfc{lfc_test}.csv"
            tab = pd.read_csv(p, index_col=0)
            lfcs.append(tab["logFC"])
    
            for fdr in FDRs:
                for logFC in logFCs:
                    dea_dict[dea][fdr][logFC]["DEGs"] = tab[(tab["FDR"]<fdr) & (tab["logFC"].abs() > logFC)]["logFC"]
             
        # Store logFC truth df of all genes
        savepath = datasets[data]["outpath"] + f"/truth_lfc.csv"
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
                
                savepath = datasets[data]["outpath"] + f"/truth.fdr{fdr}.post_lfc{logFC}.lfc{lfc_test}.csv"
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