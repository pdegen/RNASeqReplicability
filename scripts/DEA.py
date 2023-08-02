import os, sys
import logging
import pandas as pd
import rpy2.robjects as ro
from pathlib import Path
from R_wrappers import pd_to_R
from misc import Timer

def run_dea(df, outfile, method, overwrite, design="paired", lfc=0, **kwargs):
    """Wrapper to call appropriate R method to run differential expression analysis
        
    Parameters
    ----------
    df : pd.DataFrame, count data with m rows, n columns
    method: str, "edgerqlf", "edgerlrt" or "deseq2"
    overwrite: bool, overwrite existing results table if it already exists
    design: str, only use "paired" design matrix for this project
    lfc: float, formal log2 fold change threshold when testing for differential expression
    kwargs: additional keyword arguments passed to R method
    """
    
    # This function should be called from a notebook in the notebooks folder, we need to load a script from the scripts folder
    wd = Path(str(Path(os.getcwd()).parent) + "/scripts/R_functions.r")
    ro.r['source'](str(wd)) # Loading the R script
    
    # Converting pd to R dataframe
    df_r = df if isinstance(df, ro.vectors.DataFrame) else pd_to_R(df)
    
    if method in ["edgerqlf","edgerlrt"]:
        logging.info(f"\nCalling edgeR in R with kwargs:\n{kwargs}\n")
        edgeR_QLF = ro.globalenv['run_edgeR'] # Finding the R function in the script
        edgeR_QLF(df_r, str(outfile), design, overwrite, lfc=lfc, **kwargs)
        
    elif method == "deseq2":
        logging.info(f"\nCalling DESeq2 in R with kwargs:\n{kwargs}\n")
        if isinstance(df, ro.vectors.DataFrame) or design != "paired":
            raise Exception("Not yet implemented for DESeq2: general design matrix or calling directly with df_r")
        DESeq2 = ro.globalenv['run_deseq2']
        DESeq2(df_r, str(outfile), design, overwrite=overwrite, lfc=lfc, **kwargs)
        
    else: 
        raise Exception(f"Method {method} not implemented")
    
    
def run_dea_on_full_data(datasets, DEAs, lfcs, overwrite = False, truncate_cohorts = 0):
    """Run differential expression analysis on parent data with all cohorts f
        
    Parameters
    ----------
    datasets: dict
    DEAs: list
    lfcs: list
    overwrite: bool
    truncate_cohorts: int, if greater than 0, only use the first int cohorts to speed up computation
    """
    
    for d in datasets:
        dpath = str(datasets[d]["datapath"])
        for lfc in lfcs:
            for dea in DEAs:
                respath = Path(dpath.split("csv")[0] + dea + ".lfc" + str(lfc) + ".csv")
                if not respath.is_file() or overwrite:
                    logging.info(f"Running DEA for {d} {dea}")
                    df_full = pd.read_csv(dpath, index_col=0)
                    
                    if truncate_cohorts > 0:
                        logging.info(f"Truncating to first {truncate_cohorts} cohorts")
                        n = len(df_full.columns)//2
                        assert truncate_cohorts <= n
                        df_full = df_full.iloc[:,list(range(truncate_cohorts))+list(range(n,n+truncate_cohorts))]
                        
                    with Timer(name="context manager"):
                        if dea == "deseq2": 
                            run_dea(df_full, respath, method=dea, overwrite=overwrite, design="paired", cols_to_keep="all",lfc=lfc)
                        elif dea == "edgerlrt": 
                            run_dea(df_full, respath, method=dea, overwrite=overwrite, design="paired", cols_to_keep="all", test="lrt",lfc=lfc)
                        elif dea == "edgerqlf": 
                            run_dea(df_full, respath, method=dea, overwrite=overwrite, design="paired", cols_to_keep="all", test="qlf",lfc=lfc)
                        else: 
                            raise Exception(f"Method {dea} not implemented")
                    
                    
def normalize_counts(df):
    """Use DESeq2 estimateSizeFactors to normalize a count matrix"""
    
    # This function should be called from a notebook in the notebooks folder, we need to load a script from the scripts folder
    wd = Path(str(Path(os.getcwd()).parent) + "/scripts/R_functions.r")
    ro.r['source'](str(wd)) # Loading the R script
    
    df_r = df if isinstance(df, ro.vectors.DataFrame) else pd_to_R(df) # Converting to R dataframe
    DESeq2 = ro.globalenv['deseq2']
    sizefactors = DESeq2(df_r, outfile="", design="paired", overwrite=True, 
                         print_summary=False, cols_to_keep="", size_factors_only=True)
    return df/list(sizefactors)

# def get_edgeR_paired_meta(df, overwrite=False, filter_expr=False):
    
#     ro.r['source']('/storage/homefs/pd21v747/RepProject/scripts/R_functions.R') # Loading the R script    
#     df_r = df if isinstance(df, ro.vectors.DataFrame) else pd_to_R(df) # Converting to R dataframe
#     edgeR_QLF = ro.globalenv['edgeR_QLF'] # Finding the R function in the script
#     return edgeR_QLF(df_r, outfile="", design="paired", overwrite=overwrite, filter_expr=filter_expr, meta_only=True)