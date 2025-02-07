import os, sys
import logging
import pandas as pd
import rpy2.robjects as ro
from pathlib import Path
from R_wrappers import pd_to_R
from misc import Timer

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger

def load_rprofile():
    print("loading .Rprofile...")
    # Set the working directory to the project root (where .Rprofile is located)
    cwd = os.getcwd()
    print(cwd)
    project_root = Path(os.getcwd()).parent
    os.chdir(str(project_root))
    
    # Manually source .Rprofile to ensure it is loaded
    rprofile_path = project_root / '.Rprofile'
    if rprofile_path.exists():
        ro.r['source'](str(rprofile_path))
        print(".Rprofile loaded successfully")
    else:
        print(".Rprofile not found in project root")
        
    os.chdir(cwd)
    print(cwd)
    
def run_dea(df, outfile, method, overwrite, design="paired", lfc=0, verbose=False, **kwargs):
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

    load_rprofile()
    # This function should be called from a notebook in the notebooks folder, we need to load a script from the scripts folder
    wd = Path(str(Path(os.getcwd()).parent) + "/scripts/R_functions.r")
    ro.r['source'](str(wd))  # Loading the R script

    # Converting pd to R dataframe
    df_r = df if isinstance(df, ro.vectors.DataFrame) else pd_to_R(df)

    if not verbose:
        rpy2_logger.setLevel(logging.ERROR)

    if method in ["edgerqlf", "edgerlrt"]:
        logging.info(f"\nCalling edgeR in R with kwargs:\n{kwargs}\n")
        edgeR = ro.globalenv['run_edgeR']  # Finding the R function in the script
        edgeR(df_r, str(outfile), design, overwrite, lfc=lfc, **kwargs)

    elif method == "deseq2":
        logging.info(f"\nCalling DESeq2 in R with kwargs:\n{kwargs}\n")
        if isinstance(df, ro.vectors.DataFrame):
            raise Exception("Not yet implemented for DESeq2: calling directly with df_r")
        DESeq2 = ro.globalenv['run_deseq2']
        DESeq2(df_r, str(outfile), design, overwrite=overwrite, lfc=lfc, **kwargs)
        
    elif method == "wilcox":
        logging.info(f"\nCalling Wilcoxon rank-sum")
        wilcox_test(df, outfile=str(outfile), design=design, overwrite=overwrite)

    else:
        raise Exception(f"Method {method} not implemented")

def run_dea_on_full_data(datasets, DEAs, lfcs, design, overwrite=False, truncate_cohorts=0):
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
        mfile = str(datasets[d]["metafile"])
        design_i = mfile if design == "custom" else design
        for lfc in lfcs:
            for dea in DEAs:
                if dea == "wilcox" and design == "custom":
                    logging.info("Skipping Wilcox test with general design matrix")
                    continue
                respath = Path(dpath.split("csv")[0] + dea + ".lfc" + str(lfc) + ".csv")
                if not respath.is_file() or overwrite:
                    
                    if dea == "wilcox" and lfc != 0:
                        continue
                    
                    logging.info(f"Running DEA for {d} {dea} lfc{lfc}")
                    print(os.system("pwd"))
                    df_full = pd.read_csv(dpath, index_col=0)

                    if truncate_cohorts > 0:
                        logging.info(f"Truncating to first {truncate_cohorts} cohorts")
                        n = len(df_full.columns) // 2
                        assert truncate_cohorts <= n
                        df_full = df_full.iloc[:, list(range(truncate_cohorts)) + list(range(n, n + truncate_cohorts))]

                    with Timer(name="context manager"):
                        if dea == "deseq2":
                            run_dea(df_full, respath, method=dea, overwrite=overwrite, design=design_i,
                                    cols_to_keep="all", lfc=lfc)
                        elif dea == "edgerlrt":
                            run_dea(df_full, respath, method=dea, overwrite=overwrite, design=design_i,
                                    cols_to_keep="all", test="lrt", lfc=lfc)
                        elif dea == "edgerqlf":
                            run_dea(df_full, respath, method=dea, overwrite=overwrite, design=design_i,
                                    cols_to_keep="all", test="qlf", lfc=lfc)
                        elif dea == "wilcox":
                            if lfc == 0:
                                run_dea(df_full, respath, method=dea, overwrite=overwrite, design=design_i)
                        else:
                            raise Exception(f"Method {dea} not implemented")


def normalize_counts(df):
    """Use DESeq2 estimateSizeFactors to normalize a count matrix"""
    load_rprofile()
    # This function should be called from a notebook in the notebooks folder, we need to load a script from the scripts folder
    wd = Path(str(Path(os.getcwd()).parent) + "/scripts/R_functions.r")
    ro.r['source'](str(wd))  # Loading the R script

    df_r = df if isinstance(df, ro.vectors.DataFrame) else pd_to_R(df)  # Converting to R dataframe
    DESeq2 = ro.globalenv['run_deseq2']
    sizefactors = DESeq2(df_r, outfile="", design="paired", overwrite=True,
                         print_summary=False, cols_to_keep="", size_factors_only=True)
    return df / list(sizefactors)


from scipy.stats import ranksums, wilcoxon
from statsmodels.stats.multitest import multipletests

def wilcox_test(df_unnormalized, design, outfile="", overwrite=False):
    """Perform Wilcoxon rank sum test for a count matrix"""
    
    if design == "paired":
        if len(df_unnormalized.columns) % 2 != 0:
            raise Exception("df must have even number of columns for paired design")
        print("Paired design")
        test_func = wilcoxon

    elif design == "unpaired":
        print("Unpaired design")
        test_func = ranksums
    else:
        raise Exception("General design matrix not supported, must be paired or unpaired")
        
    if os.path.isfile(outfile) and not overwrite:
        logging.info("Existing file not overwritten")
        return
    
    df = normalize_counts(df_unnormalized)
    
    wilcoxon_results = {}
    for i, gene in enumerate(df.index):
        control_values = df.iloc[i, :len(df.columns)//2]
        treatment_values = df.iloc[i, len(df.columns)//2:]

        try:
            statistic, p_value = test_func(control_values, treatment_values)
        except ValueError: # wilcox can fail for insufficient sample size
            statistic, p_value = np.nan, np.nan
        
        wilcoxon_results[gene] = {'statistic': statistic, 'p_value': p_value}

    wilcoxon_results_df = pd.DataFrame.from_dict(wilcoxon_results, orient='index')
    p_values = wilcoxon_results_df['p_value']
    
    _, corrected_p_values, _, _ = multipletests(p_values.dropna(), method='fdr_bh')
    wilcoxon_results_df.loc[~wilcoxon_results_df['p_value'].isna(), 'FDR'] = corrected_p_values    

    if outfile != "":
        save_df(wilcoxon_results_df, outfile)
    return wilcoxon_results_df


def save_df(df, path):
    file_extension = path.split('.')[-1].lower()
    if file_extension == 'csv':
        df.to_csv(path)
    elif file_extension == 'feather':
        df.to_feather(path)
    else:
        raise ValueError("Unsupported file extension. Supported extensions are: .csv and .feather")

# def get_edgeR_paired_meta(df, overwrite=False, filter_expr=False):

#     ro.r['source']('/storage/homefs/pd21v747/RepProject/scripts/R_functions.R') # Loading the R script    
#     df_r = df if isinstance(df, ro.vectors.DataFrame) else pd_to_R(df) # Converting to R dataframe
#     edgeR_QLF = ro.globalenv['edgeR_QLF'] # Finding the R function in the script
#     return edgeR_QLF(df_r, outfile="", design="paired", overwrite=overwrite, filter_expr=filter_expr, meta_only=True)
