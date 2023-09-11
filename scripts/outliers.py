import sys
import os
import logging
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from R_wrappers import pd_to_R
from misc import get_matching_treatment_col_ix


def get_outlier_patients(df, method, outname="", outpath="", **kwargs) -> list:
    """
    Returns list of outlier patients (control and treatment column names)
    """

    # Individual outlier methods must return names of outlier control samples only
    if method == "none":
        return []
    elif method == "jk":
        outliers = iterative_jackknife(df, outname=outname, path=outpath, DEA="edgerqlf", **kwargs)
    elif method == "pcah":
        outliers = pca_hubert_wrapper(df, control_cols_only=True, **kwargs)
    elif method == "existing":

        slurmfile = f"{outpath}/slurm/slurm-*.{kwargs['original_outlier_method']}.edgerqlf.{kwargs['param_set']}.out"
        slurmfile = sorted(glob.glob(slurmfile), reverse=True)[0]

        with open(slurmfile, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("Outliers_f: "):
                outliers = line.split("Outliers_f: ")[-1][2:-3]
                outliers = outliers.split("', '")
                outliers = outliers[:len(outliers) // 2]
                break

    else:
        raise Exception(f"Outlier method {method} not implemented")

    if len(outliers) < 1: return []

    outliers_ix = get_matching_treatment_col_ix(df, outliers)
    return df.columns[outliers_ix].tolist()


###################
#### Jackknife ####
###################

def n_DEG_jack_merged(tab, FDR=0.01) -> (np.array, np.array, int):
    """
    Given merged df of DEG tables from jackknifing, calculate number of DE genes (<FDR) for each table
    Column 0 can include the results from the full data with no samples removed (col name must be "0")
    The remaining col names correspond to the sample names that were tentatively removed through jackknifing
    
    Returns
    -------
    ind: np.array of sample names that were tentatively removed through jackknifing, same order as n_DEG
    n_DEG: np.array of number of DEGs for each sample, descending order
    unjacked: int, number of DEGs from full data with no sample removed
    """

    n_DEG = []
    ind = []

    for c in tab.groupby(level=0, axis=1):
        ind.append(c[0])
        t = c[1]
        t = t.xs("FDR", level=1, axis=1)
        n = len(t[t < FDR].dropna())
        n_DEG.append(n)

    if ind[0] == "0":
        ind.pop(0)
        unjacked = n_DEG.pop(0)
    else:
        logging.info("No unjacked data found")
        unjacked = None

    ind = np.array(ind)
    n_DEG = np.array(n_DEG)

    s = sorted(zip(range(len(n_DEG)), n_DEG), key=lambda x: -x[1])
    ind = ind[[k[0] for k in s]]
    n_DEG = n_DEG[[k[0] for k in s]]

    return ind, n_DEG, unjacked


def jackknife_wrapper(df, outname, path, DEA="edgerqlf", FDR=0.01, overwrite=False, include_full=True,
                      skip_cols="skip_none", **DEA_kwargs) -> int:
    """
    Return exist staus:
    0: jackknife ran succesfully
    1: existing tables found, not overwritten
    """

    # This function should be called from a notebook in the notebooks folder, we need to load a script from the scripts folder
    wd = Path(str(Path(os.getcwd()).parent) + "/scripts/R_functions.r")
    ro.r['source'](str(wd))  # Loading the R script

    df_r = df if isinstance(df, ro.vectors.DataFrame) else pd_to_R(df)  # Converting to R dataframe

    jackknife_paired = ro.globalenv['jackknife_paired']  # Finding the R function in the script
    cols_to_keep = DEA_kwargs["cols_to_keep"] if "cols_to_keep" in DEA_kwargs else "all"
    if list(DEA_kwargs.keys()).remove("cols_to_keep") != None: raise exception(
        "DEA_kwargs beyond cols_to_keep not yet implemented for jackknife")
    return \
    jackknife_paired(df_r, outname, path, overwrite=overwrite, include_full=include_full, DEA=DEA, skip_cols=skip_cols,
                     cols_to_keep=cols_to_keep)[0]


def jackknife_merger(path, name, cleanup=True) -> pd.DataFrame:
    """
    Parameters
    ----------
    path: str, path to folder containing list of jacknifed tables to be merged into single df
    """
    all_files = glob.glob(path + f"/{name}*_table.csv")
    li = []
    names = []

    for i, filename in enumerate(all_files):
        df = pd.read_csv(filename, index_col=0, header=0)
        li.append(df)
        n = filename.split("/")[-1].split("_")[-2]
        names.append(n)

    df = pd.concat(li, axis=1, keys=names).sort_index()
    df = df[sorted(df.columns)]

    savepath = f"{path}/{name}_jacked_merged.csv"
    df.to_csv(savepath)
    # logging.info(f"Merged df saved in {savepath}")

    if cleanup:
        os.system(f"rm {path}/{name}*_table.csv")
        # logging.info("Removed merged tables")
    return df


def iterative_jackknife(df, outname, path, DEA="edgerqlf", FDR=0.01, overwrite=False, max_removed_frac=1,
                        efficient=False,
                        tolerance=1, cleanup=True, **DEA_kwargs) -> list:
    """
    Parameters
    ----------
    max_removed_frac: float in [0,1], maximum number of removable outliers as fraction of total sample pairs (patients)
    efficient: bool, if True, after 0th iteration skip those patients least likely to be outliers 
               i.e. 100*max_removed_frac% of patients with least DEG increase in 0th iteration
    cleanup: bool, if True, remove intermediate tables
    
    Returns
    -------
    outliers_f: list of outlier sample names (control samples only)
    """
    outliers = [0]
    skip_cols = "skip_none"
    i = 0
    total_patients = len(df.columns) // 2
    total_outliers = 0
    max_removable_patients = int(max_removed_frac * total_patients)
    samples_i = list(df.columns)[:len(df.columns) // 2]
    logging.info(
        f"\nStarting jackknife with:\nSamples_i {samples_i}\nMax removable: {max_removable_patients} | FDR < {FDR} | Cleanup {cleanup}\n")  # initial samples

    if cleanup:
        # create temporary folder to store intermediate results (avoid conflict with parallel jackknife jobs)
        path = f"{path}/tmp_{str(time.time()).replace('.', '')}"
        os.system(f"mkdir {path}")
    else:
        logging.warning(f"No cleanup for jackknife, make sure no conflict occurs with parallel jobs")

    while len(outliers) > 0 and total_outliers < max_removable_patients:

        name = outname + "_i" + str(i)  # Jackknife iteration

        if i >= 1 and max_removed_frac < 1 and efficient:
            if i == 1:
                skip_cols = ind[max_removable_patients:].tolist()
                logging.info(f"Skipping {skip_cols} for efficiency")
            elif total_patients > 12 and len(ind) > 1:
                pass
                # skip_cols += ind[-1].tolist()
                # logging.info(f"Skipping {skip_cols} for efficiency")

        # Run jackknife
        exit_status = jackknife_wrapper(df, outname=name, path=path, DEA=DEA, FDR=FDR, overwrite=overwrite,
                                        skip_cols=skip_cols, **DEA_kwargs)
        if exit_status == 1: return []

        # Merge jackknife tables
        jackknife_merger(path=path, name=name, cleanup=cleanup)

        # Get number of DEGs for each table
        tab = pd.read_csv(f"{path}/{name}_jacked_merged.csv", index_col=0, header=[0, 1])
        ind, n_DEG, unjacked = n_DEG_jack_merged(tab, FDR=FDR)
        logging.info(f"i{i}: Samples/#DEGs: {ind} {n_DEG}")

        # Flag as outliers those samples where the number of DEGs > limit
        limit = tolerance * (unjacked - min(n_DEG)) + min(n_DEG)
        logging.info(f"All samples #DEGs: {unjacked} | Outlier threshold: {limit}")
        outliers_ind = np.where(n_DEG > limit)[0]
        total_outliers += len(outliers_ind)

        if total_outliers > max_removable_patients:
            # See if we can still remove only the most extreme outliers remaining before reaching max removable patients
            n_final_outliers = len(df.columns) // 2 - total_patients + max_removable_patients
            if n_final_outliers > 0:
                outliers_ind = outliers_ind[:n_final_outliers]
                outliers = ind[outliers_ind]
                outlier_cols = [o for o in outliers] + ["T" + o[1:] for o in outliers]
                df = df.drop(outlier_cols, axis=1)
                logging.info(f"Outliers: {outliers} (final before reaching max removable)")
            logging.info("Max removed patients reached, breaking")
            break

        outliers = ind[outliers_ind]
        outlier_cols = [o for o in outliers] + ["T" + o[1:] for o in outliers]
        logging.info(f"Outliers: {outliers}\n")
        df = df.drop(outlier_cols, axis=1)
        i += 1

    samples_f = list(df.columns)[:len(df.columns) // 2]
    outliers_f = sorted(list(set(samples_i).difference(set(samples_f))))
    logging.info(f"Samples_f {samples_f}")  # final samples

    if cleanup:
        p = path + "/*jacked_merged.csv"
        os.system(f"rm -r {path}")  # delete temporary folder

    return outliers_f


####################
#### PCA Hubert ####
####################

def pca_hubert_wrapper(df, k=2, plot=False, control_cols_only=True) -> list:
    """
    Returns
    -------
    outliers: list of outlier samples (control and/or treatment)
    control_cols_only: if true, outlying treatment samples will be returned as their matching control samples, 
                       regardless of whether the respective control samples are outliers
    """

    # This function should be called from a notebook in the notebooks folder, we need to load a script from the scripts folder
    wd = Path(str(Path(os.getcwd()).parent) + "/scripts/R_functions.r")
    ro.r['source'](str(wd))  # Loading the R script

    pcahubert = ro.globalenv['pcahubert']  # Finding the R function in the script
    df_r = pd_to_R(df)
    outliers = list(pcahubert(df_r, k=k, plot=plot))
    outliers = df.iloc[:, [o - 1 for o in outliers]].columns

    if control_cols_only:
        ix = [df.columns.get_loc(c) for c in outliers]
        N = len(df.columns) // 2
        ix = [c if c + N < 2 * N else c - N for c in ix]
        outliers = list(set(df.columns[ix].tolist()))

    return outliers
