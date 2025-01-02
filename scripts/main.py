import os
import glob
import logging
import json
import argparse
import random
import pandas as pd
from outliers import get_outlier_patients
from misc import Timer, replicate_sampler, get_matching_treatment_col_ix
from DEA import run_dea
from process import get_init_samples_from_cohort

def subsample(datapath, outpath, replicates, sampler, seed=None, verbose=False, return_df=True):

    #### Get the df for this cohort

    configfile = f"{outpath}/config.json"
    samples_i = get_init_samples_from_cohort(configfile)

    if len(samples_i) == 2 * replicates:
        if verbose:
            logging.info(f"Using existing subsample: {samples_i}")
        if not return_df: return
        df_full = pd.read_csv(datapath, index_col=0)
        df_sub = df_full[samples_i]
    else:
        df_full = pd.read_csv(datapath, index_col=0)
        if seed:
            random.seed(seed)#random.seed(cohort)
        if sampler == "paired":
            df_sub, _ = replicate_sampler(df_full, replicates, ispaired=True)
        elif sampler == "unpaired":
            df_sub, _ = replicate_sampler(df_full, replicates, ispaired=False)
        else:
            raise Exception(f"Invalid sampler: {sampler}")
            
        if verbose: logging.info(f"Using new subsample: {df_sub.columns.to_list()}")
        with open(f"{outpath}/config.json", "r+") as f:
            configdict = json.load(f)
            configdict["samples_i"] = df_sub.columns.to_list()
            f.seek(0)
            json.dump(configdict, f)
            f.truncate()

    if return_df:
        return df_sub

@Timer(name="decorator")
def main(config, DEA_method, outlier_method, param_set, sampler="paired"):
    """    
    Parameters
    ----------
    config: path to config params file
    DEA_method: str, DEA methods, e.g. "edgeR" or "DESeq2"
    outlier_method: str, outlier detection method, e.g. "jk" or "pcah"
    param_set: int, parameter set to use in config file
    """

    #### Load the config_params dict

    with open(config, "r") as f:
        j = json.load(f)
        cohort = j["Cohort"]
        config_params = j["config_params"][param_set]
        
    DEA_kwargs = config_params["DEA_kwargs"][DEA_method]
    outlier_kwargs = config_params["outlier_kwargs"][outlier_method]
    datapath = config_params["data"]
    outpath = config_params["outpath"]
    outname = config_params["outname"]
    replicates = config_params["replicates"]

    df_sub = subsample(datapath, outpath, replicates, sampler, verbose=True)

    #### Construct covariate df to control for covariates

    if DEA_kwargs["design"] == "custom":
        logging.info(f"Constructing covariate df from file")
        metadata = pd.read_csv(datapath.replace(".csv", ".meta.csv"), index_col=0)
        
        meta_sub = metadata.loc[df_sub.columns]
        covariate_file = f"{outpath}/covariates.csv"
        meta_sub.to_csv(covariate_file)
        DEA_kwargs["design"] = covariate_file

    #### Find the outliers and remove from cohort

    if DEA_method == "edgerqlf" or outlier_method == "none":
        outliers = get_outlier_patients(df_sub, outlier_method, outname, outpath, **outlier_kwargs)
    else:
        outlier_kwargs["param_set"] = param_set
        outlier_kwargs["original_outlier_method"] = outlier_method
        outliers = get_outlier_patients(df_sub, "existing", outname, outpath, **outlier_kwargs)
    logging.info(f"Outliers_f: {outliers}")

    # If final cohort has less than 2 patients, we deem it so heterogeneous that outlier removal is meaningless (rare)
    if len(df_sub.columns) - len(outliers) < 4:
        logging.info(f"N={replicates} with {len(outliers) // 2} outliers: setting outliers=[]")
        outliers = []

    df_sub = df_sub.drop(outliers, axis=1)

    #### Run DEA on post-outlier removed cohort
    
    outfile = f"{outpath}/tab.{outlier_method}.{DEA_method}.{param_set}.feather"
    run_dea(df_sub, outfile, method=DEA_method, overwrite=config_params["overwrite"], **DEA_kwargs)


if __name__ == "__main__":
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--DEA_method')
    parser.add_argument('--outlier_method')
    parser.add_argument('--param_set')
    parser.add_argument('--sampler')
    args = parser.parse_args()
    main(**vars(args))
	