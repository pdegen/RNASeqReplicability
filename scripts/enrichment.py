import numpy as np
import pandas as pd
import gseapy
from pathlib import Path
import os
import logging
import json
import time
import argparse
import pickle
from datetime import datetime
import time
import biomart
import rpy2.robjects as ro
from misc import pickler, Timer, open_table
from process import signal_to_noise, get_init_samples_from_cohort
from DEA import normalize_counts


def get_biomart_data():
    # Set up connection to server                                               
    server = biomart.BiomartServer('http://uswest.ensembl.org/biomart')
    mart = server.datasets['hsapiens_gene_ensembl']

    # List the types of data we want                                            
    attributes = ['ensembl_gene_id', 'hgnc_symbol', 'entrezgene_id']

    # Get the mapping between the attributes                                    
    response = mart.search({'attributes': attributes})
    data = response.raw.data.decode('ascii')
    return data


def get_ensembl_mappings(data):
    ensg, sym, entr = [], [], []
    for i, line in enumerate(data.splitlines()):
        line = line.split('\t')
        # The entries are in the same order as in the `attributes` variable
        ensg.append(line[0])
        sym.append(line[1])
        entr.append(line[2])

    # conv = pd.DataFrame(np.array([sym]).T, index=ensg, columns=["Symbol"])
    conv = pd.DataFrame(np.array([sym, entr]).T, index=ensg, columns=["Symbol", "Entrez"])

    conv.index.name = "ENSG"
    return conv


def convert_ids_biomart(queries):
    """
    queries: pd.index of ENSG symbols
    """

    datafile = "../data/biomart/data.txt"
    if not Path(datafile).is_file():
        logging.info("Loading biomart data...")
        data = get_biomart_data()
        pickler(data, datafile)
    else:
        with open(datafile, "rb") as f:
            data = pickle.load(f)

    conv = get_ensembl_mappings(data)
    common = queries.intersection(conv.index)
    bm_results = pd.DataFrame(conv.loc[common], index=conv.loc[common].index)
    # bm_results.rename(columns={"Symbol":"external_gene_name"}, inplace=True)
    return bm_results


def convert_ensg(tab, conv_table, target="Entrez"):
    """target can be either 'Entrez' or 'Symbol'
    drops rows that don't have target ID"""
    common = tab.index.intersection(conv_table.index)
    tab = tab.loc[common]
    tab["ID"] = conv_table.loc[common, target]
    tab = tab.set_index("ID")
    notna = tab.index.notnull()
    logging.info(f"Dropped {np.sum(~notna)} missing gene IDs out of {len(tab)}")
    tab = tab[notna]
    if target == "Entrez": tab.index = tab.index.astype(int)
    return tab


def find_conv_table_all(datasets, savepath, overwrite=False):
    """Create conversion table from ENSG to gene symbols for all genes in all data sets"""

    if not overwrite and Path(savepath).is_file():
        logging.info("Existing conversion table not overwritten")
        return

    ensgs = set()
    for data in datasets:
        t = pd.read_csv(datasets[data]["datapath"], index_col=0, usecols=["Unnamed: 0"])
        ensgs = ensgs.union(set(t.index))
    bm_results = convert_ids_biomart(ensgs)

    logging.info(f"Unique ENSG identifiers: {len(ensgs)}")
    dupes = bm_results.index.duplicated(keep='first')
    logging.info(f"Duplicated symbols: {len(dupes)}")
    bm_results = bm_results[~dupes]
    notna = bm_results[bm_results["Symbol"] != ""]
    logging.info(f"Unique existing symbols: {len(notna)}")
    notna.to_csv(savepath)


def clean_tab(tab, bm_results):
    bm_results = bm_results[~bm_results.index.duplicated(keep='first')]
    notna = bm_results[bm_results["Symbol"] != ""]
    tab_cleaned = tab.loc[notna.index]
    tab_cleaned["Symbol"] = bm_results.loc[notna.index, "Symbol"]
    tab_cleaned = tab_cleaned.set_index(['Symbol'], append=True)
    return tab_cleaned


def prepare_gsea(tab):
    """
    Drop duplicate gene symbols, missing gene symbols, add and sort by logFC
    tab: pd.DataFrame edgeR table
    """

    prepared_tab = tab["logFC"]
    nameless = tab[tab.index.get_level_values('Symbol').isin([np.NaN])].index
    prepared_tab = prepared_tab.drop(nameless)
    print(f"Dropped {len(nameless)} nameless genes")
    prepared_tab = prepared_tab.reset_index(drop=False)

    # Drop duplicates, keep biggest abs(logFC)
    prepared_tab["abs_logFC"] = np.abs(prepared_tab["logFC"])
    prepared_tab.sort_values(by="abs_logFC", ascending=False, inplace=True)
    dupes = prepared_tab["Symbol"].duplicated(keep="first")
    prepared_tab = prepared_tab.loc[~dupes]
    prepared_tab = prepared_tab.drop("abs_logFC", axis=1)
    prepared_tab.sort_values(by="logFC", ascending=False, inplace=True)

    return prepared_tab


def run_gseapy(prepared_tab, library, outpath, threads=4, permutation_num=100, file_id="", ranking="logFC", min_size=15,
               max_size=500):
    """
    prepared_tab: pandas.DataFrame, output of prepare_gsea(tab)
    geneset: str, name of Enrichr library geneset
    """
    pre_res = gseapy.prerank(rnk=prepared_tab[["Symbol", ranking]], gene_sets=library,
                             threads=threads,
                             permutation_num=permutation_num,  # reduce number to speed up testing
                             outdir=outpath, seed=6, no_plot=True, min_size=min_size, max_size=max_size)

    terms = pd.read_csv(f"{outpath}/gseapy.gene_set.prerank.report.csv")
    terms.loc[:, 'Term ID'] = terms["Term"].str.split(" \(", expand=True)[1].str[:-1]
    terms.loc[:, 'Term'] = terms["Term"].str.split(" \(", expand=True)[0]
    terms = terms[["Name", "Term", "Term ID", "ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val", "Tag %", "Gene %",
                   "Lead_genes"]]
    terms = terms.rename(columns={"FDR q-val": "FDR"})
    if library.startswith("KEGG"):
        terms.drop("Term ID", axis=1, inplace=True)

    fname = f"gseapy.{ranking}.{library}.{file_id}.feather" if file_id != "" else f"{outpath}/gseapy.{ranking}.{library}.feather"
    terms.to_feather(
        f"{outpath}/{fname}")
    os.system(f"rm {outpath}/gseapy.gene_set.prerank.report.csv")
    print(f"Significant (5% FDR): {len(terms[terms['FDR']<0.05])} out of {len(terms)}")
    
    return pre_res


def run_gseapy_libraries(prepared_tab, gseapath, libraries, overwrite_all_gsea, file_id="", permutation_num=100,
                         save_full_results=False, threads=4, ranking="logFC", min_size=15, max_size=500):
    
    for library in libraries:
        gsea_out = f"{gseapath}/gseapy.{ranking}.{library}.{file_id}.feather" if file_id != "" else f"{outpath}/gseapy.{ranking}.{library}.feather"
        if not Path(gsea_out).is_file() or overwrite_all_gsea:
            logging.info(library)
            gsea_results = run_gseapy(prepared_tab, library, gseapath, permutation_num=permutation_num,
                                      file_id=file_id, threads=threads, ranking=ranking, min_size=min_size,
                                      max_size=max_size)
            if save_full_results:
                pickler(gsea_results, gsea_out.split(".feather")[0] + ".txt")
        else:
            logging.info(f"Existing file not overwritten: {gsea_out}")


def run_clusterORA(degs, universe, file_id, go_ont, prefix="", overwrite=False, use_internal_data=False,
                   internal_data_path="", minGSSize=15, maxGSSize=500, **unused):
    """
    Wrapper for R function that runs clsuterProfiler over-representation analysis for GO BP and KEGG
    """

    # This function should be called from a notebook in the notebooks folder, we need to load a script from the scripts folder
    wd = Path(str(Path(os.getcwd()).parent) + "/scripts/R_functions.r")
    ro.r['source'](str(wd))  # Loading the R script

    clusterORA = ro.globalenv['clusterORA']  # Finding the R function in the script
    clusterORA(degs, universe, file_id, go_ont, prefix=prefix, overwrite=overwrite, minGSSize=minGSSize,
               maxGSSize=maxGSSize,
               use_internal_data=use_internal_data, internal_data_path=internal_data_path)


def run_gsea(tab, gseapath, libraries, gsea_method, overwrite=False, file_id="", **gsea_kwargs):
    """
    Wrapper for all enrichment methods
    """
    if gsea_method.startswith("gseapy"):
        logging.info(f"\nCalling GSEApy with kwargs:\n{gsea_kwargs}\n")
        run_gseapy_libraries(tab, gseapath, libraries, overwrite_all_gsea=overwrite, file_id=file_id, **gsea_kwargs)

    elif gsea_method.startswith("clusterORA"):
        logging.info(f"\nCalling clusterProfiler ORA with kwargs:\n{gsea_kwargs}\n")

        # find degs
        use_internal_data = gsea_kwargs["use_internal_data"]
        internal_data_path = gsea_kwargs["internal_data_path"]
        go_ont = gsea_kwargs["go_ont"]
        for fdr in gsea_kwargs["FDRs"]:
            for logFC in gsea_kwargs["logFCs"]:
                degs = tab[(tab["FDR"] < fdr) & (tab["logFC"].abs() > logFC)]
                degs = list(degs.index)
                universe = list(tab.index)
                prefix = f"{gseapath}/{gsea_method}.fdr{fdr}.post_lfc{logFC}.lfc{logFC}"
                # run clusterProfiler
                if len(degs) > 5:
                    run_clusterORA(degs, universe, file_id=file_id, prefix=prefix, overwrite=overwrite, **gsea_kwargs)
                else:
                    logging.info(f"No DEGs found for FDR<{fdr}, |logFC| > {logFC}")
                    # for ease of processing, still save results as ORA results regular dataframe, but set pval = 1 for all terms
                    outfile_go = f"{prefix}.GO_Biological_Process_2021{file_id}.feather"
                    outfile_kegg = f"{prefix}.KEGG_2021_Human{file_id}.feather"
                    os.system(f"cp ../data/clusterORA/empty_kegg.feather {outfile_kegg}")
                    os.system(f"cp ../data/clusterORA/empty_go.feather {outfile_go}")

    else:
        raise Exception(f"{gsea_method} not implemented!")


@Timer(name="decorator")
def main_enrich(config, DEA_method, outlier_method, gsea_method, gsea_param_set, conv_file=""):
    starttime = datetime.now()

    #### Loading, preparing 

    # Load the config_params dict
    logging.info(f"Loading config params; starttime = {starttime}")
    with open(config, "r") as f:
        j = json.load(f)
        cohort = j["Cohort"]
        config_params = j["config_params"][gsea_param_set]
        gsea_kwargs = config_params["gsea_kwargs"][gsea_method]
        outpath = config_params["outpath"]
        libraries = config_params["libraries"]
        overwrite = config_params["overwrite"]
        dea_param_set = config_params["dea_param_set"]
        dea_param_set_lfc = config_params["dea_param_set_lfc"]
        rankings = config_params["rankings"]

    for ranking in rankings:

        config_params["gsea_kwargs"][gsea_method]["ranking"] = ranking
        
        # Load gene conversion table
        if conv_file == "": conv_file = "../data/multi/conv_table.csv"
        logging.info(f"Start enrichment with {ranking} ranking; timedelta = {datetime.now() - starttime}")
        conv_table = pd.read_csv(conv_file, index_col=0)
    
        # Load DEA results table
        logging.info(f"Loading DEA table; timedelta = {datetime.now() - starttime}")
        if gsea_method == "clusterORA_lfc":
            tab = open_table(f"{outpath}/tab.{outlier_method}.{DEA_method}.{dea_param_set_lfc}")
        else:
            tab = open_table(f"{outpath}/tab.{outlier_method}.{DEA_method}.{dea_param_set}")
    
        # Calculate |S2N|
        if config_params["gsea_kwargs"][gsea_method]["ranking"] == "|S2N|":
            samples_i = get_init_samples_from_cohort(config.replace("/gsea", ""))
            counts = pd.read_csv(config_params["data"], usecols=["Unnamed: 0"] + samples_i, index_col=0)
            counts = normalize_counts(counts)
            tab.loc[counts.index, "|S2N|"] = signal_to_noise(counts)
    
        if tab.index[0].startswith("ENSG"):
            tab_cleaned = tab.loc[tab.index.intersection(conv_table.index)]
            tab_cleaned["Symbol"] = conv_table.loc[tab_cleaned.index,"Symbol"]
        else: # assume symbol
            tab_cleaned = tab.loc[tab.index.intersection(conv_table.set_index("Symbol").index)]
            tab_cleaned["Symbol"] = conv_table.set_index("Symbol").loc[tab_cleaned.index].index
    
        logging.info(f"Original tab: {len(tab)} genes\nCleaned tab: {len(tab_cleaned)} genes\n")
    
        # Convert ENSG to Entrez
        if gsea_method.startswith("clusterORA"):
            logging.info(f"Converting to Entrez; timedelta = {datetime.now() - starttime}")
            tab_cleaned = convert_ensg(tab_cleaned, conv_table, target="Entrez")
    
        #### GSEA
    
        # create temporary folder to store intermediate results (avoid conflict with parallel GSEA jobs)
        gseapath = f"{outpath}/gsea"
        tmppath = f"{gseapath}/tmp_{str(time.time()).replace('.', '')}"
        os.system(f"mkdir {tmppath}")
        file_id = f"{DEA_method}.{outlier_method}.{gsea_param_set}"
        logging.info(f"Starting GSEA; timedelta = {datetime.now() - starttime}")
        run_gsea(tab_cleaned, tmppath, libraries, gsea_method, overwrite=overwrite, file_id=file_id, **gsea_kwargs)
        logging.info(f"Finished GSEA, moving files; timedelta = {datetime.now() - starttime}")
        print("tmp", tmppath)
        os.system(f"ls {tmppath}")
        os.system(f"mv {tmppath}/*.feather {tmppath}/..")
        os.system(f"rm -r {tmppath}")
        finishtime = datetime.now()
        logging.info(f"Done at {finishtime}; timedelta = {finishtime - starttime}")


if __name__ == "__main__":
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger()
    # logger.addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--DEA_method')
    parser.add_argument('--outlier_method')
    parser.add_argument('--gsea_method')
    parser.add_argument('--gsea_param_set')
    args = parser.parse_args()
    main_enrich(**vars(args))
