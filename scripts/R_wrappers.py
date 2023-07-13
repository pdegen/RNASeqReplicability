import sys
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from pathlib import Path

def filterByExpr_wrapper(inpath, outpath, design):
    """Filter low-expressed genes using edgeR's filterByExpr()
    
    Result will be saved as a csv file in outpath
    """
    #if not isinstance(df, ro.vectors.DataFrame): df = pd_to_R(df)

    ro.r['source']('../scripts/R_functions.r') # Loading the R script
    R_filterByExpr = ro.globalenv['edgeR_filterByExpression'] # Finding the R function in the script
    R_filterByExpr(inpath, outpath, design)