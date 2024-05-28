# RNASeqReplicability

[![DOI](https://zenodo.org/badge/665607063.svg)](https://zenodo.org/badge/latestdoi/665607063)

Code and data for Degen &amp; Medo (2023).

Calculations were performed on UBELIX ([http://www.id.unibe.ch/hpc](http://www.id.unibe.ch/hpc)), the HPC cluster at the University of Bern.

## Instructions

To download the TCGA raw data from the GDC, use [notebooks/gdc_api.ipynb](https://github.com/pdegen/RNASeqReplicability/blob/main/notebooks/gdc_api.ipynb) and specify the desired primary cancer site. The notebook will save a count matrix of shape g x (N + T) = (number of genes) x (number of normal tissue samples + number of matching tumor tissue samples) as a CSV file in the data folder. Additionally, a metadata CSV file with clinical variables for each patient will be saved.

To subsample the data and perform differential expression analysis on the subsampled cohorts, we recommend the use of an HPC environment. Please contact the author for detailed instructions if you wish to reproduce this analysis.
