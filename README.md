# RNASeqReplicability

[![DOI](https://zenodo.org/badge/665607063.svg)](https://zenodo.org/badge/latestdoi/665607063)

Code and data for Degen &amp; Medo (2025).

## Instructions

To be expanded... This repo still needs some cleaning up

### Bootstrapping

We have a separate repository for reseacrhers who wish to perform our suggested bootstrapping procedure on their own data: 

[https://github.com/pdegen/BootstrapSeq](https://github.com/pdegen/BootstrapSeq)

### Reproducing figures

Aggregated, processed data to reproduce the figures can be found in [data/multi](./data/multi). Most of the figures are created using [notebooks/figures_revised.ipynb](notebooks/figures_revised.ipynb).

### Downloading raw data

To download the TCGA raw data from the GDC, use [notebooks/gdc_api.ipynb](https://github.com/pdegen/RNASeqReplicability/blob/main/notebooks/gdc_api.ipynb) and specify the desired primary cancer site. The notebook will save a count matrix of shape g x (N + T) = (number of genes) x (number of normal tissue samples + number of matching tumor tissue samples) as a CSV file in the data folder. Additionally, a metadata CSV file with clinical variables for each patient will be saved. The patient IDs can also be retrieved from the metadata files in data/`site`/`site`_meta.csv, where `site` is in {breast, colorectal, liver, lung, lung2, kidney, prostate, thyroid}.

To subsample the data and perform differential expression analysis on the subsampled cohorts, we recommend using an HPC environment with SLURM. Please contact the author ([@pdegen](https://github.com/pdegen)) for detailed instructions to reproduce this analysis.

## Acknowledgments

Calculations were performed on UBELIX ([http://www.id.unibe.ch/hpc](http://www.id.unibe.ch/hpc)), the HPC cluster at the University of Bern.

