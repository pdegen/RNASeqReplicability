#!/bin/bash
#SBATCH --time=00:03:00    # Each task takes max 03 minutes
#SBATCH --mem-per-cpu=4G  # Each task uses max 4G of memory

date
echo "Starting job with 04 minutes to go"

source ../../welcome.sh
conda activate rna-rep

n=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
j=$SLURM_ARRAY_JOB_ID

outpath_cohort="$1/$2_$n"
config="$outpath_cohort/gsea/config.json"

echo "Outpath cohort:" $outpath_cohort 
echo "DEA:" $3
echo "Outlier method:" $4
echo "GSEA method:" $5
echo "Parameter set:" $6

mkdir -p $outpath_cohort/gsea/slurm

srun python3 ../scripts/enrichment.py --config $config --DEA_method $3 --outlier_method $4 --gsea_method $5 --gsea_param_set $6

mv slurm-${j}_${SLURM_ARRAY_TASK_ID}.out $outpath_cohort/gsea/slurm/slurm-${j}_${SLURM_ARRAY_TASK_ID}.$4.$3.$5.$6.out