#!/bin/bash
#SBATCH --time=00:03:00    # Each task takes max 03 minutes
#SBATCH --mem-per-cpu=2G   # Each task uses max 2G of memory

n=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
j=$SLURM_ARRAY_JOB_ID

outpath_cohort="$1/$2_$n"
config="$outpath_cohort/config.json"

echo "Outpath cohort:" $outpath_cohort 
echo "DEA:" $3
echo "Outlier method:" $4
echo "Parameter set:" $5
echo "Sampler:" $6

mkdir -p $outpath_cohort/slurm

srun python3 ../scripts/main.py --config $config --DEA_method $3 --outlier_method $4 --param_set $5 --sampler $6

mv slurm-${j}_${SLURM_ARRAY_TASK_ID}.out $outpath_cohort/slurm/slurm-${j}_${SLURM_ARRAY_TASK_ID}.$4.$3.$5.out