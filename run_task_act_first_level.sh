#!/usr/bin/bash
#SBATCH --partition {PARTITION_HERE}
#SBATCH --ntasks 1
#SBATCH --time 01-00:00:00
#SBATCH --mem-per-cpu 20G
#SBATCH --cpus-per-task 10
#SBATCH --out "slurm-%j.out"

module load miniconda
module load AFNI
conda activate functional_brain_states # Just an example, change to your environment name

subj="A0001" # Just an example, change to your subj(s) or use job arrays

# Motion
motion_regressors_file=""
censor_file=""

# Stims
stims_file=""

# Scan
scan_file=""

# Output directory and file prefix
out_dir=""
out_file_pre=""

# Number of parallel CPU cores
n_cores=""

# PARAMETERS ABOVE ARE REQUIRED
# BELOW OPTIONAL
# MAKE SURE YOU EDIT THE VARIABLES AND PYTHON COMMAND BELOW FOR OPTIONAL ARGUMENTS)

# Template for extracting activation
template_name=""
template_file=""


python3 task_act_first_level.py \
    --scan_path "${scan_file}" \
    --task_timing_path "${stims_file}" \
    --motion_path "${motion_regressors_file}" \
    --censor_path "${censor_file}" \
    --cond_labels "Neg2Back,Neg0Back" \
    --out_dir "${out_dir}" \
    --out_file_pre "${out_file_pre}" \
    --num_cores "${n_cores}" \
    --out_cond_labels "Neg2Back,Neg0Back,Neg2Back_Neg0Back" \
    --out_stat "t" \
    --contrast_labels "Neg2Back+Neg0Back" \
    --contrast_functions "1*Neg2Back-1*Neg0Back" \
    --extract \
    --template_name "${template_name}" \
    --template_path "${template_file}" \
    --remove_previous 
