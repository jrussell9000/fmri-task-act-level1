TASK-BASED fMRI ACTIVATION ANALYSIS
1. Creates task design matrix (including denoising and censoring)
2. First-level regression with 3dDeconvolve
3. (Optional) Creates condition/contrast-specific files (optional) with 3dcalc
4. (Optional) Extracts activation using a provided template with 3dROIstats

Author: Taylor J. Keding, Ph.D.
Version: 1.0
Last updated: 11/24/25

REQUIREMENTS:
- AFNI
- numpy
- pandas

INPUTS:
(required)
-scan_path: global file path to the run-concatenated dense time series (preprocessed UP TO first-level analyses)
    format = .nii (volumetric)
--task_timing_path: global file path to the run-concatenated stimulus timing file (should have been stripped for dummy scans)
    format = .csv; column 1 name = 'CONDITION', column 2 name = 'ONSET' (local times matching --scan_path scan length), column 3 name = 'DURATION' (optional; assuming DUR=1 s if not included)
--motion_path: global file path to the run-concatenated motion regressors (expect 6, not including derivatives for now; should have been stripped for dummy scans)
    format = .txt (tab-delimited) with no headers; rows = TR/frame, columns = 6 motion directions
--censor_path: global file path to the run-concatenated frame motion/outlier censor file (should have been stripped for dummy scans)
    format = .txt (tab-delimited) with no headers; rows= TR/frame, single column = binary (1=include,0=exclude)
--cond_labels: list (comma-separated) of task conditions to generate beta series for - labels should match the 'CONDITION' column from the timing file
    format = string-list of conditions e.g. 'stimFear,stimSad,stimNeu'
--out_dir: global directory path for derivative intermediates and task-condition beta series; will create if doesn't exist
--out_file_pre: prefix to be prepended to the files stored in out_dir (should usually contain an ID, task name, time point, etc. but NOT the global file path; eg. 'subj001_nback_baseline')
--num_cores: number of CPU cores available for parallel processing (int)

(optional)
--remove_previous: (no value) include this option if "out_dir" should have all files removed before processing starts (including connectivity); if not included, will not overwrite pre-existing files
--out_cond_labels:
--out_stat:
--contrast_labels:
--contrast_functions:
--extract:
--template_name:
--template_path: template file for the option specified in --calc_conn; if 'seed_to_voxel', should be a binary mask with a single ROI; if 'parcellated', should be a set of ROIs (at least 2 required, each labeled with an integer>0)
    format = .nii (volumetric and registered to the same atlas as task beta series, unless --force_diff_atlas used; if there is a mismatch in grid spacing, will resample the template to the beta series grid)
--force_diff_atlas: (no value) include this option if you know the template and scan are in different standard spaces, but want to force connectivity analyses anyway (USE WITH CAUTION)
--average_type:

OUTPUTS: