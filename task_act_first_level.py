#!/usr/bin/env python3

# ============================================================================
# TASK-BASED fMRI ACTIVATION ANALYSIS
# 1. Creates task design matrix (including denoising and censoring)
# 2. First-level regression with 3dDeconvolve
# 3. (Optional) Creates condition/contrast-specific files (optional) with 3dcalc
# 4. (Optional) Extracts activation using a provided template with 3dROIstats
#
# Author: Taylor J. Keding, Ph.D.
# Version: 1.0
# Last updated: 11/24/25
# ============================================================================
'''
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
'''

# Imports
import os
import shutil
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import time
import re
from io import StringIO

def dir_path_exists(path_string):
    if not os.path.isdir(path_string):
        os.makedirs(path_string, exist_ok=True)
    return path_string

def file_path_exists(path_string):
    if not os.path.isfile(path_string):
        raise argparse.ArgumentTypeError(f"[ERROR] The file '{path_string}' does not exist or is not a valid file.")
    return path_string

def valid_task_conds(conds_string):
    try:
        conds = conds_string.split(',')
        conds_s = [cond.strip() for cond in conds]
        for cond in conds_s:
            if cond is None or cond == "":
                raise argparse.ArgumentError(f"[ERROR] Use list-string notation (e.g. 'cond1label,cond2label...') for --cond_labels and --contrast_labels.")
        return conds_s
    except:
        raise argparse.ArgumentError(f"[ERROR] Use list-string notation (e.g. 'cond1label,cond2label...') for --cond_labels.")

def valid_stat_type(stat_string):
    valid_types = ['z', 't', 'r']
    if stat_string not in valid_types:
        raise argparse.ArgumentError(f"[ERROR] --stat_type must be 'z', 't', or 'r'")

def valid_ave_type(ave_string):
    if ave_string not in ["mean", "median"]:
        raise argparse.ArgumentTypeError(f"[ERROR] '{ave_string} must be one of ['mean', 'median'].")
    return ave_string

def valid_contrast_functions(contrast_list, contrast_labs_list, cond_list, out_list):

    # Check everything in out_list is in cond_list or contrast_labs_list
    if out_list is not None:
        for cond in out_list:
            if cond not in contrast_labs_list and cond not in cond_list:
                print("[ERROR] All output effects in --out_cond_labels must be in either --cond_labels or --contrast_labels.")
                sys.exit()

    # Check that contrast_list and contrast_labs_list are same size
    if len(contrast_list) != len(contrast_labs_list):
        print("[ERROR] Every contrast in --constrast_functions must have a label in --contrast_labels")
        sys.exit()

    # Assumes a stricly linear function (e.g. 1*COND1-1*COND2 [simple subtraction], 0.5*COND1+0.5*COND2 [mean of two conditions])
    try:
        out = []
        oper_chars = r"[+-]"
        for contrast in contrast_list:
            coefs = []
            conds = []
            opers = [char for char in contrast if (char == "+" or char == "-")]
            contrast_split = re.split(oper_chars, contrast)
            for term in contrast_split:
                term_split = term.split("*")
                coefs.append(term_split[0])
                conds.append(term_split[1])

            # Check all contrast conditions exist in cond_list
            for item in conds:
                if item not in cond_list:
                    print(f"[ERROR] {contrast} contains a condition not in --cond_labels")
                    sys.exit()

            # Check that all coefficients can be cast to float
            for item in coefs:
                try:
                    float(item)
                except:
                    print(f"[ERROR] {contrast} contains invalid coefficients.")
                    sys.exit()

            out.append({'COEFS': coefs, 'CONDS': conds, 'OPERATORS': opers})
        return out
    except:
        print(f"[ERROR] At least one contrast in --contrast_functions was improperly formatted.")
        print(f"        Assumes a stricly linear function (e.g. 1*COND1-1*COND2)")
        sys.exit()

def validate_template(scan_string, template_string, out_string, force_diff_atlas):
    try:
        template_info_out = subprocess.run(['3dinfo', f"{template_string}"], capture_output=True, text=True, check=True)
    except:
        print(f"[ERROR] Could not open --template_path {template_string} with AFNI's 3dinfo - please check your template.")
        sys.exit()

    template_info_out = template_info_out.stdout
    template_info_out = template_info_out.split()

    upper = None
    for i, entry in enumerate(template_info_out):
        if entry == "to":
            upper = template_info_out[i + 1]
            break

    atlas = None
    for i, entry in enumerate(template_info_out):
        if entry == "Space:":
            atlas = template_info_out[i + 1]
            break

    orient = None
    for i, entry in enumerate(template_info_out):
        if entry == "[-orient":
            orient = template_info_out[i + 1]
            orient = orient.replace("]", "")
            break

    spacing = None
    for i, entry in enumerate(template_info_out):
        if entry == "-step-":
            spacing = template_info_out[i + 1]
            break

    if upper is None or orient is None or spacing is None or atlas is None:
        print(f"[ERROR] Could not get info from --template_path {template_string} with AFNI's 3dinfo - please check your template.")
        sys.exit()

    # Check if template is binary vs. ROIs and compare to conn_string
    try:
        int_upper = int(upper)
    except:
        print(f"[ERROR] Template provided to --template_path contained floats, and should only contains ints (ROI set or binary mask) - please check your template.")
        sys.exit()
    if int_upper == 0:
        print(f"[ERROR] Template provided to --template_path contained no data - please check your template.")
        sys.exit()

    # If everything else is valid with the template, check for matching atlas space, orientation, and grid spacing
    # If orientation or grid spacing do not match, resample the template to the raw data
    try:
        scan_info_out = subprocess.run(['3dinfo', f"{scan_string}"], capture_output=True, text=True, check=True)
    except:
        print(f"[ERROR] Could not open --scan_path {scan_string} with AFNI's 3dinfo - please check your scan.")
        sys.exit()

    scan_info_out = scan_info_out.stdout
    scan_info_out = scan_info_out.split()

    atlas_scan = None
    for i, entry in enumerate(scan_info_out):
        if entry == "Space:":
            atlas_scan = scan_info_out[i + 1]
            break

    orient_scan = None
    for i, entry in enumerate(scan_info_out):
        if entry == "[-orient":
            orient_scan = scan_info_out[i + 1]
            orient_scan = orient_scan.replace("]", "")
            break

    spacing_scan = None
    for i, entry in enumerate(scan_info_out):
        if entry == "-step-":
            spacing_scan = scan_info_out[i + 1]
            break
    print(f"[INFO] Specs for scan {scan_string} and template {template_string}")
    print(f"       - orient template: {orient}")
    print(f"       - orient scan: {orient_scan}")
    print(f"       - atlas template: {atlas}")
    print(f"       - atlas scan: {atlas_scan}")
    print(f"       - spacing template: {spacing}")
    print(f"       - spacing scan: {spacing_scan}")

    if orient_scan is None or spacing_scan is None or atlas_scan is None:
        print(f"[ERROR] Could not get info from --scan_path {scan_string} with AFNI's 3dinfo - please check your scan.")
        sys.exit()

    if atlas != atlas_scan:
        if force_diff_atlas:
            print(f"[WARNING] {scan_string} and {template_string} are in different standard template spaces, but proceeding anyway with --force_diff_atlas")
        else:
            print(f"[ERROR] Atlas space mismatch between --scan_path {scan_string} and --template_path {template_string} - scan and template must be registered to the same atlas.")
            sys.exit()

    if orient != orient_scan or spacing != spacing_scan:
        print(f"[WARNING] Orientation and/or grid spacing for --scan_path {scan_string} and --template_path {template_string} do not match.")
        print(f"[INFO] Resampling --template_path {template_string} to match --scan_path {scan_string}.")
        template_split = template_string.split("/")
        template_file = template_split[(len(template_split) - 1)]
        template_file_pre = template_file.replace(".nii.gz", "")
        template_file_pre = template_file_pre.replace(".nii", "")
        if not os.path.exists(f"{out_string}/{template_file_pre}_resampled.nii.gz"):
            resamp_command = ["3dresample", "-master", f"{scan_string}", "-input", f"{template_string}", "-prefix", f"{out_string}/{template_file_pre}_resampled.nii.gz"]
            subprocess.run(resamp_command)
            if os.path.exists(f"{out_string}/{template_file_pre}_resampled.nii.gz"):
                print(f"[INFO] Sucessfully resampled template; saved to {out_string}/{template_file_pre}_resampled.nii.gz")
                return f"{out_string}/{template_file_pre}_resampled.nii.gz"
            else:
                print(f"[ERROR] Could not resample --template_path {template_string}. Please manually resample and try again.")
                sys.exit()
        else:
            print(f"[INFO] Resampled template {out_string}/{template_file_pre}_resampled.nii.gz already exists.")
            return f"{out_string}/{template_file_pre}_resampled.nii.gz"
    else:
        return f"{template_string}"

def remove_files_from_dir(dir_string):
    if not os.path.isdir(dir_string):
        print(f"Error: '{dir_string}' does not yet exist. --remove previous will be ignored.")
    else:
        for item_name in os.listdir(dir_string):
            item_path = os.path.join(dir_string, item_name)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except OSError as e:
                print(f"Error deleting '{item_path}': {e}")

def get_stim_data(args):
    stim_df = pd.read_csv(args.task_timing_path, sep=',')
    for curr_col in ["CONDITION", "ONSET", "DURATION"]:
        if curr_col not in stim_df.columns.to_list():
            print(f"[ERROR] --task_timing_path file did not contain necessary columns 'CONDITION', 'ONSET', and 'DURATION'")
            sys.exit()
    sorted_df = stim_df.sort_values(by='ONSET', ascending=True)

    # Check conditions to generate activation for exist in the timing file
    checks = [True if cond in np.unique(sorted_df['CONDITION']) else False for cond in args.cond_labels]
    if not all(checks):
        print(f"[ERROR] At least one task condition in --cond_labels does not exist in the timing file.")
        sys.exit()

    # Create individual (run-concatenated) timing files
    for cond in np.unique(sorted_df['CONDITION']):
        cond_df = sorted_df[sorted_df['CONDITION'] == cond]
        if cond in args.cond_labels:
            if os.path.exists(f"{args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt"):
                print(f"[INFO] {args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt already exists.")
            else:
                cond_df['ONSET'].to_csv(f"{args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt", header=False, index=False, sep='\n')
                print(f"[INFO] Created {args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt")

    return sorted_df.reset_index()

def run_first_level(stim_data, args):

    if not os.path.exists(f"{args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz"):
        # Set-up base command, including temporal detrending, frame censors, and motion regressors
        decon_command = ["3dDeconvolve", "-quiet",
                         "-input", f"{args.scan_path}",
                         "-polort", "A",
                         "-censor", f"{args.censor_path}",
                         "-ortvec", f"{args.motion_path}", "motion_regressors",
                         "-num_stimts", f"{len(args.cond_labels)}"]

        # Iteratively add stim timing for conditions NOT included in beta series
        for i, cond in enumerate(args.cond_labels):
            mean_dur = np.mean(stim_data[stim_data['CONDITION'] == cond]['DURATION'])
            decon_command.extend(["-stim_times", f"{i + 1}", f"{args.out_dir}/{args.out_file_pre}_concat_{cond}_onsets.txt",
                                 f"GAM(8.6,.547,{mean_dur})", "-stim_label", f"{i + 1}", f"{cond}"])

        # Iteratively add contrasts
        for i, cont in enumerate(args.contrast_functions):
            to_extend = ["-gltsym"]
            sym_eq = "SYM:"
            for i in range(len(cont["COEFS"])):
                sym_eq = f"{sym_eq} {cont["COEFS"][i]}*{cont["CONDS"][i]}"
                if i != (len(cont['COEFS']) - 1):
                    sym_eq = f"{sym_eq} {cont["OPERATORS"][i]}"
            to_extend.append(sym_eq)
            to_extend.append("-glt_label")
            to_extend.append(f"{i + 1}")
            to_extend.append(f"{args.contrast_labels[i]}")
            decon_command.extend(to_extend)

        # Add statistics output
        decon_command.extend(["-fout", "-rout", "-tout",
                              "-bucket", f"{args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz",
                              "-jobs", f"{args.num_cores}"])

        # Run command and be sure we have output
        subprocess.run(decon_command)
        if os.path.exists(f"{args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz"):
            print(f"[INFO] Sucessfully created first level activation stats for {args.out_file_pre}")
        else:
            print(f"[ERROR] Failed to create first level activation stats for {args.out_file_pre}")
            sys.exit()
    else:
        print(f"[INFO] Design matrix for {args.out_file_pre} already exists.")
    if os.path.exists("3dDeconvolve.err"):
        os.remove("3dDeconvolve.err")

def save_out_effects(args):

    # Go through --out_cond_labels, get the corresponding subbrik from the bucket stats file
    # Save each to its own file
    try:
        bucket_info = subprocess.run(['3dinfo', '-subbrick_info', f"{args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz"], capture_output=True, text=True, check=True)
    except:
        print(f"[ERROR] Could not open {args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz with AFNI's 3dinfo - check your first-level outputs.")
        sys.exit()

    bucket_split = bucket_info.stdout.split()
    for curr_cond in args.out_cond_labels:
        if not os.path.exists(f"{args.out_dir}/{args.out_file_pre}_act_{curr_cond}_{args.out_stat}.nii.gz"):
            stat_string = f"{args.out_stat.upper()}stat"
            out_sub_brik_indx = None
            for i, substr in enumerate(bucket_split):
                if curr_cond in substr and stat_string in substr:
                    try:
                        out_sub_brik_indx = int(bucket_split[i - 1][1:])
                        break
                    except:
                        print(f"[ERROR] Could not find appropriate bucket subbrik for condition {curr_cond} in {args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz")
                        sys.exit()
            if out_sub_brik_indx is None:
                print(f"[ERROR] Could not find appropriate bucket subbrik for condition {curr_cond} in {args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz")
                sys.exit()
            else:
                calc_cmd = ["3dcalc", "-a", f"{args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz'[{out_sub_brik_indx}]'",
                            "-expr", "a", "-prefix", f"{args.out_dir}/{args.out_file_pre}_act_{curr_cond}_{args.out_stat}.nii.gz"]
                subprocess.run(calc_cmd)
                if not os.path.exists(f"{args.out_dir}/{args.out_file_pre}_act_{curr_cond}_{args.out_stat}.nii.gz"):
                    print(f"[ERROR] Could not create {args.out_file_pre}_act_{curr_cond}_{args.out_stat}.nii.gz from {args.out_file_pre}_act_bucket_stats.nii.gz")
                    sys.exit()
        else:
            print(f"[INFO] {args.out_file_pre}_act_{curr_cond}_{args.out_stat}.nii.gz already exists (skipping).")

def extract_effects(args):

    for curr_cond in args.out_cond_labels:
        if not os.path.exists(f"{args.out_dir}/{args.out_file_pre}_act_{curr_cond}_{args.out_stat}_{args.template_name}.csv"):
            try:
                extract_cmd = ["3dROIstats", f"-nz{args.average_type}", "-mask", f"{args.template_path}", f"{args.out_dir}/{args.out_file_pre}_act_{curr_cond}_{args.out_stat}.nii.gz"]
                extract_out = subprocess.run(extract_cmd, capture_output=True, text=True, check=True).stdout
                out_df = pd.read_csv(StringIO(extract_out), sep='\t')
            except:
                print(f"[ERROR] Could not extract {curr_cond} from {args.out_dir}/{args.out_file_pre}_act_bucket_stats.nii.gz with AFNI's 3dROIstats.")
                sys.exit()

            if args.out_stat == "median":
                ave_type_flag = "NZMed"
            else:
                ave_type_flag = "NZMean"

            cols_to_keep = [colname for colname in out_df.columns if ave_type_flag in colname]
            new_df = out_df[cols_to_keep]
            new_df.rename(columns={oldname: f"ROI_{str(oldname).split(sep="_")[1]}" for oldname in new_df.columns}, inplace=True)
            new_df.rename(index={0: f"{args.out_stat}"}, inplace=True)

            new_df.to_csv(f"{args.out_dir}/{args.out_file_pre}_act_{curr_cond}_{args.out_stat}_{args.template_name}.csv")
            if os.path.exists(f"{args.out_dir}/{args.out_file_pre}_act_{curr_cond}_{args.out_stat}_{args.template_name}.csv"):
                print(f"[INFO] Successfully extracted {curr_cond} using ROI template {args.template_name}.")
            else:
                print(f"[ERROR] Could not extract {curr_cond} using ROI template {args.template_name}.")
                sys.exit()
        else:
            print(f"[INFO] {args.out_dir}/{args.out_file_pre}_act_{curr_cond}_{args.out_stat}_{args.template_name}.csv already exists (skipping).")

def main():

    # Start the timer
    start_time = time.time()

    # ---------------------------------
    # Parse arguments
    # ---------------------------------
    parser = argparse.ArgumentParser()
    # required:
    parser.add_argument("--scan_path", type=file_path_exists, required=True)
    parser.add_argument("--task_timing_path", type=file_path_exists, required=True)
    parser.add_argument("--motion_path", type=file_path_exists, required=True)
    parser.add_argument("--censor_path", type=file_path_exists, required=True)
    parser.add_argument("--cond_labels", type=valid_task_conds, required=True)
    parser.add_argument("--out_dir", type=dir_path_exists, required=True)
    parser.add_argument("--out_file_pre", type=str, required=True)
    parser.add_argument("--num_cores", type=int, required=True)

    # optional:
    parser.add_argument("--remove_previous", action='store_true', required=False)
    parser.add_argument("--out_cond_labels", type=valid_task_conds, required=False)
    parser.add_argument("--out_stat", type=valid_stat_type, required=False)
    parser.add_argument("--contrast_labels", type=valid_task_conds, required=False)
    parser.add_argument("--contrast_functions", type=valid_task_conds, required=False)
    parser.add_argument("--extract", action='store_true', required=False)
    parser.add_argument("--template_name", type=str, required=False)
    parser.add_argument("--template_path", type=file_path_exists, required=False)
    parser.add_argument("--force_diff_atlas", action='store_true', required=False)
    parser.add_argument("--average_type", action='valid_ave_type', required=False)

    args = parser.parse_args()
    if args.force_diff_atlas is None:
        args.force_diff_atlas = False

    # First check that if user provided contrast_functions that they are appropriately formatted
    if args.contrast_labels is not None:
        if args.contrast_functions is None:
            print(f"[ERROR] --contrast_functions must exist if --contrast_labels exist.")
            sys.exit()
    else:
        if args.contrast_functions is not None:
            print(f"[ERROR] --contrast_labels must exist if --contrast_functions exist.")
            sys.exit()
    args.contrast_functions = valid_contrast_functions(args.contrast_functions, args.contrast_labels, args.cond_labels, args.out_cond_labels)

    # Check for out_stat if out_cond_labels exists
    if args.out_cond_labels is not None:
        if args.out_stat is None:
            print(f"[WARNING] --out_cond_labels was provided, but --out_stat was not. Defaulting to z-score.")
            args.out_stat = "z"

    print(f"[INFO] Activation maps will be created for the following task effects: {args.out_cond_labels}")
    print(f"       All other task conditions will be controlled for, but won't be saved individually.")

    # Make sure optional args exist if --extract is provided
    if args.extract:

        # Checks for extraction
        if args.template_name is None:
            parser.error("[ERROR] --template_name must be provided if --extract is present.")
            sys.exit()
        if args.template_path is None:
            parser.error(f"[ERROR] --extract requires an ROI set (.nii) be provided to --template_path.")
            sys.exit()
        else:
            # Check for valid template
            args.template_path = validate_template(args.scan_path, args.template_path, args.out_dir, args.force_diff_atlas)
        print("[INFO] Extracting parcellated (ROI-based) activation matrices after generating voxelwise maps.")
        # Check if average type provided
        if args.average_type is None:
            print("[INFO] No average_type provided for ROI extraction; defaulting to 'mean'.")
        else:
            print(f"[INFO] Using {args.average_type} for calculating average ROI activation.")
    else:
        print("[INFO] Activation will not be extracted after generating voxelwise maps.")

    # If --remove_previous is provided, clear-out the output dir
    if args.remove_previous:
        print(f"[INFO] Removing all files from previous analyses in {args.out_dir} (if they exist).")
        remove_files_from_dir(args.out_dir)

    # ---------------------------------

    # Get stimulus timing data and save single-column .txt file with beta conds stim timing {out_dir}/{our_file_pre}_concat_betas_onsets.txt
    # Also save separate timing files (same format) for conditions to be controlled for {out_dir}/{our_file_pre}_concat_{cond}_onsets.txt
    stim_data = get_stim_data(args)

    # Run activation first-level analysis
    run_first_level(stim_data, args)

    # Save specified output effects to their own file
    if args.out_cond_labels is not None:
        save_out_effects(args)

    # Run extraction if indicated
    if args.extract:
        extract_effects(args)

    # Display total runtime
    print(f"Total runtime: {time.time() - start_time}")


if __name__ == "__main__":
    main()
