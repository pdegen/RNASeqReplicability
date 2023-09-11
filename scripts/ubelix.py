import os
import logging
import glob
import json
import time
from pathlib import Path
from main import main


def adjust_job_time(N, DEA_method, outlier_method, script_path) -> int:
    """Adjust the maximum job time specified in the shell script for sending batch jobs to Ubelix"""

    time_str = "#SBATCH --time="

    with open(script_path, "r+") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):

            if line.startswith(time_str):

                if outlier_method == "jk":
                    max_time = 0.37 * N + 0.053 * N ** 2
                elif outlier_method == "pcah":
                    max_time = N / 3
                elif outlier_method == "none":
                    max_time = 1 + N / 8
                else:
                    raise Exception(f"Outlier method {outlier_method} not implemented")

                max_time = int(max(max_time, 1))

                if DEA_method != "edgerqlf": max_time = 2 + int(N / 8)

                lines[i] = f"#SBATCH --time=00:{max_time:02.0f}:00    # Each task takes max {max_time:02.0f} minutes\n"

                break

        f.truncate(0)  # truncates the file
        f.seek(0)  # moves the pointer to the start of the file
        f.writelines(lines)  # write the new data to the file

    return max_time


def adjust_gsea_job_time(N, libraries, gsea_method, gsea_script_path, just_testing=False) -> int:
    """Adjust the maximum job time specified in the shell script for sending batch jobs to Ubelix"""

    time_str = "#SBATCH --time="

    if gsea_method.startswith("gseapy"):
        max_time = 6
    elif gsea_method.startswith("clusterORA"):
        max_time = 4 + N // 7
    else:
        raise Exception(f"Enrichment method {gsea_method} not implemented")

    max_time = int(max(max_time, 1))

    if just_testing: return max_time

    with open(gsea_script_path, "r+") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):

            if line.startswith(time_str):

                lines[i] = f"#SBATCH --time=00:{max_time:02.0f}:00    # Each task takes max {max_time:02.0f} minutes\n"

            elif line.startswith("Starting job with"):
                lines[i] = f"Starting job with {max_time:02.0f} minutes to go"
                break

        f.truncate(0)  # truncates the file
        f.seek(0)  # moves the pointer to the start of the file
        f.writelines(lines)  # write the new data to the file

    return max_time


def all_outliers_found(outpath_N, outlier_method, n_cohorts, param_set) -> bool:
    """
    Check if outliers have already been detected for all cohorts
    """
    cohorts = sorted([f.name for f in os.scandir(outpath_N) if f.is_dir()])[:n_cohorts]
    for cohort in cohorts:
        slurmfile = f"{outpath_N}/{cohort}/slurm/slurm-*.{outlier_method}.edgerqlf.{param_set}.out"

        slurmfile = sorted(glob.glob(slurmfile), reverse=True)
        if len(slurmfile) < 1:
            logging.info(f"Outliers not found: {cohort}")
            return False
        slurmfile = slurmfile[0]
        with open(slurmfile, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("Outliers_f: "):
                # outliers = line.split("Outliers_f: ")[-1][:-1]
                break
        else:
            logging.info(f"Outliers not found: {cohort}")
            return False
    return True


def run_multi_batch(config_params, all_N, n_cohorts, script_path, mode="check n_jobs"):
    outname_original = config_params["outname"]
    outpath_original = config_params["outpath"]
    param_set = config_params['param_set']
    total_jobs = 0

    for N in all_N:

        outname_N = outname_original + "_N" + str(N)
        outpath_N = outpath_original + "/" + outname_N

        for DEA in config_params["DEA_methods"]:

            for out in config_params["outlier_methods"]:

                job_ids = []

                if DEA != "edgerqlf" and out != "none":

                    # check existing outliers
                    outliers_found = all_outliers_found(outpath_N, out, n_cohorts, param_set)
                    if not outliers_found:
                        logging.info("Finish edgerqlf jobs first")
                        continue

                        # Check if results table already exists, if not: append cohort to job_ids
                for cohort in range(1, n_cohorts + 1):

                    outname_c = outname_N + f"_{int(cohort):04}"
                    outpath_c = Path(outpath_N + "/" + outname_c)
                    config_params_c = config_params.copy()
                    config_params_c["replicates"] = N
                    config_params_c["outpath"] = str(outpath_c)
                    config_params_c["outname"] = outname_c

                    if outpath_c.exists():
                        if not (Path(f"{outpath_c}/tab.{out}.{DEA}.{param_set}.csv").exists() or Path(
                                f"{outpath_c}/tab.{out}.{DEA}.{param_set}.feather").exists()):
                            job_ids.append(str(cohort))
                            # Update the config file
                            with open(f"{outpath_c}/config.json", "r+") as f:
                                configdict = json.load(f)
                                configdict["config_params"][param_set] = config_params_c
                                f.seek(0)
                                json.dump(configdict, f)
                                f.truncate()

                    else:
                        os.system(f"mkdir -p {outpath_c}")
                        job_ids.append(str(cohort))
                        # Create the config file
                        with open(f"{outpath_c}/config.json", "w") as f:
                            configdict = {
                                "Cohort": cohort,
                                "config_params": {param_set: config_params_c}
                            }
                            configjson = json.dumps(configdict)
                            f.write(configjson)

                # Send the jobs
                max_time = adjust_job_time(N, DEA_method=DEA, outlier_method=out, script_path=script_path)
                logging.info(
                    f"\n{'Sending' if mode != 'just testing' else 'Testing'} {len(job_ids)} jobs for N={N}, DEA={DEA}, out={out}, max time: {max_time} min")
                total_jobs += len(job_ids)
                if len(job_ids) > 0:

                    if mode in ["send jobs", "just testing"]:
                        command = f"sbatch --array={','.join(job_ids)} {script_path} {outpath_N} {outname_N} {DEA} {out} {param_set}"
                        logging.info(command)

                        if mode == "send jobs":
                            os.system(command)
                        elif mode == "just testing":
                            logging.info("Just testing...")

                    elif mode in ["test main", "test main terminal"]:
                        from main import main
                        for cohort in job_ids:
                            outname_c = outname_N + f"_{int(cohort):04}"
                            outpath_c = Path(outpath_N + "/" + outname_c)
                            config_params_file = f"{outpath_c}/config.json"
                            if mode == "test main":
                                main(config_params_file, DEA, out, param_set)
                            elif mode == "test main terminal":
                                command = f"python3 ../scripts/main.py --config {config_params_file} --DEA_method {DEA} --outlier_method {out} --param_set {param_set}"
                                logging.info(command)
                                os.system(command)

    logging.info(f"Total jobs: {total_jobs}")


def run_gsea_batch(config_params, all_N, n_cohorts, libraries, gsea_script_path, mode="just testing", sleep_seconds=0):
    outname_original = config_params["outname"]
    outpath_original = config_params["outpath"]
    dea_param_set = config_params["dea_param_set"]
    gsea_param_set = config_params['gsea_param_set']
    total_jobs = 0
    overwrite = config_params["overwrite"]

    for N in all_N:

        outname_N = outname_original + "_N" + str(N)
        outpath_N = outpath_original + "/" + outname_N

        for DEA in config_params["DEA_methods"]:

            for out in config_params["outlier_methods"]:

                for gsea_method in config_params["gsea_methods"]:

                    job_ids = []

                    # Check if gsea results table already exists, if not: append cohort to job_ids
                    for cohort in range(1, n_cohorts + 1):

                        outname_c = outname_N + f"_{int(cohort):04}"
                        outpath_c = Path(outpath_N + "/" + outname_c)
                        tabfile_c = f"{outpath_c}/tab.{out}.{DEA}.{dea_param_set}"
                        config_params_c = config_params.copy()
                        config_params_c["replicates"] = N
                        config_params_c["outpath"] = str(outpath_c)
                        config_params_c["outname"] = outname_c

                        # Check if the DEA results table exists
                        if Path(f"{tabfile_c}.csv").is_file() or Path(f"{tabfile_c}.feather").is_file():

                            gseapath = Path(f"{outpath_c}/gsea")

                            if gseapath.exists():

                                for library in libraries:

                                    ## annoying workaround
                                    if "clusterORA" in gsea_method:
                                        g = glob.glob(
                                            f"{outpath_c}/gsea/{gsea_method}*.{library}.{DEA}.{out}.{gsea_param_set}.feather")
                                    else:
                                        g = glob.glob(
                                            f"{outpath_c}/gsea/{gsea_method}.{library}.{DEA}.{out}.{gsea_param_set}.feather")

                                    if len(g) >= 1 and gsea_method in ["gseapy", "gseapy_s2n"]:
                                        gsea_results_exist = True
                                    elif len(g) >= 1 and gsea_method in ["clusterORA", "clusterORA_lfc"]:
                                        gsea_results_exist = True
                                    else:
                                        gsea_results_exist = False

                                    if (not gsea_results_exist) or overwrite:
                                        # Update the config file
                                        with open(f"{outpath_c}/gsea/config.json", "r+") as f:
                                            configdict = json.load(f)
                                            configdict["config_params"][gsea_param_set] = config_params_c
                                            f.seek(0)
                                            json.dump(configdict, f)
                                            f.truncate()
                                        job_ids.append(str(cohort))
                                        break

                            else:
                                os.system(f"mkdir -p {outpath_c}/gsea")
                                job_ids.append(str(cohort))
                                # Create the config file
                                with open(f"{outpath_c}/gsea/config.json", "w") as f:
                                    configdict = {
                                        "Cohort": cohort,
                                        "config_params": {gsea_param_set: config_params_c}
                                    }
                                    configjson = json.dumps(configdict)
                                    f.write(configjson)


                        else:
                            logging.info(f"DEA table {tabfile_c} doesn't exist, no jobs sent")

                    # Send the jobs
                    max_time = adjust_gsea_job_time(N=N, libraries=libraries, gsea_method=gsea_method,
                                                    gsea_script_path=gsea_script_path)
                    logging.info(
                        f"\n{'Sending' if mode != 'just testing' else 'Testing'} {len(job_ids)} jobs for N={N}, DEA={DEA}, out={out}, GSEA={gsea_method}, max time: {max_time} min")
                    total_jobs += len(job_ids)
                    if len(job_ids) > 0:

                        if mode in ["send jobs", "just testing"]:
                            command = f"sbatch --array={','.join(job_ids)} {gsea_script_path} {outpath_N} {outname_N} {DEA} {out} {gsea_method} {gsea_param_set}"
                            logging.info(command)

                            if mode == "send jobs":
                                os.system(command)
                                if sleep_seconds > 0:
                                    logging.info(f"Sleeping {sleep_seconds} seconds...")
                                    time.sleep(sleep_seconds)
                            elif mode == "just testing":
                                logging.info("Just testing...")

                        elif mode in ["test main", "test main terminal"]:
                            from enrichment import main_enrich
                            for cohort in job_ids:
                                outname_c = outname_N + f"_{int(cohort):04}"
                                outpath_c = Path(outpath_N + "/" + outname_c)
                                config_params_file = f"{outpath_c}/gsea/config.json"
                                if mode == "test main":
                                    main_enrich(config_params_file, DEA, out, gsea_method, gsea_param_set, conv_file="")
                                elif mode == "test main terminal":
                                    command = f"python3 ../scripts/enrichment.py --config {config_params_file} --DEA_method {DEA} --outlier_method {out} --gsea_method {gsea_method} --param_set {gsea_param_set}"
                                    logging.info(command)
                                    os.system(command)

    logging.info(f"Total jobs: {total_jobs}")
