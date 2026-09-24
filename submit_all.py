from pathlib import Path
import re
import subprocess

import pandas as pd


ACCOUNT = "def-hadi87"

JOBS_FILE = Path("jobs.csv")
LOG_DIR = Path("logs")

LOG_DIR.mkdir(
    exist_ok=True,
)

jobs = pd.read_csv(JOBS_FILE)


def safe_name(text):
    """
    Make dataset/model names safe for Slurm job names and log filenames.
    """
    text = str(text)

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    )


submitted = []
failed = []


for row_index, job in jobs.iterrows():

    dataset = job["dataset"]
    model = job["model"]

    profile = job["profile"]
    cpus = int(job["cpus"])
    mem_gb = int(job["mem_gb"])
    walltime = job["time"]

    safe_dataset = safe_name(dataset)
    safe_model = safe_name(model)

    job_name = f"{safe_dataset}__{safe_model}"

    print()
    print("=" * 80)
    print(f"Job index : {row_index}")
    print(f"Dataset   : {dataset}")
    print(f"Model     : {model}")
    print(f"Profile   : {profile}")
    print(f"CPUs      : {cpus}")
    print(f"RAM       : {mem_gb} GB")
    print(f"Time      : {walltime}")
    print("=" * 80)

    command = [
        "sbatch",

        f"--account={ACCOUNT}",

        f"--job-name={job_name}",

        f"--cpus-per-task={cpus}",

        f"--mem={mem_gb}G",

        f"--time={walltime}",

        # Narval A100
        "--gpus-per-node=a100:1",

        (
            "--output="
            f"logs/{job_name}_%j.out"
        ),

        (
            "--error="
            f"logs/{job_name}_%j.err"
        ),

        (
            "--export="
            f"ALL,"
            f"JOB_INDEX={row_index},"
            f"JOBS_FILE={JOBS_FILE.resolve()}"
        ),

        "./run_job.sh",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:

        message = result.stdout.strip()

        print(message)

        # Usually:
        # "Submitted batch job 3712345"
        slurm_job_id = (
            message.split()[-1]
            if message
            else None
        )

        submitted.append({
            "row_index": row_index,
            "dataset": dataset,
            "model": model,
            "slurm_job_id": slurm_job_id,
        })

    else:

        print("SUBMISSION FAILED")

        print("STDOUT:")
        print(result.stdout)

        print("STDERR:")
        print(result.stderr)

        failed.append({
            "row_index": row_index,
            "dataset": dataset,
            "model": model,
            "error": result.stderr,
        })


# --------------------------------------------------
# Save submission records
# --------------------------------------------------

if submitted:

    submitted_df = pd.DataFrame(submitted)

    submitted_df.to_csv(
        "submitted_jobs.csv",
        index=False,
    )

    print()
    print(
        f"Successfully submitted "
        f"{len(submitted_df)} jobs."
    )


if failed:

    failed_df = pd.DataFrame(failed)

    failed_df.to_csv(
        "failed_submissions.csv",
        index=False,
    )

    print(
        f"{len(failed_df)} submissions failed."
    )