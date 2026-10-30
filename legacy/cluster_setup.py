"""Archived experimental Dask/SLURM launcher; not part of supported SMAC runs.

This retains the original site-specific cluster settings for reference.
Importing this file does not launch workers. Calling start_cluster can submit
Slurm jobs; the launcher has not been validated for current deployments.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path


def start_cluster(n_workers: int, processes_per_worker: int, use_slurm: bool):
    """Archived experiment: start local workers or submit site-specific Slurm jobs.

    Kept for reference only. The supported optimisation does not call this.
    """
    from distributed import Client

    if not use_slurm:
        client = Client(n_workers=n_workers, threads_per_worker=processes_per_worker)  # LocalCluster default
        print("[cluster] Local Dask cluster started:", client)
        return client

    from dask_jobqueue import SLURMCluster

    work_root = Path(os.environ.get("SMAC_WORK_ROOT", ".")).resolve()
    log_dir   = work_root / "dask-logs"
    tmp_dir   = work_root / "dask-tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Bind scheduler to a host/interface visible to workers on the compute node
    # (when driver runs under SLURM, hostname is the compute node)
    scheduler_host = socket.gethostname()

    cluster = SLURMCluster(
        queue="normal",               # partition
        processes=1,
        cores=processes_per_worker,   # cpu per worker process
        memory="10GB",
        walltime="02:00:00",
        local_directory=str(tmp_dir),
        job_extra_directives=[
            "--account=oz061",
            f"--output={log_dir}/dask.%x.%j.out",
            f"--error={log_dir}/dask.%x.%j.err",
            "--ntasks=1",
            f"--cpus-per-task={processes_per_worker}",
            "--mem=10G",
            "--time=02:00:00",
        ],
        job_script_prologue=[
            "set -euo pipefail",
            "module --force purge",
            "module load python-scientific/3.11.5-foss-2023b",
            "source /home/kandrych/venvs/obriy311/bin/activate",
            "export OMP_NUM_THREADS=1",
            'echo "PYTHON = $(which python)"',
        ],
        python=sys.executable,
        scheduler_options={
            "host": scheduler_host,        # avoid binding to an external/unroutable IP
            "dashboard_address": ":0",     # disable dashboard port hard-coding
        },
    )

    # Persist the exact worker script we’re submitting
    (log_dir / "dask_worker_template.sh").write_text(cluster.job_script())

    cluster.scale(n_workers)
    client = Client(cluster)
    client.wait_for_workers(n_workers, timeout="180s")
    print("[cluster] up:", client)
    
    return client
