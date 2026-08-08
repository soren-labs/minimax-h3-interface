from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


config = json.loads(Path("/content/h3_job.json").read_text(encoding="utf-8"))
job_id = str(config["job_id"])
log = Path(f"/content/h3-job-{job_id}.log")
pid_file = Path(f"/content/h3-job-{job_id}.pid")
for existing_pid_file in Path("/content").glob("h3-job-*.pid"):
    try:
        existing_pid = int(existing_pid_file.read_text(encoding="utf-8"))
        os.kill(existing_pid, 0)
        stat = Path(f"/proc/{existing_pid}/stat")
        if not (stat.exists() and stat.read_text().split()[2] == "Z"):
            raise RuntimeError(
                f"an H3 job is already running: pid={existing_pid} state={existing_pid_file}"
            )
    except ProcessLookupError:
        continue
with log.open("wb") as handle:
    process = subprocess.Popen(
        [sys.executable, "-u", "/content/h3_remote_entry.py"],
        cwd="/content",
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
pid_file.write_text(str(process.pid), encoding="utf-8")
print(f"BACKGROUND_STARTED job_id={job_id} pid={process.pid} log={log}")
