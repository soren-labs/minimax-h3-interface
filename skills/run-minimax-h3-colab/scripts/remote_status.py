from __future__ import annotations

import json
import os
from pathlib import Path


config = json.loads(Path("/content/h3_job.json").read_text(encoding="utf-8"))
job_id = str(config["job_id"])
pid_file = Path(f"/content/h3-job-{job_id}.pid")
log = Path(f"/content/h3-job-{job_id}.log")
offset_file = Path(f"/content/h3-job-{job_id}.monitor-offset")
archive = Path(f"/content/h3-results-{job_id}.tar.gz")
pid = int(pid_file.read_text(encoding="utf-8")) if pid_file.exists() else None
running = False
if pid is not None:
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        running = not (stat.exists() and stat.read_text().split()[2] == "Z")
    except ProcessLookupError:
        pass
size = log.stat().st_size if log.exists() else 0
offset = int(offset_file.read_text()) if offset_file.exists() else 0
if offset < 0 or offset > size:
    offset = 0
if log.exists():
    with log.open("rb") as handle:
        handle.seek(offset)
        new_log = handle.read().decode("utf-8", errors="replace")
        handle.seek(max(0, size - 65536))
        tail = handle.read().decode("utf-8", errors="replace")
else:
    new_log = ""
    tail = ""
offset_file.write_text(str(size), encoding="utf-8")
success = "REMOTE_STAGE_COMPLETE script=/content/h3_remote_job.py" in tail
print(json.dumps({
    "job_id": job_id,
    "pid": pid,
    "running": running,
    "success": success,
    "archive": str(archive),
    "archive_exists": archive.is_file(),
    "log_bytes": size,
    "new_log": new_log,
}, ensure_ascii=False))
