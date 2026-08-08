from __future__ import annotations

import runpy
from pathlib import Path


for script in (Path("/content/h3_remote_cuda13.py"), Path("/content/h3_remote_job.py")):
    print(f"REMOTE_STAGE_START script={script}", flush=True)
    runpy.run_path(str(script), run_name="__main__")
    print(f"REMOTE_STAGE_COMPLETE script={script}", flush=True)

