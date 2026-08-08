#!/usr/bin/env python3
"""One-command, fail-preserving MiniMax H3 production on Colab G4."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time


HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
WHEEL = SKILL / "assets" / "sageattention-2.2.0-cp312-cp312-cu130_sm120_linux_x86_64.whl"
REMOTE_FILES = {
    HERE / "remote_cuda13.py": "/content/h3_remote_cuda13.py",
    HERE / "remote_h3_job.py": "/content/h3_remote_job.py",
    HERE / "remote_entry.py": "/content/h3_remote_entry.py",
    HERE / "remote_launch.py": "/content/h3_remote_launch.py",
    HERE / "remote_status.py": "/content/h3_remote_status.py",
    WHEEL: "/content/sageattention-2.2.0-cp312-cp312-cu130_sm120_linux_x86_64.whl",
}


class PreservedFailure(RuntimeError):
    pass


def emit(message: str) -> None:
    print(f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def run(command: list[str], *, timeout: float | None = None, capture: bool = False) -> str:
    emit("+ " + " ".join(command))
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        if capture and result.stdout:
            print(result.stdout, end="", flush=True)
        raise PreservedFailure(f"command failed ({result.returncode}): {' '.join(command)}")
    return result.stdout or ""


def last_json(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise PreservedFailure(f"no JSON object in command output: {text[-1000:]}")


def colab_python() -> str:
    executable = shutil.which("colab")
    if not executable:
        raise PreservedFailure("colab CLI is not installed")
    first = Path(executable).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if not first.startswith("#!"):
        raise PreservedFailure("cannot locate the Colab CLI Python runtime")
    return first[2:]


def ensure_session(name: str, gpu: str) -> dict[str, object]:
    decision = last_json(run([colab_python(), str(HERE / "session_guard.py"), name], capture=True))
    action = decision.get("action")
    if action == "create":
        emit("No server assignment exists; creating exactly one new Colab session.")
        run(["colab", "new", "-s", name, "--gpu", gpu], timeout=600)
        decision = last_json(run([colab_python(), str(HERE / "session_guard.py"), name], capture=True))
    elif action == "rebind":
        emit(f"Recovered [?] assignment as session '{name}'; no new instance created.")
    elif action == "reuse":
        emit(f"Reusing active session '{name}'.")
    elif action == "ambiguous":
        raise PreservedFailure(
            "multiple server assignments exist and the requested session is not bound; "
            f"refusing to create or guess: {decision.get('endpoints')}"
        )
    if decision.get("action") not in {"reuse", "rebind"}:
        raise PreservedFailure(f"session did not become ready: {decision}")
    return decision


def rounded_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def valid_frames(duration: float, fps: int) -> int:
    requested = duration * fps
    n = max(0, round((requested - 5) / 17))
    return 5 + 17 * n


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_state(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def build_config(args: argparse.Namespace) -> dict[str, object]:
    if args.prompt and args.prompt_file:
        raise PreservedFailure("use either --prompt or --prompt-file, not both")
    prompt = args.prompt or (args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else "")
    if not prompt.strip():
        raise PreservedFailure("a non-empty --prompt or --prompt-file is required")
    if not args.reference_image.is_file():
        raise PreservedFailure(f"reference image does not exist: {args.reference_image}")
    if args.resolution:
        try:
            requested_width, requested_height = (
                int(part) for part in args.resolution.lower().split("x", 1)
            )
        except (ValueError, TypeError) as exc:
            raise PreservedFailure("--resolution must look like 1920x1080") from exc
        if args.width is not None or args.height is not None:
            raise PreservedFailure("use --resolution or --width/--height, not both")
    elif args.width is not None and args.height is not None:
        requested_width, requested_height = args.width, args.height
    else:
        raise PreservedFailure("provide --resolution or both --width and --height")
    width = rounded_multiple(requested_width, 32)
    height = rounded_multiple(requested_height, 32)
    frames = args.frames if args.frames is not None else valid_frames(args.duration, args.fps)
    if (frames - 5) % 17:
        raise PreservedFailure("--frames must satisfy frames = 5 + 17*n")
    if not 1 <= args.steps <= 50:
        raise PreservedFailure("--steps must be between 1 and 50")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = args.job_id or f"h3-{width}x{height}-{frames}f-{args.steps}s-{stamp}"
    return {
        "job_id": job_id,
        "label": f"{width}x{height}_{frames}f",
        "width": width,
        "height": height,
        "frames": frames,
        "fps": args.fps,
        "steps": args.steps,
        "seed": args.seed,
        "prompt": prompt,
        "warmup": args.warmup,
        "timeout_minutes": args.timeout_minutes,
        "remote_reference_image": f"/content/h3-reference-{job_id}{args.reference_image.suffix.lower()}",
        "requested": {"width": requested_width, "height": requested_height, "duration": args.duration},
    }


def upload(session: str, local: Path, remote: str) -> None:
    run(["colab", "upload", "-s", session, str(local), remote], timeout=900)


def monitor(session: str, config: dict[str, object], poll_seconds: int) -> dict[str, object]:
    while True:
        try:
            output = run(
                ["colab", "exec", "-s", session, "-f", str(HERE / "remote_status.py"), "--timeout", "30"],
                timeout=90,
                capture=True,
            )
        except (PreservedFailure, subprocess.TimeoutExpired) as exc:
            raise PreservedFailure(f"log polling lost connection; instance preserved: {exc}") from exc
        status = last_json(output)
        new = str(status.get("new_log", ""))
        if new:
            print(new, end="" if new.endswith("\n") else "\n", flush=True)
        if not status.get("running"):
            if status.get("success") and status.get("archive_exists"):
                return status
            raise PreservedFailure("remote job stopped without a success marker; instance preserved")
        time.sleep(poll_seconds)


def execute(args: argparse.Namespace) -> int:
    config = build_config(args)
    job_id = str(config["job_id"])
    output_dir = args.output_dir.resolve() / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "operator-state.json"
    state: dict[str, object] = {
        "status": "planned",
        "session": args.session,
        "job": config,
        "output_dir": str(output_dir),
    }
    write_state(state_path, state)
    emit(
        f"job={job_id} effective={config['width']}x{config['height']} "
        f"frames={config['frames']} duration={int(config['frames']) / int(config['fps']):.3f}s "
        f"steps={config['steps']}"
    )
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        emit(f"DRY_RUN_COMPLETE state={state_path}")
        return 0

    created_or_bound = False
    try:
        decision = ensure_session(args.session, args.gpu)
        created_or_bound = True
        state.update(status="session-ready", endpoint=decision.get("endpoint"))
        write_state(state_path, state)
        config_path = output_dir / "job-config.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        for local, remote in REMOTE_FILES.items():
            upload(args.session, local, remote)
        upload(args.session, args.reference_image.resolve(), str(config["remote_reference_image"]))
        upload(args.session, config_path, "/content/h3_job.json")
        state["status"] = "uploaded"
        write_state(state_path, state)
        run(
            ["colab", "exec", "-s", args.session, "-f", str(HERE / "remote_launch.py"), "--timeout", "30"],
            timeout=90,
        )
        state["status"] = "running"
        write_state(state_path, state)
        monitor(args.session, config, args.poll_seconds)
        archive = output_dir / f"h3-results-{job_id}.tar.gz"
        remote_archive = f"/content/h3-results-{job_id}.tar.gz"
        run(["colab", "download", "-s", args.session, remote_archive, str(archive)], timeout=1800)
        run(["colab", "download", "-s", args.session, f"/content/h3-job-{job_id}.log", str(output_dir / "remote-job.log")], timeout=600)
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if not members or not any(member.name.endswith("/runs.json") for member in members):
                raise PreservedFailure("downloaded archive is incomplete; instance preserved")
        state.update(status="verified", archive=str(archive), archive_sha256=sha256(archive))
        write_state(state_path, state)
        if not args.keep_on_success:
            run(["colab", "stop", "-s", args.session], timeout=180)
            inventory = last_json(run([colab_python(), str(HERE / "session_inventory.py")], capture=True))
            if decision.get("endpoint") in inventory.get("endpoints", []):
                raise PreservedFailure("Colab stop returned but the assignment endpoint is still active")
            state["instance_stopped"] = True
        else:
            state["instance_stopped"] = False
        state["status"] = "complete"
        write_state(state_path, state)
        emit(f"COMPLETE archive={archive} sha256={state['archive_sha256']}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status="failed-preserved", error=repr(exc), instance_preserved=created_or_bound)
        write_state(state_path, state)
        emit(f"FAILED_INSTANCE_PRESERVED state={state_path} error={exc!r}")
        return 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--steps", type=int, required=True)
    result.add_argument("--resolution", help="WIDTHxHEIGHT, for example 1920x1080")
    result.add_argument("--width", type=int)
    result.add_argument("--height", type=int)
    length = result.add_mutually_exclusive_group(required=True)
    length.add_argument("--duration", type=float)
    length.add_argument("--frames", type=int)
    prompt = result.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    result.add_argument("--reference-image", type=Path, required=True)
    result.add_argument("--fps", type=int, default=24)
    result.add_argument("--seed", type=int, default=424242)
    result.add_argument("--session", default="minimax-h3-production")
    result.add_argument("--gpu", default="G4")
    result.add_argument("--job-id")
    result.add_argument("--output-dir", type=Path, default=Path.cwd() / "h3-colab-results")
    result.add_argument("--poll-seconds", type=int, default=30)
    result.add_argument("--timeout-minutes", type=int, default=180)
    result.add_argument("--warmup", action="store_true")
    result.add_argument("--keep-on-success", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(execute(parser().parse_args()))
