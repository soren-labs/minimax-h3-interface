#!/usr/bin/env python3
"""One-command, fail-preserving MiniMax H3 production on Colab G4."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
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


def parse_resolution(args: argparse.Namespace) -> tuple[int, int, int, int]:
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
    return (
        requested_width,
        requested_height,
        rounded_multiple(requested_width, 32),
        rounded_multiple(requested_height, 32),
    )


def probe_video(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise PreservedFailure(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def video_shape(info: dict[str, object], path: Path) -> tuple[int, int, int, int, float]:
    streams = info.get("streams", [])
    if not isinstance(streams, list):
        raise PreservedFailure(f"invalid ffprobe streams for {path}")
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if not isinstance(video, dict) or not isinstance(audio, dict):
        raise PreservedFailure(f"repaint input must contain video and audio: {path}")
    rate_text = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
    numerator, denominator = (int(part) for part in rate_text.split("/", 1))
    fps = round(numerator / denominator) if denominator else 0
    duration = float(dict(info.get("format", {})).get("duration", 0.0))
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or round(duration * fps))
    return int(video["width"]), int(video["height"]), fps, frames, duration


def build_generate_config(args: argparse.Namespace) -> dict[str, object]:
    if args.prompt and args.prompt_file:
        raise PreservedFailure("use either --prompt or --prompt-file, not both")
    prompt = args.prompt or (args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else "")
    if not prompt.strip():
        raise PreservedFailure("a non-empty --prompt or --prompt-file is required")
    if args.reference_image is None or not args.reference_image.is_file():
        raise PreservedFailure(f"reference image does not exist: {args.reference_image}")
    if args.frames is None and args.duration is None:
        raise PreservedFailure("generate mode requires --duration or --frames")
    requested_width, requested_height, width, height = parse_resolution(args)
    frames = args.frames if args.frames is not None else valid_frames(args.duration, args.fps)
    if (frames - 5) % 17:
        raise PreservedFailure("--frames must satisfy frames = 5 + 17*n")
    if not 1 <= args.steps <= 50:
        raise PreservedFailure("--steps must be between 1 and 50")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = args.job_id or f"h3-{width}x{height}-{frames}f-{args.steps}s-{stamp}"
    return {
        "mode": "generate",
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


def build_repaint_config(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_manifest is None or not args.batch_manifest.is_file():
        raise PreservedFailure("repaint mode requires an existing --batch-manifest")
    if args.ngc_env is None or not args.ngc_env.is_file():
        raise PreservedFailure("repaint mode requires an existing --ngc-env")
    ngc_lines = args.ngc_env.read_text(encoding="utf-8").splitlines()
    if not any(
        line.startswith("NGC_CLI_API_KEY=") and line.split("=", 1)[1].strip()
        for line in ngc_lines
    ):
        raise PreservedFailure("--ngc-env must contain a non-empty NGC_CLI_API_KEY")
    requested_width, requested_height, width, height = parse_resolution(args)
    if not 1 <= args.steps <= 50:
        raise PreservedFailure("--steps must be between 1 and 50")
    if not 0.0 < args.denoise <= 1.0:
        raise PreservedFailure("--denoise must be in (0, 1]")
    manifest = json.loads(args.batch_manifest.read_text(encoding="utf-8"))
    raw_jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise PreservedFailure("batch manifest must contain a non-empty jobs array")
    jobs: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_jobs, start=1):
        if not isinstance(raw, dict):
            raise PreservedFailure(f"manifest job {index} must be an object")
        identifier = str(raw.get("id", "")).strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", identifier) or identifier in seen:
            raise PreservedFailure(f"invalid or duplicate repaint job id: {identifier!r}")
        seen.add(identifier)
        video_path = Path(str(raw.get("input_video", ""))).expanduser().resolve()
        if not video_path.is_file():
            raise PreservedFailure(f"repaint input video does not exist: {video_path}")
        prompt = str(raw.get("prompt", ""))
        if raw.get("prompt_file"):
            prompt_path = Path(str(raw["prompt_file"])).expanduser().resolve()
            if not prompt_path.is_file():
                raise PreservedFailure(f"repaint prompt file does not exist: {prompt_path}")
            prompt = prompt_path.read_text(encoding="utf-8")
        if not prompt.strip():
            raise PreservedFailure(f"repaint job {identifier} has an empty prompt")
        info = probe_video(video_path)
        source_width, source_height, fps, frames, duration = video_shape(info, video_path)
        if fps != args.fps or (frames - 5) % 17:
            raise PreservedFailure(
                f"{identifier} must be {args.fps}fps on the H3 17k+5 grid; got fps={fps}, frames={frames}"
            )
        if not 2.0 <= duration <= 15.1:
            raise PreservedFailure(f"{identifier} duration must be 2-15s; got {duration:.3f}s")
        remote_video = f"/content/h3-repaint-input-{identifier}{video_path.suffix.lower()}"
        jobs.append(
            {
                "id": identifier,
                "input_video": str(video_path),
                "remote_video": remote_video,
                "source_sha256": sha256(video_path),
                "source_width": source_width,
                "source_height": source_height,
                "frames": frames,
                "duration": duration,
                "prompt": prompt.strip(),
                "seed": int(raw.get("seed", args.seed)),
            }
        )
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = args.job_id or f"h3-repaint-{width}x{height}-{len(jobs)}clips-{args.steps}step-{stamp}"
    return {
        "mode": "repaint",
        "job_id": job_id,
        "label": f"repaint_{width}x{height}_{len(jobs)}clips",
        "width": width,
        "height": height,
        "fps": args.fps,
        "steps": args.steps,
        "denoise": args.denoise,
        "jobs": jobs,
        "warmup": False,
        "timeout_minutes": args.timeout_minutes,
        "upscale_method": "nvidia_vfx_upscale_2x_then_lanczos",
        "remote_ngc_env": "/content/.ngc.env",
        "requested": {"width": requested_width, "height": requested_height},
    }


def build_config(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "repaint":
        return build_repaint_config(args)
    return build_generate_config(args)


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


def finalize_repaint(archive: Path, output_dir: Path, config: dict[str, object]) -> Path:
    extracted = output_dir / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(extracted, filter="data")
    expected = [str(job["id"]) for job in config["jobs"]]
    source_audio_dir = output_dir / "source-audio-clips"
    source_audio_dir.mkdir(parents=True, exist_ok=True)
    media_files: list[Path] = []
    verification: dict[str, object] = {"clips": []}
    for identifier in expected:
        candidates = sorted(extracted.glob(f"**/media/repaint_{identifier}.mp4"))
        if len(candidates) != 1:
            raise PreservedFailure(f"expected one media file for {identifier}, found {candidates}")
        media = candidates[0]
        info = probe_video(media)
        media_width, media_height, fps, frames, duration = video_shape(info, media)
        if (media_width, media_height, fps, frames) != (
            int(config["width"]), int(config["height"]), int(config["fps"]),
            int(next(job["frames"] for job in config["jobs"] if job["id"] == identifier)),
        ):
            raise PreservedFailure(
                f"invalid repaint media {identifier}: {media_width}x{media_height} {fps}fps {frames}f"
            )
        streams = info.get("streams", [])
        video_stream = next(stream for stream in streams if stream.get("codec_type") == "video")
        audio_stream = next(stream for stream in streams if stream.get("codec_type") == "audio")
        if video_stream.get("codec_name") != "h264" or audio_stream.get("codec_name") != "aac":
            raise PreservedFailure(f"invalid codecs for {identifier}")
        if int(audio_stream.get("channels", 0)) != 2:
            raise PreservedFailure(f"output audio is not stereo for {identifier}")
        decode = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(media), "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if decode.returncode:
            raise PreservedFailure(f"decode verification failed for {identifier}: {decode.stderr.strip()}")
        job = next(job for job in config["jobs"] if job["id"] == identifier)
        source = Path(str(job["input_video"]))
        if not source.is_file():
            raise PreservedFailure(f"source video for original audio is missing: {source}")
        source_audio_media = source_audio_dir / f"repaint_{identifier}.mp4"
        run(
            [
                "ffmpeg", "-y", "-v", "warning", "-i", str(media), "-i", str(source),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
                "-map_metadata", "0", str(source_audio_media),
            ],
            timeout=300,
        )
        source_audio_info = probe_video(source_audio_media)
        source_audio_stream = next(
            stream for stream in source_audio_info.get("streams", [])
            if stream.get("codec_type") == "audio"
        )
        if source_audio_stream.get("codec_name") != "aac" or int(source_audio_stream.get("channels", 0)) != 2:
            raise PreservedFailure(f"source-audio remux is invalid for {identifier}")
        media_files.append(source_audio_media)
        verification["clips"].append(
            {
                "id": identifier,
                "remote_path": str(media),
                "remote_sha256": sha256(media),
                "final_clip_path": str(source_audio_media),
                "final_clip_sha256": sha256(source_audio_media),
                "source_audio_path": str(source),
                "duration": duration,
            }
        )
    concat_file = output_dir / "concat.txt"
    if any("'" in str(path) for path in media_files):
        raise PreservedFailure("cannot concatenate media paths containing apostrophes")
    concat_file.write_text(
        "".join(f"file '{path}'\n" for path in media_files),
        encoding="utf-8",
    )
    final_video = output_dir / f"xianxia-h3-repaint-{config['width']}x{config['height']}.mp4"
    run(
        ["ffmpeg", "-y", "-v", "warning", "-f", "concat", "-safe", "0", "-i", str(concat_file),
         "-c", "copy", str(final_video)],
        timeout=900,
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(final_video), "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if decode.returncode:
        raise PreservedFailure(f"final video decode verification failed: {decode.stderr.strip()}")
    final_probe = probe_video(final_video)
    final_width, final_height, final_fps, final_frames, final_duration = video_shape(final_probe, final_video)
    expected_frames = sum(int(job["frames"]) for job in config["jobs"])
    if (final_width, final_height, final_fps, final_frames) != (
        int(config["width"]), int(config["height"]), int(config["fps"]), expected_frames,
    ):
        raise PreservedFailure(
            f"invalid final media: {final_width}x{final_height} {final_fps}fps {final_frames}f"
        )
    verification["final"] = {
        "path": str(final_video), "sha256": sha256(final_video), "duration": final_duration,
        "ffprobe": final_probe,
    }
    (output_dir / "local-verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return final_video


def validate_archive(archive: Path, config: dict[str, object]) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        member_names = {member.name for member in members}
        required_suffixes = ["/runs.json"]
        if config["mode"] == "repaint":
            required_suffixes.extend(
                [
                    "/benchmark.json", "/benchmark.csv", "/environment-lock.json",
                    "/important-logs/comfyui.log", "/important-logs/setup.log",
                    "/important-logs/models.log", "/important-logs/nvidia-upscale-smoke.log",
                    "/important-logs/pt-concat-smoke.log",
                ]
            )
            required_suffixes.extend(
                f"/shot-archives/{job['id']}.tar.gz" for job in config["jobs"]
            )
        missing = [
            suffix for suffix in required_suffixes
            if not any(name.endswith(suffix) for name in member_names)
        ]
        if not members or missing:
            raise PreservedFailure(
                f"downloaded archive is incomplete; missing={missing}; instance preserved"
            )


def resume_existing(args: argparse.Namespace) -> int:
    output_dir = args.resume_job_dir.resolve()
    config_path = output_dir / "job-config.json"
    state_path = output_dir / "operator-state.json"
    if not config_path.is_file() or not state_path.is_file():
        raise PreservedFailure("--resume-job-dir must contain job-config.json and operator-state.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job_id = str(config["job_id"])
    decision: dict[str, object] | None = None
    try:
        decision = ensure_session(args.session, args.gpu)
        if state.get("endpoint") and decision.get("endpoint") != state.get("endpoint"):
            raise PreservedFailure(
                f"resume endpoint mismatch: expected={state.get('endpoint')} actual={decision.get('endpoint')}"
            )
        if args.repair_relaunch:
            completed = list(dict.fromkeys(args.resume_completed or []))
            known = {str(job["id"]) for job in config.get("jobs", [])}
            invalid = sorted(set(completed) - known)
            if invalid:
                raise PreservedFailure(f"unknown --resume-completed ids: {invalid}")
            config["resume_completed_ids"] = completed
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            for local, remote in REMOTE_FILES.items():
                upload(args.session, local, remote)
            upload(args.session, config_path, "/content/h3_job.json")
            run(
                ["colab", "exec", "-s", args.session, "-f", str(HERE / "remote_launch.py"), "--timeout", "30"],
                timeout=90,
            )
            state.update(status="repair-relaunched", resume_completed_ids=completed)
        else:
            state.update(status="resumed-running")
        state["endpoint"] = decision.get("endpoint")
        write_state(state_path, state)
        monitor(args.session, config, args.poll_seconds)
        archive = output_dir / f"h3-results-{job_id}.tar.gz"
        run(
            ["colab", "download", "-s", args.session, f"/content/h3-results-{job_id}.tar.gz", str(archive)],
            timeout=1800,
        )
        run(
            ["colab", "download", "-s", args.session, f"/content/h3-job-{job_id}.log", str(output_dir / "remote-job.log")],
            timeout=600,
        )
        validate_archive(archive, config)
        final_video = finalize_repaint(archive, output_dir, config) if config["mode"] == "repaint" else None
        state.update(status="verified", archive=str(archive), archive_sha256=sha256(archive))
        if final_video is not None:
            state["final_video"] = str(final_video)
        write_state(state_path, state)
        if not args.keep_on_success:
            run(["colab", "stop", "-s", args.session], timeout=180)
            inventory = last_json(run([colab_python(), str(HERE / "session_inventory.py")], capture=True))
            if decision.get("endpoint") in inventory.get("endpoints", []):
                raise PreservedFailure("Colab stop returned but the assignment endpoint is still active")
            state["instance_stopped"] = True
        else:
            state["instance_stopped"] = False
        state.pop("error", None)
        state.pop("instance_preserved", None)
        state["status"] = "complete"
        write_state(state_path, state)
        emit(f"COMPLETE_RESUMED archive={archive} sha256={state['archive_sha256']}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status="failed-preserved", error=repr(exc), instance_preserved=decision is not None)
        write_state(state_path, state)
        emit(f"FAILED_INSTANCE_PRESERVED state={state_path} error={exc!r}")
        return 2


def execute(args: argparse.Namespace) -> int:
    if args.resume_job_dir is not None:
        return resume_existing(args)
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
    if config["mode"] == "repaint":
        emit(
            f"job={job_id} mode=repaint effective={config['width']}x{config['height']} "
            f"clips={len(config['jobs'])} steps={config['steps']} denoise={config['denoise']}"
        )
    else:
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
        if config["mode"] == "repaint":
            upload(args.session, args.ngc_env.resolve(), str(config["remote_ngc_env"]))
            for job in config["jobs"]:
                upload(args.session, Path(str(job["input_video"])), str(job["remote_video"]))
        else:
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
        validate_archive(archive, config)
        final_video = None
        if config["mode"] == "repaint":
            final_video = finalize_repaint(archive, output_dir, config)
        state.update(status="verified", archive=str(archive), archive_sha256=sha256(archive))
        if final_video is not None:
            state["final_video"] = str(final_video)
        write_state(state_path, state)
        if not args.keep_on_success:
            run(["colab", "stop", "-s", args.session], timeout=180)
            inventory = last_json(run([colab_python(), str(HERE / "session_inventory.py")], capture=True))
            if decision.get("endpoint") in inventory.get("endpoints", []):
                raise PreservedFailure("Colab stop returned but the assignment endpoint is still active")
            state["instance_stopped"] = True
        else:
            state["instance_stopped"] = False
        state.pop("error", None)
        state.pop("instance_preserved", None)
        state["status"] = "complete"
        write_state(state_path, state)
        emit(f"COMPLETE archive={archive} sha256={state['archive_sha256']}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if created_or_bound:
            for remote, local in (
                (f"/content/h3-results-{job_id}.tar.gz", output_dir / f"h3-results-{job_id}.failure.tar.gz"),
                (f"/content/h3-job-{job_id}.log", output_dir / "remote-job.failure.log"),
            ):
                try:
                    run(["colab", "download", "-s", args.session, remote, str(local)], timeout=600)
                except Exception:
                    pass
        state.update(status="failed-preserved", error=repr(exc), instance_preserved=created_or_bound)
        write_state(state_path, state)
        emit(f"FAILED_INSTANCE_PRESERVED state={state_path} error={exc!r}")
        return 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("generate", "repaint"), default="generate")
    result.add_argument("--steps", type=int, required=True)
    result.add_argument("--denoise", type=float, default=0.2)
    result.add_argument("--resolution", help="WIDTHxHEIGHT, for example 1920x1080")
    result.add_argument("--width", type=int)
    result.add_argument("--height", type=int)
    length = result.add_mutually_exclusive_group()
    length.add_argument("--duration", type=float)
    length.add_argument("--frames", type=int)
    prompt = result.add_mutually_exclusive_group()
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    result.add_argument("--reference-image", type=Path)
    result.add_argument("--batch-manifest", type=Path)
    result.add_argument("--resume-job-dir", type=Path)
    result.add_argument("--repair-relaunch", action="store_true")
    result.add_argument("--resume-completed", action="append")
    result.add_argument("--ngc-env", type=Path, help="env file containing NGC_CLI_API_KEY")
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
