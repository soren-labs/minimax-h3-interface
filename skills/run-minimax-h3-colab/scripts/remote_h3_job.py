from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
import uuid


START = time.monotonic()
CONFIG_PATH = Path(os.environ.get("H3_JOB_CONFIG", "/content/h3_job.json"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
JOB_ID = str(CONFIG["job_id"])
HARD_DEADLINE = START + int(CONFIG.get("timeout_minutes", 180)) * 60
WORK = Path("/content/h3-turbo-work")
COMFY = WORK / "ComfyUI"
RESULTS = Path("/content") / f"h3-results-{JOB_ID}"
MEDIA = RESULTS / "media"
LOGS = RESULTS / "logs"
COMPARE = RESULTS / "comparisons"
SERVER_LOG = LOGS / "comfyui.log"
TELEMETRY = RESULTS / "telemetry.csv"
ARCHIVE = Path("/content") / f"h3-results-{JOB_ID}.tar.gz"

ASSET_SOURCE = Path(str(CONFIG["remote_reference_image"]))
ASSET_BASENAME = "reference_h3"
SAGE_WHEEL_SOURCE = Path(
    "/content/sageattention-2.2.0-cp312-cp312-cu130_sm120_linux_x86_64.whl"
)
SAGE_WHEEL_SHA256 = "a055eb25f7c11a05d3725216a4853bc406c5c0266ac5c995f57e1283651d4cc2"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFY_REV = "dd79c643a95402136a75a28f6187d843bcf457ed"
TURBO_REPO = "https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git"
TURBO_REV = "55fee864dd7b2976b1c4ce3c3d5f7968f181409f"
SAGE_REPO = "https://github.com/thu-ml/SageAttention.git"
SAGE_REV = "eb615cf6cf4d221338033340ee2de1c37fbdba4a"
KJ_REPO = "https://github.com/kijai/ComfyUI-KJNodes.git"
KJ_REV = "60cd6bc1870db94c6eeb05fbe455147a8e91c4e9"
MODEL_REPO = "Comfy-Org/MiniMax-H3"
MODEL_REV = "eb8a16107c595128b3a578f82d2ce2f75920c355"
LORA_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
LORA_REV = "afc0346516372a17162c14df3c5264de1d9aa1c0"

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"

FPS = int(CONFIG.get("fps", 24))
SEED = int(CONFIG.get("seed", 424242))
CASES = (
    {
        "label": str(CONFIG["label"]),
        "width": int(CONFIG["width"]),
        "height": int(CONFIG["height"]),
        "length": int(CONFIG["frames"]),
        "steps": (int(CONFIG["steps"]),),
    },
)

PROMPT = str(CONFIG["prompt"])


def remaining(reserve: float = 0.0) -> float:
    value = HARD_DEADLINE - time.monotonic() - reserve
    if value <= 0:
        raise TimeoutError("remote hard deadline reached")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def command_text(cmd: list[str]) -> str:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout


def run_logged(
    cmd: list[str],
    log_name: str,
    timeout: float,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    log_path = LOGS / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        deadline = time.monotonic() + min(timeout, remaining(600))
        assert proc.stdout is not None
        try:
            while True:
                line = proc.stdout.readline()
                if line:
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
                elif proc.poll() is not None:
                    break
                else:
                    if time.monotonic() >= deadline:
                        proc.terminate()
                        try:
                            proc.wait(20)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        raise TimeoutError(f"command timed out: {' '.join(cmd)}")
                    time.sleep(0.2)
            for line in proc.stdout:
                log.write(line)
                print(line.rstrip(), flush=True)
        finally:
            if proc.poll() is None:
                proc.kill()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def preflight() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / 1024**3
    if "RTX PRO 6000" not in props.name or vram_gib < 90:
        raise RuntimeError(
            f"expected G4 RTX PRO 6000 >=90GiB, got {props.name} {vram_gib:.1f}GiB"
        )
    disk = shutil.disk_usage("/content")
    if disk.free < 75 * 1024**3:
        raise RuntimeError(
            f"insufficient /content free space: {disk.free / 1024**3:.1f}GiB"
        )
    info = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu": props.name,
        "vram_gib": round(vram_gib, 2),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "nvidia_smi": command_text(["nvidia-smi"]),
    }
    write_json(RESULTS / "hardware.json", info)
    return info


def checkout(repo: str, destination: Path, revision: str, log_name: str) -> None:
    if not destination.exists():
        run_logged(
            ["git", "clone", "--filter=blob:none", repo, str(destination)],
            log_name,
            300,
        )
    run_logged(
        ["git", "-C", str(destination), "fetch", "origin", revision, "--depth", "1"],
        log_name,
        180,
    )
    run_logged(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        log_name,
        60,
    )


def setup() -> list[Path]:
    checkout(COMFY_REPO, COMFY, COMFY_REV, "setup.log")
    turbo = COMFY / "custom_nodes" / "ComfyUI-MiniMax-H3-Turbo"
    checkout(TURBO_REPO, turbo, TURBO_REV, "setup.log")
    kj_nodes = COMFY / "custom_nodes" / "ComfyUI-KJNodes"
    checkout(KJ_REPO, kj_nodes, KJ_REV, "setup.log")
    run_logged(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(COMFY / "requirements.txt")],
        "setup.log",
        900,
    )
    run_logged(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(kj_nodes / "requirements.txt")],
        "setup.log",
        600,
    )
    run_logged(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "huggingface_hub", "hf_xet"],
        "setup.log",
        300,
    )

    download_code = f'''from huggingface_hub import hf_hub_download, snapshot_download
print("download pinned MiniMax H3 base files", flush=True)
snapshot_download(
    repo_id={MODEL_REPO!r},
    revision={MODEL_REV!r},
    allow_patterns={[
        'diffusion_models/' + UNET,
        'text_encoders/' + CLIP,
        'vae/' + VIDEO_VAE,
        'vae/' + AUDIO_VAE,
    ]!r},
    local_dir={str(COMFY / 'models')!r},
    max_workers=4,
)
print("download pinned Turbo LoRA", flush=True)
hf_hub_download(
    repo_id={LORA_REPO!r},
    revision={LORA_REV!r},
    filename={LORA!r},
    local_dir={str(COMFY / 'models' / 'loras')!r},
)
'''
    code_file = WORK / "download_models.py"
    code_file.write_text(download_code, encoding="utf-8")
    env = os.environ.copy()
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    run_logged(
        [sys.executable, str(code_file)],
        "models.log",
        min(45 * 60, remaining(600)),
        env=env,
    )

    expected = [
        COMFY / "models" / "diffusion_models" / UNET,
        COMFY / "models" / "text_encoders" / CLIP,
        COMFY / "models" / "vae" / VIDEO_VAE,
        COMFY / "models" / "vae" / AUDIO_VAE,
        COMFY / "models" / "loras" / LORA,
    ]
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size < 100_000_000]
    if missing:
        raise RuntimeError(f"model files missing or incomplete: {missing}")
    write_json(
        RESULTS / "models.json",
        {str(path.relative_to(COMFY)): path.stat().st_size for path in expected},
    )
    return expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_sage_from_source() -> None:
    sage = WORK / "SageAttention"
    checkout(SAGE_REPO, sage, SAGE_REV, "sageattention.log")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "EXT_PARALLEL": "4",
            "NVCC_APPEND_FLAGS": "--threads 8",
            "MAX_JOBS": "16",
        }
    )
    run_logged(
        [sys.executable, "setup.py", "install"],
        "sageattention.log",
        min(15 * 60, remaining(600)),
        cwd=sage,
        env=env,
    )


def install_and_verify_sage(hardware: dict[str, object]) -> dict[str, object]:
    wheel_compatible = (
        SAGE_WHEEL_SOURCE.is_file()
        and hardware.get("torch") == "2.11.0+cu130"
        and hardware.get("torch_cuda") == "13.0"
        and hardware.get("compute_capability") == [12, 0]
        and sys.version_info[:2] == (3, 12)
    )
    mode = "source"
    wheel_hash = None
    if wheel_compatible:
        wheel_hash = sha256(SAGE_WHEEL_SOURCE)
        if wheel_hash != SAGE_WHEEL_SHA256:
            raise RuntimeError(f"SageAttention wheel checksum mismatch: {wheel_hash}")
        # The archived filename carries explicit cu130/sm120 provenance, while
        # pip requires the canonical PEP 427 wheel tag to parse compatibility.
        install_wheel = Path("/tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl")
        shutil.copy2(SAGE_WHEEL_SOURCE, install_wheel)
        run_logged(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(install_wheel)],
            "sageattention.log",
            180,
        )
        mode = "prebuilt-wheel"
    else:
        compile_sage_from_source()
    verify = (
        "import json, sageattention, torch; "
        "import sageattention._qattn_sm80, sageattention._qattn_sm89; "
        "print(json.dumps({'module': sageattention.__file__, "
        "'torch': torch.__version__, 'cuda': torch.version.cuda, "
        "'capability': torch.cuda.get_device_capability(0)}))"
    )
    run_logged([sys.executable, "-c", verify], "sageattention.log", 60)
    result = {
        "mode": mode,
        "revision": SAGE_REV,
        "wheel_sha256": wheel_hash,
        "required": True,
        "verified": True,
    }
    write_json(RESULTS / "sageattention.json", result)
    return result


def asset_name(width: int, height: int) -> str:
    return f"{ASSET_BASENAME}_{width}x{height}.png"


def prepare_assets() -> list[Path]:
    from PIL import Image, ImageOps

    if not ASSET_SOURCE.is_file():
        raise FileNotFoundError(ASSET_SOURCE)
    source = Image.open(ASSET_SOURCE).convert("RGB")
    assets = RESULTS / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    targets = []
    dimensions = {(608, 352)} | {
        (int(case["width"]), int(case["height"])) for case in CASES
    }
    for width, height in sorted(dimensions):
        image = ImageOps.fit(
            source,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        name = asset_name(width, height)
        target = COMFY / "input" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG")
        shutil.copy2(target, assets / name)
        targets.append(target)
    return targets


def http_json(
    path: str,
    method: str = "GET",
    payload: object | None = None,
    timeout: float = 20,
) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:8188{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _tee_server_output(proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    with SERVER_LOG.open("a", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"[comfy] {line.rstrip()}", flush=True)


def wait_server(proc: subprocess.Popen[str], timeout: float = 240) -> dict[str, object]:
    required = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "MiniMaxH3ImageToVideo",
        "MiniMaxH3TurboLoRA",
        "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "MiniMaxH3TurboSampler",
        "SamplerCustomAdvanced",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    }
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}")
        try:
            info = http_json("/object_info", timeout=10)
            if not isinstance(info, dict):
                raise TypeError("/object_info was not an object")
            absent = sorted(required - set(info))
            if absent:
                raise RuntimeError(f"required ComfyUI nodes unavailable: {absent}")
            return info
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(3)
    raise TimeoutError(f"ComfyUI did not become ready: {last_error}")


def start_server() -> tuple[subprocess.Popen[str], threading.Thread]:
    args = [
        sys.executable,
        str(COMFY / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--highvram",
        "--fast",
        "fp16_accumulation",
        "--use-sage-attention",
        "--disable-auto-launch",
    ]
    print("+", " ".join(args), flush=True)
    proc = subprocess.Popen(
        args,
        cwd=COMFY,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    thread = threading.Thread(target=_tee_server_output, args=(proc,), daemon=True)
    thread.start()
    wait_server(proc)
    log_text = SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    if "Using sage attention" not in log_text:
        raise RuntimeError("ComfyUI started without confirming SageAttention")
    return proc, thread


def stop_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def stop_stale_comfy_server() -> None:
    subprocess.run(
        ["pkill", "-f", str(COMFY / "main.py")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(2)


def workflow(
    *,
    width: int,
    height: int,
    length: int,
    steps: int,
    save_name: str | None,
) -> dict[str, object]:
    nodes: dict[str, object] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "5": {"class_type": "LoadImage", "inputs": {"image": asset_name(width, height)}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": PROMPT,
            "width": width, "height": height, "length": length,
            "first_frame": ["5", 0],
        }},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "8": {"class_type": "MiniMaxH3TurboLoRA", "inputs": {
            "model": ["1", 0], "lora_name": LORA, "strength": 1.0,
            "low_vram": False,
        }},
        "18": {"class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch", "inputs": {
            "model": ["8", 0],
        }},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["18", 0], "conditioning": ["6", 0]}},
        "10": {"class_type": "BasicScheduler", "inputs": {
            "model": ["18", 0], "scheduler": "simple", "steps": steps,
            "denoise": 1.0,
        }},
        "11": {"class_type": "MiniMaxH3TurboSampler", "inputs": {}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["7", 0], "guider": ["9", 0], "sampler": ["11", 0],
            "sigmas": ["10", 0], "latent_image": ["6", 1],
        }},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
    }
    if save_name is None:
        nodes["14"] = {"class_type": "PreviewImage", "inputs": {"images": ["13", 0]}}
    else:
        nodes.update(
            {
                "15": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
                "16": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["15", 0], "fps": FPS, "bit_depth": 8}},
                "17": {"class_type": "SaveVideo", "inputs": {
                    "video": ["16", 0], "filename_prefix": f"h3-turbo/{save_name}",
                    "format": "auto", "codec": "auto",
                }},
            }
        )
    return nodes


def prompt_error(history: dict[str, object]) -> str | None:
    status = history.get("status", {})
    if isinstance(status, dict) and status.get("status_str") == "error":
        return json.dumps(status.get("messages", []), ensure_ascii=False)
    return None


def gpu_heartbeat() -> str:
    return command_text(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).strip()


def submit_and_wait(
    nodes: dict[str, object], timeout: float, label: str
) -> tuple[str, float, dict[str, object]]:
    client_id = str(uuid.uuid4())
    response = http_json("/prompt", "POST", {"prompt": nodes, "client_id": client_id})
    if not isinstance(response, dict) or "prompt_id" not in response:
        raise RuntimeError(f"invalid /prompt response: {response}")
    prompt_id = str(response["prompt_id"])
    started = time.monotonic()
    deadline = started + min(timeout, remaining(300))
    next_heartbeat = started
    print(f"submitted label={label} prompt_id={prompt_id}", flush=True)
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_heartbeat:
            print(
                f"heartbeat label={label} elapsed={now - started:.1f}s gpu={gpu_heartbeat()}",
                flush=True,
            )
            next_heartbeat = now + 15
        data = http_json(f"/history/{prompt_id}")
        if isinstance(data, dict) and prompt_id in data:
            entry = data[prompt_id]
            if not isinstance(entry, dict):
                raise TypeError("history entry was not an object")
            error = prompt_error(entry)
            if error:
                raise RuntimeError(f"ComfyUI execution error: {error}")
            return prompt_id, time.monotonic() - started, entry
        time.sleep(2)
    try:
        http_json("/interrupt", "POST", {})
    except Exception:
        pass
    raise TimeoutError(f"prompt {prompt_id} exceeded {timeout}s")


def copy_output(save_name: str) -> Path:
    candidates = sorted(
        glob.glob(str(COMFY / "output" / "h3-turbo" / f"{save_name}_*")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no saved output for {save_name}")
    source = Path(candidates[-1])
    target = MEDIA / f"{save_name}{source.suffix}"
    shutil.copy2(source, target)
    return target


def run_case(case: dict[str, object], steps: int) -> dict[str, object]:
    label = str(case["label"])
    width = int(case["width"])
    height = int(case["height"])
    length = int(case["length"])
    name = f"i2v_turbo_{label}_{steps}step"
    print(f"=== production {name} ===", flush=True)
    nodes = workflow(
        width=width,
        height=height,
        length=length,
        steps=steps,
        save_name=name,
    )
    write_json(RESULTS / "workflows" / f"{name}.json", nodes)
    timeout = 40 * 60 if length > 200 else 25 * 60
    prompt_id, seconds, history = submit_and_wait(nodes, timeout, name)
    media = copy_output(name)
    write_json(RESULTS / "history" / f"{name}.json", history)
    result = {
        "name": name,
        "steps": steps,
        "case": label,
        "width": width,
        "height": height,
        "frames": length,
        "fps": FPS,
        "video_seconds": round(length / FPS, 3),
        "seconds": round(seconds, 3),
        "seconds_per_step": round(seconds / steps, 3),
        "prompt_id": prompt_id,
        "media": str(media.relative_to(RESULTS)),
        "status": "success",
    }
    print(f"completed {name} seconds={seconds:.3f} media={media}", flush=True)
    return result


def ffprobe(path: Path) -> dict[str, object]:
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        return {"error": process.stderr.strip()}
    return json.loads(process.stdout)


def make_contact_sheet(run: dict[str, object]) -> None:
    from PIL import Image, ImageDraw

    name = str(run["name"])
    video = RESULTS / str(run["media"])
    width = int(run["width"])
    height = int(run["height"])
    duration = float(run["video_seconds"])
    frames: list[tuple[str, Path]] = [
        ("input", RESULTS / "assets" / asset_name(width, height))
    ]
    sample_seconds = (0.0, duration / 2.0, max(0.0, duration - 1.0))
    for index, second in enumerate(sample_seconds):
        label = f"{second:.1f}s"
        target = COMPARE / f"{name}_{second}s.png"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(second), "-i", str(video), "-frames:v", "1", str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if target.is_file():
            frames.append((label, target))
    thumbs: list[tuple[str, Image.Image]] = []
    for label, path in frames:
        image = Image.open(path).convert("RGB")
        image.thumbnail((448, 256), Image.Resampling.LANCZOS)
        thumbs.append((label, image.copy()))
    sheet = Image.new("RGB", (448 * len(thumbs), 292), "#111111")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(thumbs):
        x = index * 448
        sheet.paste(image, (x, 0))
        draw.text((x + 10, 264), label, fill="white")
    sheet.save(COMPARE / f"{name}_contact_sheet.jpg", quality=92)


def build_report(runs: list[dict[str, object]], failure: str | None) -> None:
    media_info = {}
    for run in runs:
        if run.get("status") == "success":
            media_info[str(run["name"])] = ffprobe(RESULTS / str(run["media"]))
    write_json(RESULTS / "media-info.json", media_info)
    lines = [
        "# MiniMax H3 Turbo production report",
        "",
        f"- Job: `{JOB_ID}`",
        f"- FPS: `{FPS}`",
        f"- Seed: `{SEED}`",
        f"- LoRA: `{LORA}` at strength `1.0`",
        "- SageAttention: required and verified",
        f"- Failure: `{failure or 'none'}`",
        "",
        "| case | canvas | frames | steps | total seconds | seconds / step | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in runs:
        lines.append(
            f"| {run.get('case')} | {run.get('width')}x{run.get('height')} | "
            f"{run.get('frames')} | {run.get('steps')} | {run.get('seconds', '-')} | "
            f"{run.get('seconds_per_step', '-')} | {run.get('status')} |"
        )
    lines.extend(
        [
            "",
            "Visual review must compare architectural line stability, cloud trailing, "
            "robe motion, fine texture, oversharpening, and audio integrity.",
        ]
    )
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_results() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(RESULTS, arcname=RESULTS.name)
    print(f"artifact={ARCHIVE} bytes={ARCHIVE.stat().st_size}", flush=True)


def main() -> None:
    for path in (RESULTS, MEDIA, LOGS, COMPARE, WORK):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, RESULTS / "job-config.json")
    server: subprocess.Popen[str] | None = None
    telemetry: subprocess.Popen[bytes] | None = None
    telemetry_file = None
    runs: list[dict[str, object]] = []
    failure: str | None = None
    fatal: BaseException | None = None
    try:
        stop_stale_comfy_server()
        hardware = preflight()
        print(
            f"preflight gpu={hardware['gpu']} vram={hardware['vram_gib']}GiB "
            f"torch={hardware['torch']} cuda={hardware['torch_cuda']}",
            flush=True,
        )
        telemetry_file = TELEMETRY.open("ab", buffering=0)
        telemetry = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv",
                "-l",
                "1",
            ],
            stdout=telemetry_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        models = setup()
        sage = install_and_verify_sage(hardware)
        prepared_assets = prepare_assets()
        server, _thread = start_server()
        write_json(
            RESULTS / "manifest.json",
            {
                "comfy_revision": COMFY_REV,
                "turbo_revision": TURBO_REV,
                "model_revision": MODEL_REV,
                "lora_revision": LORA_REV,
                "sage_revision": SAGE_REV,
                "kj_revision": KJ_REV,
                "sage": sage,
                "models": [str(path.relative_to(COMFY)) for path in models],
                "cases": list(CASES),
                "prepared_assets": [str(path.relative_to(COMFY)) for path in prepared_assets],
                "fps": FPS,
                "step_matrix": {
                    str(case["label"]): list(case["steps"]) for case in CASES
                },
                "seed": SEED,
                "prompt": PROMPT,
            },
        )

        if bool(CONFIG.get("warmup", False)):
            warmup = workflow(width=608, height=352, length=56, steps=4, save_name=None)
            write_json(RESULTS / "workflows" / "warmup.json", warmup)
            submit_and_wait(warmup, 12 * 60, "warmup")
            print("warmup complete", flush=True)
        print("starting parameterized production", flush=True)

        for case in CASES:
            for steps in case["steps"]:
                try:
                    run = run_case(case, steps)
                    runs.append(run)
                    try:
                        make_contact_sheet(run)
                    except Exception as sheet_error:
                        print(
                            f"contact-sheet warning for {run['name']}: {sheet_error!r}",
                            flush=True,
                        )
                except Exception as exc:
                    runs.append(
                        {
                            "name": f"i2v_turbo_{case['label']}_{steps}step",
                            "case": case["label"],
                            "width": case["width"],
                            "height": case["height"],
                            "frames": case["length"],
                            "steps": steps,
                            "status": "failed",
                            "error": repr(exc),
                        }
                    )
                    raise
    except BaseException as exc:
        fatal = exc
        failure = repr(exc)
        traceback.print_exc()
    finally:
        stop_server(server)
        if telemetry is not None and telemetry.poll() is None:
            try:
                os.killpg(telemetry.pid, signal.SIGTERM)
                telemetry.wait(timeout=10)
            except Exception:
                pass
        if telemetry_file is not None:
            telemetry_file.close()
        try:
            write_json(RESULTS / "runs.json", runs)
            build_report(runs, failure)
            cuda_log = Path("/content/cuda13-upgrade.log")
            if cuda_log.is_file():
                shutil.copy2(cuda_log, LOGS / cuda_log.name)
        finally:
            package_results()
    if fatal is not None:
        raise fatal


if __name__ == "__main__":
    main()
