from __future__ import annotations

import datetime as dt
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import re
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

MODE = str(CONFIG.get("mode", "generate"))
if MODE not in {"generate", "repaint"}:
    raise ValueError(f"unsupported mode: {MODE}")
ASSET_SOURCE = Path(str(CONFIG.get("remote_reference_image", "")))
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
SOL_REPO = "https://github.com/kijai/ComfyUI-SolAttn_triton.git"
SOL_REV = "842c4eaa7d91dbaef3fee3ccdbf36a39521e82fc"
PT_REPO = "https://github.com/ptmaster/ComfyUI-PT_H3ConcatAVLatent.git"
PT_REV = "2387e990a025f9b2f98ccac4ff1c466086e34a6e"
VHS_REPO = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
VHS_REV = "4ee72c065db22c9d96c2427954dc69e7b908444b"
MODEL_REPO = "Comfy-Org/MiniMax-H3"
MODEL_REV = "eb8a16107c595128b3a578f82d2ce2f75920c355"
LORA_REPO = "larryvrh/MiniMax-H3-Turbo-Lora"
LORA_REV = "afc0346516372a17162c14df3c5264de1d9aa1c0"

GENERATE_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
REPAINT_UNET = "minimax_h3_ref2va_int8_convrot.safetensors"
UNET = REPAINT_UNET if MODE == "repaint" else GENERATE_UNET
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
) if MODE == "generate" else ()

PROMPT = str(CONFIG.get("prompt", ""))
REPAINT_JOBS = tuple(CONFIG.get("jobs", ()))
REPAINT_STEPS = int(CONFIG.get("steps", 4))
REPAINT_DENOISE = float(CONFIG.get("denoise", 0.20))
NGC_ENV = Path(str(CONFIG.get("remote_ngc_env", "/content/.ngc.env")))
VFX_CORE_URL = (
    "https://api.ngc.nvidia.com/v2/org/nvidia/team/maxine/resources/"
    "vfx_sdk_core/versions/1.2.0.0_linux/files/VFXSDK_linux_1.2.0.0.tgz"
)
VFX_ROOT = Path("/usr/local/VideoFX")
NVIDIA_UPSCALE_LIB = VFX_ROOT / "features" / "nvvfxupscale" / "lib" / "libnvVFXUpscale.so"
NVIDIA_UPSCALE_NODE = COMFY / "custom_nodes" / "ComfyUI-NVIDIA-VFX-Upscale" / "__init__.py"


def nvidia_vfx_env() -> dict[str, str]:
    env = os.environ.copy()
    directories = [
        "/usr/local/VideoFX/features/nvvfxupscale/lib",
        "/usr/local/VideoFX/lib",
        "/usr/local/lib/python3.12/dist-packages/nvvfx/libs",
    ]
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(directories + ([current] if current else []))
    return env


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
    kj_nodes = COMFY / "custom_nodes" / "ComfyUI-KJNodes"
    checkout(KJ_REPO, kj_nodes, KJ_REV, "setup.log")
    if MODE == "generate":
        turbo = COMFY / "custom_nodes" / "ComfyUI-MiniMax-H3-Turbo"
        checkout(TURBO_REPO, turbo, TURBO_REV, "setup.log")
    else:
        sol_attn = COMFY / "custom_nodes" / "ComfyUI-SolAttn_triton"
        pt_concat = COMFY / "custom_nodes" / "ComfyUI-PT_H3ConcatAVLatent"
        vhs = COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"
        checkout(SOL_REPO, sol_attn, SOL_REV, "setup.log")
        checkout(PT_REPO, pt_concat, PT_REV, "setup.log")
        checkout(VHS_REPO, vhs, VHS_REV, "setup.log")
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
    if MODE == "repaint":
        run_logged(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(vhs / "requirements.txt")],
            "setup.log",
            600,
        )
        run_logged(
            [sys.executable, "-m", "pip", "install", "-q", "nvidia-vfx==0.1.0.1"],
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
if {MODE!r} == 'generate':
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
    ]
    if MODE == "generate":
        expected.append(COMFY / "models" / "loras" / LORA)
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size < 100_000_000]
    if missing:
        raise RuntimeError(f"model files missing or incomplete: {missing}")
    write_json(
        RESULTS / "models.json",
        {str(path.relative_to(COMFY)): path.stat().st_size for path in expected},
    )
    return expected


def ngc_key() -> str:
    if not NGC_ENV.is_file():
        raise FileNotFoundError(f"NGC environment file is missing: {NGC_ENV}")
    for line in NGC_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("NGC_CLI_API_KEY="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    raise RuntimeError("NGC_CLI_API_KEY is missing or empty")


def install_nvidia_upscale() -> dict[str, object]:
    key = ngc_key()
    installer = VFX_ROOT / "features" / "install_feature.sh"
    archive = WORK / "VFXSDK_linux_1.2.0.0.tgz"
    if not installer.is_file():
        print("downloading NVIDIA VFX SDK Core 1.2.0.0", flush=True)
        request = urllib.request.Request(
            VFX_CORE_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/octet-stream"},
        )
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as target:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        partial.replace(archive)
        run_logged(["tar", "-xzf", str(archive), "-C", "/usr/local"], "nvidia-upscale-install.log", 600)
    if not NVIDIA_UPSCALE_LIB.is_file():
        env = os.environ.copy()
        env["NGC_CLI_API_KEY"] = key
        run_logged(
            [str(installer), "-f", "nvvfxupscale", "-g", "b40", "-v", "1.2.0.0"],
            "nvidia-upscale-install.log",
            600,
            cwd=installer.parent,
            env=env,
        )
    if not NVIDIA_UPSCALE_LIB.is_file():
        raise RuntimeError(f"NVIDIA Upscale feature library was not installed: {NVIDIA_UPSCALE_LIB}")
    run_logged(["ldconfig"], "nvidia-upscale-install.log", 60)
    NVIDIA_UPSCALE_NODE.parent.mkdir(parents=True, exist_ok=True)
    NVIDIA_UPSCALE_NODE.write_text(
        '''import ctypes
from pathlib import Path

import torch

FEATURE = Path("/usr/local/VideoFX/features/nvvfxupscale/lib/libnvVFXUpscale.so")
ctypes.CDLL(str(FEATURE), mode=ctypes.RTLD_GLOBAL)
from nvvfx._ext import _Effect


class NVIDIAUpscale2x:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, strength):
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(f"expected NHWC IMAGE tensor, got {tuple(image.shape)}")
        batch, height, width, _channels = image.shape
        effect = _Effect("Upscale", 0)
        outputs = []
        try:
            effect.set_f32("Strength", float(strength))
            effect.set_output_image(int(width) * 2, int(height) * 2)
            effect.load()
            for index in range(int(batch)):
                frame = image[index, :, :, :3].permute(2, 0, 1).contiguous().to(device="cuda", dtype=torch.float32)
                effect.set_input_image(int(width), int(height))
                effect.transfer_input(frame)
                effect.run(False)
                output = torch.from_dlpack(effect.get_output(0)).clone()
                outputs.append(output.permute(1, 2, 0).contiguous().cpu())
        finally:
            effect.destroy()
        result = torch.stack(outputs, dim=0)
        print(f"NVIDIA VFX Upscale 2x: {batch} frames {width}x{height} -> {width * 2}x{height * 2}")
        return (result,)


NODE_CLASS_MAPPINGS = {"NVIDIAUpscale2x": NVIDIAUpscale2x}
NODE_DISPLAY_NAME_MAPPINGS = {"NVIDIAUpscale2x": "NVIDIA VFX Upscale 2x"}
''',
        encoding="utf-8",
    )
    result = {
        "sdk_core": "1.2.0.0_linux",
        "feature": "nvvfxupscale",
        "feature_version": "1.2.0.0",
        "gpu_target": "b40_sm120",
        "library": str(NVIDIA_UPSCALE_LIB),
        "library_bytes": NVIDIA_UPSCALE_LIB.stat().st_size,
        "library_sha256": sha256(NVIDIA_UPSCALE_LIB),
        "strength": 0.4,
        "post_resize": "lanczos_center_crop_1920x1088",
    }
    write_json(RESULTS / "nvidia-upscale.json", result)
    return result


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


def validate_repaint_object_info(info: dict[str, object]) -> None:
    expected_inputs = {
        "VHS_LoadVideoPath": {"video", "force_rate", "custom_width", "custom_height", "frame_load_cap", "skip_first_frames", "select_every_nth"},
        "MiniMaxH3ReferenceToVideo": {"clip", "vae", "audio_vae", "prompt", "width", "height", "length", "ref_image_size", "ref_videos", "ref_video_audios"},
        "ImageResizeKJv2": {"image", "width", "height", "upscale_method", "keep_proportion", "pad_color", "crop_position", "divisible_by"},
        "NVIDIAUpscale2x": {"image", "strength"},
        "VAEEncodeAudio": {"audio", "vae"},
        "PT_H3ConcatAVLatent": {"video_latent", "audio_latent"},
        "SolAttnPatch": {"model", "tau", "start_percent", "end_percent", "min_tokens", "int8_qk", "sink_conditioning", "morton", "morton_curve", "int8_pv", "verbose", "use_tma", "dense_blocks"},
    }
    for node, expected in expected_inputs.items():
        node_info = info.get(node, {})
        input_info = node_info.get("input", {}) if isinstance(node_info, dict) else {}
        available: set[str] = set()
        if isinstance(input_info, dict):
            for category in ("required", "optional", "hidden"):
                values = input_info.get(category, {})
                if isinstance(values, dict):
                    available.update(values)
        missing = sorted(expected - available)
        if missing:
            raise RuntimeError(f"{node} is missing expected inputs: {missing}")
    write_json(RESULTS / "object-info-repaint.json", {node: info[node] for node in sorted(expected_inputs)})


def wait_server(proc: subprocess.Popen[str], timeout: float = 240) -> dict[str, object]:
    generate_required = {
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
    repaint_required = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "VHS_LoadVideoPath",
        "MiniMaxH3ReferenceToVideo",
        "ImageResizeKJv2",
        "NVIDIAUpscale2x",
        "VAEEncode",
        "VAEEncodeAudio",
        "PT_H3ConcatAVLatent",
        "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "SolAttnPatch",
        "RandomNoise",
        "BasicGuider",
        "BasicScheduler",
        "KSamplerSelect",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "CreateVideo",
        "SaveVideo",
    }
    required = repaint_required if MODE == "repaint" else generate_required
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
            if MODE == "repaint":
                validate_repaint_object_info(info)
            return info
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(3)
    raise TimeoutError(f"ComfyUI did not become ready: {last_error}")


def verify_nvidia_upscale() -> None:
    code = WORK / "verify_nvidia_upscale.py"
    code.write_text(
        f"""import ctypes
import json
import torch
import torch.nn.functional as F
ctypes.CDLL({str(NVIDIA_UPSCALE_LIB)!r}, mode=ctypes.RTLD_GLOBAL)
from nvvfx._ext import _Effect

frame = torch.zeros((3, 640, 1152), device="cuda", dtype=torch.float32)
effect = _Effect("Upscale", 0)
try:
    effect.set_f32("Strength", 0.4)
    effect.set_output_image(2304, 1280)
    effect.load()
    effect.set_input_image(1152, 640)
    effect.transfer_input(frame)
    effect.run(False)
    upscaled = torch.from_dlpack(effect.get_output(0)).clone()
finally:
    effect.destroy()
output = F.interpolate(upscaled.unsqueeze(0), size=(1088, 1920), mode="bicubic", align_corners=False)[0]
if tuple(upscaled.shape) != (3, 1280, 2304) or tuple(output.shape) != (3, 1088, 1920):
    raise RuntimeError(f"unexpected NVIDIA Upscale output: {{tuple(upscaled.shape)}} -> {{tuple(output.shape)}}")
print(json.dumps({{"input": [3, 640, 1152], "nvidia_upscale_2x": list(upscaled.shape), "final": list(output.shape), "status": "success"}}))
""",
        encoding="utf-8",
    )
    run_logged(
        [sys.executable, str(code)],
        "nvidia-upscale-smoke.log",
        300,
        env=nvidia_vfx_env(),
    )


def verify_pt_concat() -> None:
    code = WORK / "verify_pt_concat.py"
    code.write_text(
        f"""import importlib.util
import json
import sys
import torch

sys.path.insert(0, {str(COMFY)!r})
node_file = {str(COMFY / 'custom_nodes' / 'ComfyUI-PT_H3ConcatAVLatent' / 'nodes.py')!r}
spec = importlib.util.spec_from_file_location('pt_h3_concat_nodes', node_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
video = {{'samples': torch.zeros((1, 24, 1, 2, 2))}}
audio = {{'samples': torch.zeros((1, 32, 2, 1))}}
merged = module.PT_H3ConcatAVLatent().merge(video, audio)[0]['samples']
print(json.dumps({{'type': type(merged).__name__, 'status': 'success'}}))
""",
        encoding="utf-8",
    )
    run_logged([sys.executable, str(code)], "pt-concat-smoke.log", 120)


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
        env=nvidia_vfx_env(),
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


def repaint_workflow(job: dict[str, object]) -> dict[str, object]:
    width = int(CONFIG["width"])
    height = int(CONFIG["height"])
    frames = int(job["frames"])
    identifier = str(job["id"])
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "5": {"class_type": "VHS_LoadVideoPath", "inputs": {
            "video": str(job["remote_video"]), "force_rate": FPS,
            "custom_width": 0, "custom_height": 0, "frame_load_cap": frames,
            "skip_first_frames": 0, "select_every_nth": 1,
        }},
        "6": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
            "prompt": str(job["prompt"]), "width": width, "height": height,
            "length": frames, "ref_image_size": "match",
            "ref_videos.ref_video_0": ["5", 0],
            "ref_video_audios.ref_video_audio_0": ["5", 2],
        }},
        "7": {"class_type": "NVIDIAUpscale2x", "inputs": {
            "image": ["5", 0], "strength": 0.4,
        }},
        "21": {"class_type": "ImageResizeKJv2", "inputs": {
            "image": ["7", 0], "width": width, "height": height,
            "upscale_method": "lanczos", "keep_proportion": "crop",
            "pad_color": "0, 0, 0", "crop_position": "center",
            "divisible_by": 32, "device": "cpu",
        }},
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["3", 0]}},
        "9": {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["5", 2], "vae": ["4", 0]}},
        "10": {"class_type": "PT_H3ConcatAVLatent", "inputs": {
            "video_latent": ["8", 0], "audio_latent": ["9", 0],
        }},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(job["seed"])}},
        "12": {"class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch", "inputs": {"model": ["1", 0]}},
        "13": {"class_type": "SolAttnPatch", "inputs": {
            "model": ["12", 0], "tau": 1.3, "start_percent": 0.2,
            "end_percent": 0.9, "min_tokens": 4096, "int8_qk": True,
            "sink_conditioning": "exact_kv", "morton": False,
            "morton_curve": "2d_frame", "int8_pv": True, "verbose": True,
            "use_tma": True, "dense_blocks": "33-35,39-42,-1",
        }},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["13", 0], "conditioning": ["6", 0]}},
        "15": {"class_type": "BasicScheduler", "inputs": {
            "model": ["13", 0], "scheduler": "beta", "steps": REPAINT_STEPS,
            "denoise": REPAINT_DENOISE,
        }},
        "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "17": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["14", 0], "sampler": ["16", 0],
            "sigmas": ["15", 0], "latent_image": ["10", 0],
        }},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["3", 0]}},
        "19": {"class_type": "CreateVideo", "inputs": {
            "images": ["18", 0], "audio": ["5", 2], "fps": FPS, "bit_depth": 8,
        }},
        "20": {"class_type": "SaveVideo", "inputs": {
            "video": ["19", 0], "filename_prefix": f"h3-repaint/repaint_{identifier}",
            "format": "auto", "codec": "auto",
        }},
    }


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


def copy_output(save_name: str, folder: str = "h3-turbo", target_name: str | None = None) -> Path:
    candidates = sorted(
        glob.glob(str(COMFY / "output" / folder / f"{save_name}_*")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"no saved output for {save_name}")
    source = Path(candidates[-1])
    target = MEDIA / f"{target_name or save_name}{source.suffix}"
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


def verify_media(path: Path, expected_frames: int) -> dict[str, object]:
    info = ffprobe(path)
    streams = info.get("streams", []) if isinstance(info, dict) else []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if not isinstance(video, dict) or not isinstance(audio, dict):
        raise RuntimeError(f"missing video or audio stream: {path}")
    if (int(video.get("width", 0)), int(video.get("height", 0))) != (
        int(CONFIG["width"]), int(CONFIG["height"]),
    ):
        raise RuntimeError(f"unexpected output size: {video.get('width')}x{video.get('height')}")
    if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
        raise RuntimeError(f"unexpected codecs: video={video.get('codec_name')} audio={audio.get('codec_name')}")
    if int(audio.get("channels", 0)) != 2:
        raise RuntimeError(f"output audio is not stereo: {audio.get('channels')}")
    decoded_frames = command_text([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(path),
    ]).strip()
    if int(decoded_frames) != expected_frames:
        raise RuntimeError(f"expected {expected_frames} decoded frames, got {decoded_frames}")
    run_logged(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
               f"decode-{path.stem}.log", 600)
    return info


def make_repaint_comparison(identifier: str, source: Path, output: Path) -> None:
    from PIL import Image, ImageDraw

    shot_compare = COMPARE / identifier
    shot_compare.mkdir(parents=True, exist_ok=True)
    run_logged(
        ["ffmpeg", "-y", "-i", str(source), "-i", str(output), "-lavfi",
         "[0:v]crop=1130:640:(iw-1130)/2:0[s];[1:v]scale=1130:640:flags=lanczos[o];[s][o]ssim=stats_file=" + str(shot_compare / "ssim.log"),
         "-f", "null", "-"],
        f"ssim-{identifier}.log", 600,
    )
    run_logged(
        ["ffmpeg", "-y", "-i", str(source), "-i", str(output), "-lavfi",
         "[0:v]crop=1130:640:(iw-1130)/2:0[s];[1:v]scale=1130:640:flags=lanczos[o];[s][o]psnr=stats_file=" + str(shot_compare / "psnr.log"),
         "-f", "null", "-"],
        f"psnr-{identifier}.log", 600,
    )
    rows: list[tuple[str, Image.Image, Image.Image]] = []
    for index, second in enumerate((0.0, 4.0, 7.0)):
        source_frame = shot_compare / f"source-{index}.png"
        output_frame = shot_compare / f"repaint-{index}.png"
        for video, frame in ((source, source_frame), (output, output_frame)):
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(second), "-i", str(video), "-frames:v", "1", str(frame)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
        source_image = Image.open(source_frame).convert("RGB")
        crop_width = round(source_image.height * 1920 / 1088)
        crop_left = (source_image.width - crop_width) // 2
        left = source_image.crop((crop_left, 0, crop_left + crop_width, source_image.height))
        left = left.resize((576, 326), Image.Resampling.LANCZOS)
        right = Image.open(output_frame).convert("RGB").resize((576, 326), Image.Resampling.LANCZOS)
        rows.append((f"{second:.1f}s", left, right))
    sheet = Image.new("RGB", (1152, 1050), "#111111")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 2), "source 1152x640", fill="white")
    draw.text((584, 2), "repaint 1920x1088", fill="white")
    for index, (label, left, right) in enumerate(rows):
        y = 40 + index * 336
        sheet.paste(left, (0, y))
        sheet.paste(right, (576, y))
        draw.text((8, y + 2), label, fill="yellow")
    sheet.save(shot_compare / "side-by-side.jpg", quality=92)


def recover_repaint_case(job: dict[str, object]) -> dict[str, object]:
    identifier = str(job["id"])
    source = Path(str(job["remote_video"]))
    output = MEDIA / f"repaint_{identifier}.mp4"
    history_path = RESULTS / "history" / f"{identifier}.json"
    telemetry_path = RESULTS / "telemetry" / f"{identifier}.csv"
    if not output.is_file() or not history_path.is_file() or not telemetry_path.is_file():
        raise RuntimeError(f"cannot recover {identifier}; required completed artifacts are missing")
    print(f"=== recovering completed repaint {identifier} without resampling ===", flush=True)
    input_info = ffprobe(source)
    output_info = verify_media(output, int(job["frames"]))
    write_json(RESULTS / "media-info" / f"{identifier}-input.json", input_info)
    write_json(RESULTS / "media-info" / f"{identifier}-output.json", output_info)
    make_repaint_comparison(identifier, source, output)
    peak_vram_mib = 0
    for line in telemetry_path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        try:
            peak_vram_mib = max(peak_vram_mib, int(line.split(",", 1)[0].strip()))
        except ValueError:
            pass
    log_text = SERVER_LOG.read_text(encoding="utf-8", errors="replace") if SERVER_LOG.is_file() else ""
    timings = re.findall(r"Prompt executed in ([0-9.]+) seconds", log_text)
    seconds = float(timings[-1]) if timings else 0.0
    result = {
        "name": identifier, "case": identifier, "steps": REPAINT_STEPS,
        "denoise": REPAINT_DENOISE, "width": int(CONFIG["width"]),
        "height": int(CONFIG["height"]), "frames": int(job["frames"]), "fps": FPS,
        "video_seconds": round(int(job["frames"]) / FPS, 3), "seconds": round(seconds, 3),
        "seconds_per_step": round(seconds / REPAINT_STEPS, 3), "prompt_id": "recovered",
        "peak_vram_mib": peak_vram_mib, "input": str(source), "input_sha256": sha256(source),
        "media": str(output.relative_to(RESULTS)), "output_sha256": sha256(output),
        "status": "success", "recovered_without_resampling": True,
    }
    shot_dir = RESULTS / "shots" / identifier
    shot_dir.mkdir(parents=True, exist_ok=True)
    for artifact in (
        RESULTS / "workflows" / f"{identifier}.json",
        history_path,
        RESULTS / "prompts" / f"{identifier}.txt",
        RESULTS / "media-info" / f"{identifier}-input.json",
        RESULTS / "media-info" / f"{identifier}-output.json",
        COMPARE / identifier / "side-by-side.jpg",
        COMPARE / identifier / "ssim.log",
        COMPARE / identifier / "psnr.log",
        telemetry_path,
        source,
        output,
    ):
        shutil.copy2(artifact, shot_dir / artifact.name)
    write_json(shot_dir / "benchmark.json", result)
    archive_dir = RESULTS / "shot-archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_dir / f"{identifier}.tar.gz", "w:gz") as archive:
        archive.add(shot_dir, arcname=identifier)
    print(f"recovered repaint {identifier} media={output}", flush=True)
    return result


def run_repaint_case(job: dict[str, object]) -> dict[str, object]:
    identifier = str(job["id"])
    source = Path(str(job["remote_video"]))
    print(f"=== repaint production {identifier} ===", flush=True)
    nodes = repaint_workflow(job)
    write_json(RESULTS / "workflows" / f"{identifier}.json", nodes)
    (RESULTS / "prompts" / f"{identifier}.txt").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "prompts" / f"{identifier}.txt").write_text(str(job["prompt"]) + "\n", encoding="utf-8")
    case_telemetry = RESULTS / "telemetry" / f"{identifier}.csv"
    case_telemetry.parent.mkdir(parents=True, exist_ok=True)
    telemetry_handle = case_telemetry.open("wb", buffering=0)
    telemetry_proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
         "--format=csv,nounits", "-l", "1"],
        stdout=telemetry_handle, stderr=subprocess.STDOUT, start_new_session=True,
    )
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        prompt_id, seconds, history = submit_and_wait(nodes, 60 * 60, identifier)
    finally:
        if telemetry_proc.poll() is None:
            os.killpg(telemetry_proc.pid, signal.SIGTERM)
            try:
                telemetry_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(telemetry_proc.pid, signal.SIGKILL)
        telemetry_handle.close()
    finished_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    peak_vram_mib = 0
    for line in case_telemetry.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
        try:
            peak_vram_mib = max(peak_vram_mib, int(line.split(",", 1)[0].strip()))
        except ValueError:
            pass
    output = copy_output(f"repaint_{identifier}", "h3-repaint", f"repaint_{identifier}")
    write_json(RESULTS / "history" / f"{identifier}.json", history)
    input_info = ffprobe(source)
    output_info = verify_media(output, int(job["frames"]))
    write_json(RESULTS / "media-info" / f"{identifier}-input.json", input_info)
    write_json(RESULTS / "media-info" / f"{identifier}-output.json", output_info)
    make_repaint_comparison(identifier, source, output)
    result = {
        "name": identifier, "case": identifier, "steps": REPAINT_STEPS,
        "denoise": REPAINT_DENOISE, "width": int(CONFIG["width"]),
        "height": int(CONFIG["height"]), "frames": int(job["frames"]), "fps": FPS,
        "video_seconds": round(int(job["frames"]) / FPS, 3), "seconds": round(seconds, 3),
        "seconds_per_step": round(seconds / REPAINT_STEPS, 3), "prompt_id": prompt_id,
        "started_utc": started_utc, "finished_utc": finished_utc,
        "peak_vram_mib": peak_vram_mib,
        "input": str(source), "input_sha256": sha256(source),
        "media": str(output.relative_to(RESULTS)), "output_sha256": sha256(output),
        "status": "success",
    }
    shot_dir = RESULTS / "shots" / identifier
    shot_dir.mkdir(parents=True, exist_ok=True)
    for artifact in (
        RESULTS / "workflows" / f"{identifier}.json",
        RESULTS / "history" / f"{identifier}.json",
        RESULTS / "prompts" / f"{identifier}.txt",
        RESULTS / "media-info" / f"{identifier}-input.json",
        RESULTS / "media-info" / f"{identifier}-output.json",
        COMPARE / identifier / "side-by-side.jpg",
        COMPARE / identifier / "ssim.log",
        COMPARE / identifier / "psnr.log",
        case_telemetry,
        source,
        output,
    ):
        shutil.copy2(artifact, shot_dir / artifact.name)
    write_json(shot_dir / "benchmark.json", result)
    archive_dir = RESULTS / "shot-archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_dir / f"{identifier}.tar.gz", "w:gz") as archive:
        archive.add(shot_dir, arcname=identifier)
    print(f"completed repaint {identifier} seconds={seconds:.3f} media={output}", flush=True)
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
        "# MiniMax H3 Base repaint report" if MODE == "repaint" else "# MiniMax H3 Turbo production report",
        "",
        f"- Job: `{JOB_ID}`",
        f"- FPS: `{FPS}`",
        f"- Seed: `{SEED}`",
        f"- Mode: `{MODE}`",
        "- Turbo LoRA: `disabled`" if MODE == "repaint" else f"- LoRA: `{LORA}` at strength `1.0`",
        f"- Scheduler: `beta`; sampler: `res_multistep`; denoise: `{REPAINT_DENOISE}`" if MODE == "repaint" else "- Turbo sampler enabled",
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


def write_benchmarks(runs: list[dict[str, object]]) -> None:
    write_json(RESULTS / "benchmark.json", {"job_id": JOB_ID, "mode": MODE, "runs": runs})
    columns = [
        "name", "status", "width", "height", "frames", "fps", "steps", "denoise",
        "seconds", "seconds_per_step", "peak_vram_mib", "input_sha256", "output_sha256", "media",
    ]
    with (RESULTS / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


def preserve_important_logs() -> None:
    destination = RESULTS / "important-logs"
    destination.mkdir(parents=True, exist_ok=True)
    for path in LOGS.glob("*"):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    for path in (TELEMETRY, RESULTS / "hardware.json", RESULTS / "environment-lock.json"):
        if path.is_file():
            shutil.copy2(path, destination / path.name)


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
        prepared_assets = prepare_assets() if MODE == "generate" else []
        nvidia_upscale = None
        if MODE == "repaint":
            nvidia_upscale = install_nvidia_upscale()
            verify_nvidia_upscale()
            verify_pt_concat()
        server, _thread = start_server()
        model_locks = {
            str(path.relative_to(COMFY)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in models
        }
        environment_lock = {
            "mode": MODE,
            "hardware": hardware,
            "python": sys.version,
            "repositories": {
                "ComfyUI": COMFY_REV,
                "ComfyUI-KJNodes": KJ_REV,
                "SageAttention": SAGE_REV,
                **({"ComfyUI-MiniMax-H3-Turbo": TURBO_REV} if MODE == "generate" else {
                    "ComfyUI-SolAttn_triton": SOL_REV,
                    "ComfyUI-PT_H3ConcatAVLatent": PT_REV,
                    "ComfyUI-VideoHelperSuite": VHS_REV,
                }),
            },
            "model_repo": {"id": MODEL_REPO, "revision": MODEL_REV},
            "models": model_locks,
            "sage": sage,
            "nvidia_vfx": "0.1.0.1" if MODE == "repaint" else None,
            "nvidia_upscale": nvidia_upscale,
        }
        write_json(RESULTS / "environment-lock.json", environment_lock)
        write_json(
            RESULTS / "manifest.json",
            {
                "mode": MODE,
                "comfy_revision": COMFY_REV,
                "turbo_revision": TURBO_REV if MODE == "generate" else None,
                "model_revision": MODEL_REV,
                "lora_revision": LORA_REV if MODE == "generate" else None,
                "sage_revision": SAGE_REV,
                "kj_revision": KJ_REV,
                "sol_revision": SOL_REV if MODE == "repaint" else None,
                "pt_concat_revision": PT_REV if MODE == "repaint" else None,
                "vhs_revision": VHS_REV if MODE == "repaint" else None,
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
                "repaint_jobs": list(REPAINT_JOBS),
                "repaint_steps": REPAINT_STEPS if MODE == "repaint" else None,
                "repaint_denoise": REPAINT_DENOISE if MODE == "repaint" else None,
                "nvidia_upscale": nvidia_upscale,
            },
        )

        if MODE == "generate" and bool(CONFIG.get("warmup", False)):
            warmup = workflow(width=608, height=352, length=56, steps=4, save_name=None)
            write_json(RESULTS / "workflows" / "warmup.json", warmup)
            submit_and_wait(warmup, 12 * 60, "warmup")
            print("warmup complete", flush=True)
        print(f"starting parameterized production mode={MODE}", flush=True)

        if MODE == "repaint":
            for job in REPAINT_JOBS:
                try:
                    if str(job.get("id")) in set(CONFIG.get("resume_completed_ids", ())):
                        runs.append(recover_repaint_case(job))
                    else:
                        runs.append(run_repaint_case(job))
                except Exception as exc:
                    runs.append({
                        "name": str(job.get("id", "unknown")), "case": str(job.get("id", "unknown")),
                        "width": int(CONFIG["width"]), "height": int(CONFIG["height"]),
                        "frames": int(job.get("frames", 0)), "steps": REPAINT_STEPS,
                        "denoise": REPAINT_DENOISE, "status": "failed", "error": repr(exc),
                    })
                    raise
        else:
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
        (RESULTS / "FAILED_INSTANCE_PRESERVED").write_text(failure + "\n", encoding="utf-8")
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
            if fatal is None:
                (RESULTS / "FAILED_INSTANCE_PRESERVED").unlink(missing_ok=True)
            write_json(RESULTS / "runs.json", runs)
            write_benchmarks(runs)
            build_report(runs, failure)
            cuda_log = Path("/content/cuda13-upgrade.log")
            if cuda_log.is_file():
                shutil.copy2(cuda_log, LOGS / cuda_log.name)
            preserve_important_logs()
        finally:
            package_results()
    if fatal is not None:
        raise fatal


if __name__ == "__main__":
    main()
