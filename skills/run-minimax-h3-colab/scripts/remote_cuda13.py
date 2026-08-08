from __future__ import annotations

import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import shutil


LOG = Path("/content/cuda13-upgrade.log")


def run(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    with LOG.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
        )
        assert process.stdout is not None
        started = time.monotonic()
        last_line = started
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                log.write(line)
                log.flush()
                print(line.rstrip(), flush=True)
                last_line = time.monotonic()
            else:
                now = time.monotonic()
                if now - started > timeout:
                    process.terminate()
                    raise TimeoutError("command timed out")
                if now - last_line > 20:
                    print(
                        f"upgrade heartbeat elapsed={now - started:.1f}s "
                        f"command={command[0]}",
                        flush=True,
                    )
                    last_line = now
                time.sleep(0.2)
        for line in process.stdout:
            log.write(line)
            print(line.rstrip(), flush=True)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def os_release() -> dict[str, str]:
    result = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value.strip().strip('"')
    return result


def ensure_cuda_toolkit() -> Path:
    target = Path("/usr/local/cuda-13.0")
    if target.joinpath("bin/nvcc").is_file():
        print("CUDA 13.0 toolkit already present", flush=True)
        return target

    run(["apt-get", "update"], timeout=600)
    policy = subprocess.run(
        ["apt-cache", "show", "cuda-toolkit-13-0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if policy.returncode != 0 or not policy.stdout.strip():
        release = os_release()
        version = re.sub(r"\D", "", release.get("VERSION_ID", ""))
        if release.get("ID") != "ubuntu" or version not in {"2204", "2404"}:
            raise RuntimeError(f"unsupported CUDA repository OS: {release}")
        repo = f"ubuntu{version}"
        keyring = Path("/tmp/cuda-keyring_1.1-1_all.deb")
        run(
            [
                "wget",
                "-q",
                "-O",
                str(keyring),
                f"https://developer.download.nvidia.com/compute/cuda/repos/{repo}/x86_64/cuda-keyring_1.1-1_all.deb",
            ],
            timeout=180,
        )
        run(["dpkg", "-i", str(keyring)], timeout=180)
        run(["apt-get", "update"], timeout=600)

    run(
        ["apt-get", "install", "-y", "--no-install-recommends", "cuda-toolkit-13-0"],
        timeout=2400,
    )
    if not target.joinpath("bin/nvcc").is_file():
        raise FileNotFoundError(target / "bin/nvcc")
    run(["ln", "-sfn", str(target), "/usr/local/cuda"], timeout=30)
    return target


def install_torch_cu130() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "torch==2.11.0+cu130",
            "torchvision==0.26.0+cu130",
            "torchaudio==2.11.0+cu130",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ],
        timeout=2400,
    )


def build_sage(cuda_home: Path) -> Path:
    source = Path("/content/h3-turbo-work/SageAttention-cu130")
    revision = "eb615cf6cf4d221338033340ee2de1c37fbdba4a"
    if not source.exists():
        run(
            ["git", "clone", "--filter=blob:none", "https://github.com/thu-ml/SageAttention.git", str(source)],
            timeout=300,
        )
    run(["git", "-C", str(source), "fetch", "origin", revision, "--depth", "1"], timeout=180)
    run(["git", "-C", str(source), "checkout", "--detach", revision], timeout=60)
    run([sys.executable, "-m", "pip", "uninstall", "-y", "sageattention"], timeout=120)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": str(cuda_home),
            "PATH": f"{cuda_home / 'bin'}:{env.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"{cuda_home / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}",
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "EXT_PARALLEL": "4",
            "NVCC_APPEND_FLAGS": "--threads 8",
            "MAX_JOBS": "16",
        }
    )
    run([sys.executable, "setup.py", "bdist_wheel"], timeout=1800, cwd=source, env=env)
    wheels = sorted(source.joinpath("dist").glob("sageattention-*.whl"))
    if not wheels:
        raise FileNotFoundError(source / "dist")
    wheel = wheels[-1]
    run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)],
        timeout=180,
        env=env,
    )
    artifact = Path("/content") / wheel.name.replace("linux_x86_64", "cu130_sm120_linux_x86_64")
    shutil.copy2(wheel, artifact)
    return artifact


def install_prebuilt_sage() -> Path | None:
    artifact = Path(
        "/content/sageattention-2.2.0-cp312-cp312-cu130_sm120_linux_x86_64.whl"
    )
    if not artifact.is_file():
        return None
    canonical = Path("/tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl")
    shutil.copy2(artifact, canonical)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(canonical),
        ],
        timeout=180,
    )
    print(f"reused prebuilt SageAttention wheel: {artifact}", flush=True)
    return artifact


def verify(cuda_home: Path, wheel: Path) -> None:
    run([str(cuda_home / "bin/nvcc"), "--version"], timeout=60)
    code = (
        "import json, torch, sageattention; "
        "import sageattention._qattn_sm80, sageattention._qattn_sm89; "
        "print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'capability':torch.cuda.get_device_capability(0),'sage':sageattention.__file__}))"
    )
    run([sys.executable, "-c", code], timeout=120)
    print(f"CUDA13_UPGRADE_COMPLETE wheel={wheel}", flush=True)


def main() -> None:
    print(
        f"upgrade host={platform.platform()} python={sys.version}",
        flush=True,
    )
    cuda_home = ensure_cuda_toolkit()
    install_torch_cu130()
    wheel = install_prebuilt_sage() or build_sage(cuda_home)
    verify(cuda_home, wheel)


if __name__ == "__main__":
    main()
