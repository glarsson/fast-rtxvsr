"""Runtime provisioning and capability probing for the standalone VSR venv.

The VSR worker (nvidia-vfx VideoSuperRes) needs an interpreter that has CUDA
torch plus NVIDIA's Video Effects SDK installed. ``fast-rtxvsr setup`` creates
a dedicated venv under this repo (torch cu130 + ``nvidia-vfx`` from NVIDIA's
index + ``PyNvVideoCodec`` + the CUDA 12 runtime DLL) so the model runtime
never has to be installed into a shared Python. You can also point
``--python`` / ``FAST_RTXVSR_PYTHON`` at any interpreter that already has the
stack - for example a ComfyUI venv that stages the same wheels.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

NVIDIA_INDEX = "https://pypi.nvidia.com"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_VERSION = "2.13.0"

_PROBE_CODE = (
    "import json, os, sys, torch\n"
    "from pathlib import Path\n"
    "if os.name == 'nt' and hasattr(os, 'add_dll_directory'):\n"
    "    for root in list(map(Path, sys.path)):\n"
    "        for rel in ('PyNvVideoCodec', 'nvidia/cuda_runtime/bin'):\n"
    "            folder = root / rel\n"
    "            if folder.is_dir():\n"
    "                os.add_dll_directory(str(folder))\n"
    "    cuda_bin = Path(os.environ.get('CUDA_PATH') or '') / 'bin'\n"
    "    if cuda_bin.is_dir():\n"
    "        os.add_dll_directory(str(cuda_bin))\n"
    "out={'cuda_available': torch.cuda.is_available(), 'nvvfx_ok': False, "
    "'pyav_ok': False, 'pynvvideocodec_ok': False}\n"
    "out['cuda_index']=0 if out['cuda_available'] else None\n"
    "out['device']=torch.cuda.get_device_name(0) if out['cuda_available'] else None\n"
    "try:\n"
    "    import nvvfx  # noqa: F401\n"
    "    out['nvvfx_ok']=True\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    import av  # noqa: F401\n"
    "    out['pyav_ok']=True\n"
    "except Exception:\n"
    "    pass\n"
    "try:\n"
    "    import PyNvVideoCodec  # noqa: F401\n"
    "    out['pynvvideocodec_ok']=True\n"
    "except Exception:\n"
    "    pass\n"
    "print(json.dumps(out))\n"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_venv_dir() -> Path:
    return repo_root() / ".venv"


def default_python() -> Path:
    return default_venv_dir() / "Scripts" / "python.exe"


def resolve_python(explicit: str | None) -> Path:
    """Resolve the VSR worker interpreter: --python, FAST_RTXVSR_PYTHON, default."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("FAST_RTXVSR_PYTHON")
    if env:
        return Path(env).expanduser().resolve()
    return default_python()


def probe_python(python: Path) -> dict:
    """Check whether an interpreter can see CUDA + nvidia-vfx + an I/O path."""
    info: dict = {
        "python": str(python),
        "python_exists": python.exists(),
        "cuda_available": False,
        "device": None,
        "cuda_index": None,
        "nvvfx_ok": False,
        "pyav_ok": False,
        "pynvvideocodec_ok": False,
        "healthy": False,
        "error": None,
    }
    if not python.exists():
        info["error"] = f"missing python: {python}"
        return info
    result = subprocess.run(
        [str(python), "-c", _PROBE_CODE],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        info["error"] = (result.stderr or result.stdout or "probe failed")[-500:]
        return info
    try:
        payload = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        info["error"] = "invalid probe output"
        return info
    info.update(payload)
    io_ok = bool(info.get("pyav_ok") or info.get("pynvvideocodec_ok"))
    info["healthy"] = bool(
        info["python_exists"]
        and info["cuda_available"]
        and info["nvvfx_ok"]
        and io_ok
    )
    if not info["healthy"] and not info.get("error"):
        if not info.get("cuda_available"):
            info["error"] = "CUDA unavailable in this Python"
        elif not info.get("nvvfx_ok"):
            info["error"] = "nvidia-vfx not importable"
        elif not io_ok:
            info["error"] = "need PyNvVideoCodec (GPU I/O) or PyAV (host fallback)"
    return info


def _pip(python: Path, args: list[str]) -> bool:
    result = subprocess.run(
        [str(python), "-m", "pip", "install", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print((result.stderr or result.stdout).strip()[-600:])
    return result.returncode == 0


def provision_venv() -> Path:
    """Create the repo venv and install the VSR stack. Returns the python."""
    venv_dir = default_venv_dir()
    print(f"[fast-rtxvsr] creating venv at {venv_dir}")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"venv creation failed: {(result.stderr or result.stdout)[-400:]}"
        )
    python = default_python()
    print(f"[fast-rtxvsr] installing torch {TORCH_VERSION} + cu130 "
          "(downloads ~3 GB, one-time) ...")
    if not _pip(python, ["--index-url", TORCH_INDEX, f"torch=={TORCH_VERSION}",
                         "torchvision", "torchaudio"]):
        raise RuntimeError("torch install failed")
    print("[fast-rtxvsr] installing PyNvVideoCodec + CUDA runtime DLL + av ...")
    if not _pip(python, ["pynvvideocodec", "nvidia-cuda-runtime-cu12", "av", "pillow"]):
        raise RuntimeError("PyNvVideoCodec / av install failed")
    print("[fast-rtxvsr] installing nvidia-vfx (Video Effects SDK wheel, ~490 MB) "
          "from NVIDIA's index ...")
    if not _pip(python, ["--extra-index-url", NVIDIA_INDEX, "nvidia-vfx"]):
        raise RuntimeError("nvidia-vfx install failed")
    return python


def ensure_worker_python(explicit: str | None) -> Path:
    """Return a worker python that is ready, provisioning the venv if needed.

    An explicit --python / FAST_RTXVSR_PYTHON is used as-is (verified by
    probe); otherwise the repo venv is created and installed on first use.
    """
    python = resolve_python(explicit)
    if explicit or os.environ.get("FAST_RTXVSR_PYTHON"):
        info = probe_python(python)
        if not info["healthy"]:
            raise RuntimeError(
                f"{python} is not ready for RTX VSR: {info.get('error')}"
            )
        return python
    if python.is_file():
        info = probe_python(python)
        if info["healthy"]:
            return python
        print(f"[fast-rtxvsr] existing venv at {python} is incomplete "
              f"({info.get('error')}); fixing")
    return provision_venv()
