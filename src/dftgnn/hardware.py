"""Hardware record attached to every cluster result.

Showed that a spec reproduces bitwise on the same GPU model and differs between GPU models, so every result
records the GPU model, UUID, driver and CUDA versions, the Kubernetes node (``NODE_NAME`` from the downward
API) and the pod hostname. Adapted from the collaborator's ``scripts/nrp/n1_run.py`` (``hardware``); missing
tools give None, never an error.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess


def _smi(*args: str) -> str | None:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _cuda_driver_api_version() -> str | None:
    """CUDA version supported by the installed driver ("13.2"), from cuDriverGetVersion; None if unavailable.

    The fallback when the ``CUDA Version`` line of the plain ``nvidia-smi`` banner cannot be read.
    ``--query-gpu=driver_version`` gives the kernel driver (``driver``), not this.
    """
    import ctypes

    try:
        import torch

        v = int(torch._C._cuda_getDriverVersion())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        try:
            n = ctypes.c_int()
            if ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(n)) != 0:
                return None
            v = n.value
        except (OSError, AttributeError):
            return None
    return f"{v // 1000}.{v % 1000 // 10}" if v > 0 else None


def hardware_info(device=None) -> dict:
    """GPU model, UUID, driver, memory, CUDA versions and host names; GPU fields None on a CPU device."""
    import torch

    out = {"device": str(device) if device is not None else None, "gpu_model": None, "gpu_uuid": None,
           "driver": None, "gpu_memory": None, "cuda_driver_version": None, "torch_cuda": torch.version.cuda,
           "torch": torch.__version__, "node_hostname": os.environ.get("NODE_NAME"),
           "pod_hostname": socket.gethostname()}
    is_cuda = torch.cuda.is_available() and (device is None or str(device).startswith("cuda"))
    if not is_cuda:
        return out
    idx = torch.device(device).index if device is not None and torch.device(device).index is not None else 0
    out["gpu_model"] = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    uuid = getattr(props, "uuid", None)
    out["gpu_uuid"] = None if uuid is None else f"GPU-{uuid}"
    out["gpu_memory"] = f"{props.total_memory // 2**20} MiB"
    q = _smi("--query-gpu=driver_version", "--format=csv,noheader")
    out["driver"] = q.strip().splitlines()[0] if q and q.strip() else None
    smi = _smi()
    m = re.search(r"CUDA(?: Driver)? Version\s*:\s*([0-9.]+)", smi or "")
    out["cuda_driver_version"] = m.group(1) if m else _cuda_driver_api_version()
    return out
