"""Deterministic seeding and environment/Git metadata capture."""

import os
import platform
import random
import subprocess
from typing import Any, Dict
import numpy as np
import torch


def seed_everything(seed: int, deterministic_cuda: bool = True) -> None:
    """Seed Python, NumPy, PyTorch CPU, and CUDA generators deterministically."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_torch(device: str, gpu_memory_fraction: float = 0.25) -> None:
    """One CPU thread per worker; cap VRAM so three workers fit on a 12 GB RTX 3060."""
    torch.set_num_threads(1)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)


def get_git_info(repo_dir: str = ".") -> Dict[str, Any]:
    """Capture current Git commit SHA, branch, and dirty flag safely."""
    def _run(args):
        try:
            res = subprocess.run(
                ["git", "-C", repo_dir, *args],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return res.stdout.strip() if res.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    sha = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": bool(status and status != "unknown"),
    }


def get_hardware_info() -> Dict[str, Any]:
    """Capture Python, PyTorch, CUDA, and CPU metadata."""
    cuda_ok = torch.cuda.is_available()
    return {
        "os": str(platform.platform()),
        "python_version": str(platform.python_version()),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(cuda_ok),
        "cuda_device": str(torch.cuda.get_device_name(0)) if cuda_ok else "cpu",
        "cpu_count": int(os.cpu_count() or 1),
    }
