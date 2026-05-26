from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_commit_hash(project_root: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def collect_system_info(seed: int = 42, project_root: str | Path = ".") -> dict[str, Any]:
    info: dict[str, Any] = {
        "random_seed": seed,
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_commit_hash": get_git_commit_hash(project_root),
        "pid": os.getpid(),
    }
    try:
        import torch

        info.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            }
        )
    except ImportError:
        info.update(
            {
                "torch_version": None,
                "cuda_version": None,
                "cudnn_version": None,
                "cuda_available": False,
                "gpu_name": None,
                "gpu_count": 0,
                "cudnn_deterministic": None,
                "cudnn_benchmark": None,
            }
        )
    return info


def save_system_info(
    output_path: str | Path = "artifacts/system_info.json",
    seed: int = 42,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    info = collect_system_info(seed=seed, project_root=project_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def configure_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    system_info_path: str | Path = "artifacts/system_info.json",
    project_root: str | Path = ".",
) -> dict[str, Any]:
    set_seed(seed=seed, deterministic=deterministic)
    return save_system_info(output_path=system_info_path, seed=seed, project_root=project_root)
