"""Unified TensorBoard + JSONL + YAML experiment logger."""

import json
import os
from pathlib import Path
from typing import Any, Dict
from torch.utils.tensorboard import SummaryWriter
import yaml

from benchmarks.utils.seeding import get_git_info, get_hardware_info


class ExperimentLogger:
    """Writes resolved config.yaml, metrics.jsonl, TensorBoard events, and checkpoints."""

    def __init__(self, run_dir: str, config: Dict[str, Any]):
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.tb_dir = self.run_dir / "tb"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        full_meta = {
            **config,
            "git": get_git_info("."),
            "hardware": get_hardware_info(),
        }
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(full_meta, f, sort_keys=False)

        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._metrics_file = open(self.metrics_path, "a", encoding="utf-8")
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        record = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
        self._metrics_file.write(json.dumps(record) + "\n")
        self._metrics_file.flush()
        for key, val in metrics.items():
            self.writer.add_scalar(key, float(val), global_step=int(step))

    def close(self) -> None:
        if not self._metrics_file.closed:
            self._metrics_file.close()
        self.writer.close()
