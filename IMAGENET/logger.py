import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Write resumable per-epoch training metrics to a run-specific JSON file."""

    def __init__(
        self,
        save_dir: str,
        run_name: str,
        config: dict[str, Any],
        environment: dict[str, Any],
        dataset: dict[str, Any],
        model: dict[str, Any],
    ) -> None:
        self.save_dir = Path(save_dir)
        self.run_id = (
            f"{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        self.log_path = self.save_dir / f"log.json"
        self._start_time = time.perf_counter()
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_name": run_name,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "finished_at": None,
            "total_training_time_seconds": 0.0,
            "best_val_acc": None,
            "best_epoch": None,
            "config": config,
            "environment": environment,
            "dataset": dataset,
            "model": model,
            "epochs": [],
        }
        self._write()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        exc_type, exc_value = exc_info[:2]
        self.data["status"] = "failed" if exc_type is not None else "completed"
        self.data["finished_at"] = datetime.now().astimezone().isoformat()
        if exc_type is not None:
            self.data["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
            }
        self._write()
        return False

    def log_epoch(
        self,
        metrics: dict[str, Any],
        best_val_acc: float,
        best_epoch: int,
    ) -> None:
        self.data["epochs"].append(metrics)
        self.data["best_val_acc"] = best_val_acc
        self.data["best_epoch"] = best_epoch
        self._write()

    def _write(self) -> None:
        self.data["total_training_time_seconds"] = round(
            time.perf_counter() - self._start_time, 6
        )
        temporary_path = self.log_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as log_file:
            json.dump(self.data, log_file, indent=2)
            log_file.write("\n")
        os.replace(temporary_path, self.log_path)
