import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class TrainingLogger:
    """Persist language-model training metrics and run metadata as JSON."""

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
            "schema_version": 2,
            "run_id": self.run_id,
            "run_name": run_name,
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
            "finished_at": None,
            "total_training_time_seconds": 0.0,
            "best_val_loss": None,
            "best_step": None,
            "config": config,
            "environment": environment,
            "dataset": dataset,
            "model": model,
            "training": [],
            "evaluations": [],
            "test": None,
        }
        self._write()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        exc_type, exc_value = exc_info[:2]
        if exc_type is not None:
            self.data["status"] = (
                "interrupted"
                if isinstance(exc_value, KeyboardInterrupt)
                else "failed"
            )
        elif self.data["status"] == "running":
            self.data["status"] = "completed"
        self.data["finished_at"] = datetime.now().astimezone().isoformat()
        if exc_type is not None:
            self.data["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_value),
            }
        self._write()
        return False

    def mark_interrupted(self) -> None:
        self.data["status"] = "interrupted"
        self._write()

    def log_training(self, metrics: dict[str, Any]) -> None:
        self.data["training"].append(metrics)
        self._write()

    def log_evaluation(
        self,
        metrics: dict[str, Any],
        best_val_loss: float,
        best_step: Optional[int],
    ) -> None:
        self.data["evaluations"].append(metrics)
        self.data["best_val_loss"] = best_val_loss
        self.data["best_step"] = best_step
        self._write()

    def log_test(self, metrics: dict[str, Any]) -> None:
        self.data["test"] = metrics
        self._write()

    def _write(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data["total_training_time_seconds"] = round(
            time.perf_counter() - self._start_time, 6
        )
        temporary_path = self.log_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as log_file:
            json.dump(self.data, log_file, indent=2)
            log_file.write("\n")
        os.replace(temporary_path, self.log_path)
