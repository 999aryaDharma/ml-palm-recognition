"""
Structured Trained Log untuk setiap PalmNet-Lite training run.

Setiap run memiliki:
    <trained_logs_dir>/<run_id>/
        run.json       - run metadata dan config
        history.csv    - epoch-level metrics
        notes.md       - manual notes
        phase1_summary.json
        phase2_summary.json
        validation_metrics.json
        metrics.json   - final metrics
        figures/       - training curves, ROC, distribution plots

run_id format: YYYYMMDD-HHMMSS-seed<seed>

Setiap run tidak boleh menimpa run sebelumnya.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class RunLogger:
    """Manages structured logging untuk satu training run.

    Membuat directory run baru dengan run_id unik.
    Tidak menimpa run sebelumnya.

    Usage:
        logger = RunLogger(base_dir="app/ml/artifacts/trained_logs/palmnet-lite-scratch",
                          seed=42)
        logger.init_run(config)
        # per epoch:
        logger.log_epoch(phase=1, epoch=1, metrics={...})
        # akhir phase:
        logger.save_phase_summary(phase=1, summary={...})
        logger.close(status="completed")
    """

    def __init__(self, base_dir: str | Path, seed: int = 42):
        self.base_dir = Path(base_dir)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{ts}-seed{seed}"
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        (self.run_dir / "tensorboard").mkdir(exist_ok=True)

        self._history_path = self.run_dir / "history.csv"
        self._run_json_path = self.run_dir / "run.json"
        self._notes_path = self.run_dir / "notes.md"
        self._csv_initialized = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @run_id.setter
    def run_id(self, value: str) -> None:
        self._run_id = value

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    def init_run(self, config: dict, extra: Optional[dict] = None) -> None:
        """Tulis run.json awal dengan status='running'."""
        run_data: dict[str, Any] = {
            "run_id": self.run_id,
            "model_id": "palmnet-lite-scratch",
            "architecture": "PalmNetLite",
            "architecture_version": "v1",
            "training_mode": "scratch",
            "timestamp": datetime.now().isoformat(),
            "status": "running",
            "config": config,
        }
        if extra:
            run_data.update(extra)
        self._write_json(self._run_json_path, run_data)
        self._write_notes(f"# Run {self.run_id}\n\n## Notes\n\n")

    def log_epoch(self, phase: int, epoch: int, metrics: dict) -> None:
        """Append satu baris ke history.csv.

        metrics dict boleh partial — field yang tidak ada diisi kosong.
        """
        columns = [
            "phase", "epoch", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy",
            "mean_positive_cosine", "mean_negative_cosine", "cosine_gap",
            "val_eer", "learning_rate", "arcface_margin", "epoch_duration_sec",
        ]

        row = {"phase": phase, "epoch": epoch}
        row.update(metrics)

        if not self._csv_initialized:
            with open(self._history_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
            self._csv_initialized = True

        with open(self._history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writerow(row)

    def save_phase_summary(self, phase: int, summary: dict) -> None:
        """Simpan ringkasan selesai satu phase."""
        fname = f"phase{phase}_summary.json"
        self._write_json(self.run_dir / fname, summary)

    def save_validation_metrics(self, metrics: dict) -> None:
        """Simpan calibration validation metrics."""
        self._write_json(self.run_dir / "validation_metrics.json", metrics)

    def save_final_metrics(self, metrics: dict) -> None:
        """Simpan final evaluation metrics."""
        self._write_json(self.run_dir / "metrics.json", metrics)

    def update_run_json(self, updates: dict) -> None:
        """Update field di run.json."""
        if self._run_json_path.exists():
            with open(self._run_json_path) as f:
                data = json.load(f)
        else:
            data = {}
        data.update(updates)
        self._write_json(self._run_json_path, data)

    def add_note(self, note: str) -> None:
        """Append note ke notes.md."""
        with open(self._notes_path, "a") as f:
            f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {note}\n")

    def close(
        self,
        status: str = "completed",
        failure_reason: Optional[str] = None,
    ) -> None:
        """Finalize run dengan status."""
        updates: dict[str, Any] = {
            "status": status,
            "completed_at": datetime.now().isoformat(),
        }
        if failure_reason:
            updates["failure_reason"] = failure_reason
        self.update_run_json(updates)

    def _write_json(self, path: Path, data: dict) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _write_notes(self, content: str) -> None:
        with open(self._notes_path, "w") as f:
            f.write(content)
