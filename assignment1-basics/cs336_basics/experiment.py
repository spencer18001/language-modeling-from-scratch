from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLogger:
    def __init__(self, output_dir: str | Path, run_name: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_name = run_name or timestamp
        self.config_path = self.output_dir / "config.json"
        self.metrics_jsonl_path = self.output_dir / "metrics.jsonl"
        self.metrics_csv_path = self.output_dir / "metrics.csv"
        self.experiment_log_path = self.output_dir / "experiment_log.md"

    def write_config(self, config: dict[str, Any]) -> None:
        self.config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def start_run(
        self,
        config: dict[str, Any],
        command: str | None = None,
        notes: str | None = None,
    ) -> None:
        if not self.experiment_log_path.exists():
            self.experiment_log_path.write_text("# Experiment Log\n\n", encoding="utf-8")

        lines = [
            f"## {self.run_name}",
            "",
            f"- started_at_utc: {datetime.now(timezone.utc).isoformat()}",
        ]
        if command is not None:
            lines.append(f"- command: `{command}`")
        if notes:
            lines.append(f"- notes: {notes}")
        lines.extend(
            [
                f"- train_data: `{config.get('train_data')}`",
                f"- valid_data: `{config.get('valid_data')}`",
                f"- output_dir: `{config.get('output_dir')}`",
                f"- model: vocab_size={config.get('vocab_size')}, context_length={config.get('context_length')}, "
                f"d_model={config.get('d_model')}, num_layers={config.get('num_layers')}, "
                f"num_heads={config.get('num_heads')}, d_ff={config.get('d_ff')}",
                f"- optimizer: max_lr={config.get('max_lr')}, min_lr={config.get('min_lr')}, "
                f"warmup_iters={config.get('warmup_iters')}, betas=({config.get('beta1')}, {config.get('beta2')}), "
                f"eps={config.get('eps')}, weight_decay={config.get('weight_decay')}",
                "",
                "| iteration | tokens | train_loss | valid_loss | lr | elapsed_seconds |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        self._append_markdown(lines)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics, sort_keys=True, default=str) + "\n")

        write_header = not self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

        self._append_markdown(
            [
                "| {iteration} | {tokens_processed} | {train_loss} | {valid_loss} | {lr} | {elapsed_seconds} |".format(
                    iteration=metrics.get("iteration"),
                    tokens_processed=metrics.get("tokens_processed"),
                    train_loss=self._format_metric(metrics.get("train_loss")),
                    valid_loss=self._format_metric(metrics.get("valid_loss")),
                    lr=self._format_metric(metrics.get("lr")),
                    elapsed_seconds=self._format_metric(metrics.get("elapsed_seconds")),
                )
            ]
        )

    def log_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {
            "event": event,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
        }
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _append_markdown(self, lines: list[str]) -> None:
        with self.experiment_log_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))
            file.write("\n")

    @staticmethod
    def _format_metric(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
