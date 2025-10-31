# tests/quick/test_export_cli.py
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys


def test_export_cli_smoke(tmp_path):
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    results.mkdir()
    reports.mkdir()

    (results / "bt.json").write_text(
        json.dumps(
            {
                "dates": ["2024-01-01", "2024-01-02"],
                "equity": [1.0, 1.002],
                "tickers": ["A", "B"],
                # 1 fila (K=1) -> el script debe expandir a T=2 automáticamente
                "weights": [[0.5, 0.5]],
            }
        )
    )
    with open(results / "returns_wide.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["date", "A", "B"])
        wr.writerow(["2024-01-01", 0.001, 0.001])
        wr.writerow(["2024-01-02", 0.0, 0.0])

    (results / "group_map.json").write_text(json.dumps({"A": "G1", "B": "G2"}))
    with open(results / "metrics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Metric", "Value"])
        wr.writerow(["Sharpe", 1.1])

    cmd = [
        sys.executable,
        "tools/export_report.py",
        "--bt",
        str(results / "bt.json"),
        "--returns",
        str(results / "returns_wide.csv"),
        "--groups",
        str(results / "group_map.json"),
        "--metrics",
        str(results / "metrics.csv"),
        "--out",
        str(reports),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run(cmd, check=False, env=env)
    assert (reports / "backtest_report.html").exists()
