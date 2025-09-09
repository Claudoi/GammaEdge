from __future__ import annotations
import json
import os
import time
from datetime import datetime
from typing import Any, Dict

class JsonRunLogger:
    """
    Logger muy simple en formato JSONL (una línea por evento) para trazabilidad.
    """
    def __init__(self, log_dir: str = "logs", run_name: str = "run"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"{run_name}-{ts}.jsonl")
        self.t0 = time.time()

    def log(self, event: str, **payload: Dict[str, Any]):
        rec = {
            "event": event,
            "utc": datetime.utcnow().isoformat(),
            "elapsed_sec": round(time.time() - self.t0, 3),
            **payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec
