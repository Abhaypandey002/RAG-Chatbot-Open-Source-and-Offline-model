from pathlib import Path
import json, time

class JsonlLogger:
    def __init__(self, dir_path: str):
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"

    def log(self, kind: str, payload: dict):
        record = {
            "ts": time.time(),
            "kind": kind,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
