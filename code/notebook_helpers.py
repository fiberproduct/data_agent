from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List


def run_pipeline(script_path: Path, data_path: Path, target: str, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(script_path),
        "--data_path",
        str(data_path),
        "--target_column",
        target,
        "--output_dir",
        str(output_dir),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": str(proc.returncode),
    }


def load_result(result_json_path: Path) -> Dict:
    return json.loads(result_json_path.read_text(encoding="utf-8"))


def extract_training_logs(logs: List[str]) -> List[str]:
    keys = ["Agent B", "SVM", "XGBoost", "MLP", "Planner", "Cleaning decision"]
    return [x for x in logs if any(k in x for k in keys)]


def model_metrics_table(result: Dict):
    try:
        import pandas as pd
    except Exception:
        return (
            result.get("模型指标")
            or result.get("模型评估")
            or result.get("model_metrics", [])
        )
    rows = (
        result.get("模型指标")
        or result.get("模型评估")
        or result.get("model_metrics", [])
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = [c for c in ["模型", "auc", "precision", "recall", "f1", "accuracy"] if c in df.columns]
    if cols:
        df = df[cols]
    if "f1" in df.columns:
        df = df.sort_values("f1", ascending=False)
    return df
