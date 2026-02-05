from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import tempfile


ROOT = Path(__file__).resolve().parent
PRED_DIR = Path(ROOT / "predictions")
BETS_DIR = Path(ROOT / "suggested_bets")
EVAL_DIR = Path(ROOT / "eval_outputs")
DASH_DIR = Path(ROOT / "dashboard_data")
LATEST = DASH_DIR / "latest"
HISTORY = DASH_DIR / "history"

EVAL_FILES = {
    "full_bet_eval": "full_bet_eval.csv",
    "betting_eval_summary": "betting_eval_summary.csv",
    "prediction_eval_summary": "prediction_eval_summary.csv",
}

def newest_csv(dir_path: Path) -> Path:
    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dir_path}")
    latest_file = csv_files[-1]
    return latest_file

def atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=out_path.parent, suffix=".parquet") as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    
def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Starting export of dashboard parquets...")
    
    pred_file = newest_csv(PRED_DIR)
    bets_file = newest_csv(BETS_DIR)

    run_date = datetime.now().date().isoformat()
    
    hist_dir = HISTORY / run_date
    hist_dir.mkdir(parents=True, exist_ok=True)
    
    latest_dir = LATEST
    latest_dir.mkdir(parents=True, exist_ok=True)
    
    df_pred = pd.read_csv(pred_file)
    df_bets = pd.read_csv(bets_file)

    
    atomic_write_parquet(df_pred, latest_dir / "predictions.parquet")
    atomic_write_parquet(df_bets, latest_dir / "suggested_bets.parquet")
    
    atomic_write_parquet(df_pred, hist_dir / "predictions.parquet")
    atomic_write_parquet(df_bets, hist_dir / "suggested_bets.parquet")
    
    eval_written = {}
    for key, filename in EVAL_FILES.items():
        source = EVAL_DIR / filename
        if not source.exists():
            print(f"[{ts}] Warning: Evaluation file {filename} not found in {EVAL_DIR}, skipping.")
            continue
        df_eval_part = pd.read_csv(source)
        atomic_write_parquet(df_eval_part, latest_dir / f"{key}.parquet")
        atomic_write_parquet(df_eval_part, hist_dir / f"{key}.parquet")
        eval_written[key] = filename
    
    meta_data = {
        "run_date": run_date,
        "pred_file": pred_file.name,
        "bets_file": bets_file.name,
        "eval_files": eval_written,
        "export_timestamp": ts,
    }
    
    meta_latest_path = latest_dir / "metadata.json"
    meta_hist_path = hist_dir / "metadata.json"
    
    meta_latest_path.write_text(json.dumps(meta_data, indent=2))
    meta_hist_path.write_text(json.dumps(meta_data, indent=2))
    
    print(f"[{ts}] Export completed successfully.")
    
if __name__ == "__main__":
    main()