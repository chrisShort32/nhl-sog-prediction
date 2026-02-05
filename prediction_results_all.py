from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    PRED_DIR = Path(ROOT / "predictions")
    OUT = Path(ROOT / "eval_outputs")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Starting tracking of prediction results...")
    
    
    update_box_df = pd.read_csv(ROOT / "data_collection/update_box.csv")
    update_pbp_df = pd.read_csv(ROOT / "data_collection/update_pbp.csv")

    # Merge PBP and box
    df = pd.merge(
        update_box_df,
        update_pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )

    pred_files = list(PRED_DIR.glob("preds_*.csv"))
    pred_dfs = [pd.read_csv(f) for f in pred_files]
    predictions = pd.concat(pred_dfs, ignore_index=True)
    
    actuals = (
        df[["game_id", "player_id", "player_name", "team", "shots_on_goal"]]
        .copy().rename(columns={"shots_on_goal": "actual_sog"})
    )

    # Normalize ids/types 
    actuals["game_id"] = pd.to_numeric(actuals["game_id"], errors="coerce").astype("Int64")
    predictions["game_id"] = pd.to_numeric(predictions["game_id"], errors="coerce").astype("Int64")

    pred_eval = predictions.merge(
        actuals[["game_id", "player_id", "player_name", "team", "actual_sog"]],
        on=["game_id", "player_id", "player_name", "team"],
        how="left",
    )
    
    # Turn p_ge2..p_ge5 into rows
    prob_cols = ["p_ge2", "p_ge3", "p_ge4", "p_ge5"]

    long = pred_eval.melt(
        id_vars=["game_id", "player_id", "player_name", "team", "opponent", "is_home", "actual_sog"],
        value_vars=prob_cols,
        var_name="prob_col",
        value_name="p_over",
    )

    # Extract threshold integer from "p_ge3" -> 3
    long["threshold"] = long["prob_col"].str.extract(r"(\d+)").astype(int)

    # Actual outcome for that threshold (over is ">= threshold")
    long["y"] = (long["actual_sog"] >= long["threshold"]).astype(int)


    eval_long = long[long["actual_sog"].notna()].copy()
    
    run_ts = pd.Timestamp.now('UTC').strftime("%Y-%m-%d %H:%M:%S UTC")
    run_date = pd.Timestamp.today().date()
    
    cutoff = 0.54
    eval_long["pred_y"] = (eval_long["p_over"] >= cutoff).astype(int)
    eval_long["err_sq"] = (eval_long["p_over"] - eval_long["y"]) ** 2
    
    # -----------------------------
    # 1) Overall metrics
    # -----------------------------
    overall = pd.DataFrame([{
        "run_date": run_date,
        "run_ts": run_ts,
        "section": "overall_metrics",
        "threshold": "ALL",
        "n": int(len(eval_long)),
        "cutoff": cutoff,
        "accuracy": float((eval_long["pred_y"] == eval_long["y"]).mean()),
        "brier": float(eval_long["err_sq"].mean()),
        "hit_rate": float(eval_long["y"].mean()),
        "avg_p": float(eval_long["p_over"].mean()),
    }])
    
    
    # -----------------------------
    # 2) Metrics by threshold
    # -----------------------------
    by_thr = (
        eval_long.groupby("threshold", as_index=False)
        .agg(
            n=("y", "size"),
            accuracy=("pred_y", lambda s: float((s == eval_long.loc[s.index, "y"]).mean())),
            brier=("err_sq", "mean"),
            hit_rate=("y", "mean"),
            avg_p=("p_over", "mean"),
        )
    )
    by_thr.insert(0, "section", "by_threshold_metrics")
    by_thr.insert(0, "run_ts", run_ts)
    by_thr.insert(0, "run_date", run_date)
    by_thr["cutoff"] = cutoff
    
    # -----------------------------
    # 3) Wide-bin calibration table
    # -----------------------------
    bins = [0, 0.2, 0.55, 0.6, 0.8, 1.0]
    eval_long["p_bin_wide"] = pd.cut(eval_long["p_over"], bins=bins, include_lowest=True)

    calib_wide = (
        eval_long.groupby(["threshold", "p_bin_wide"], as_index=False)
        .agg(
            n=("y", "size"),
            avg_p=("p_over", "mean"),
            hit_rate=("y", "mean"),
        )
    )
    calib_wide.insert(0, "section", "calibration_wide_bins")
    calib_wide.insert(0, "run_ts", run_ts)
    calib_wide.insert(0, "run_date", run_date)
    calib_wide["p_bin"] = calib_wide["p_bin_wide"].astype(str)
    calib_wide = calib_wide.drop(columns=["p_bin_wide"])

    
    # -----------------------------
    # 4) Quantile calibration (per threshold)
    # -----------------------------
    N_BINS = 5
    eval_long["p_bin_q"] = (
        eval_long.groupby("threshold")["p_over"]
        .transform(lambda s: pd.qcut(s, q=N_BINS, duplicates="drop"))
    )

    calib_q = (
        eval_long.groupby(["threshold", "p_bin_q"], as_index=False)
        .agg(
            n=("y", "size"),
            avg_p=("p_over", "mean"),
            hit_rate=("y", "mean"),
        )
    )
    calib_q.insert(0, "section", "calibration_quantile_bins")
    calib_q.insert(0, "run_ts", run_ts)
    calib_q.insert(0, "run_date", run_date)
    calib_q["p_bin"] = calib_q["p_bin_q"].astype(str)
    calib_q = calib_q.drop(columns=["p_bin_q"])

    # -----------------------------
    # 5) Combine into ONE CSV (overwrite each run)
    # -----------------------------
    results = pd.concat(
        [overall, by_thr, calib_wide, calib_q],
        ignore_index=True,
        sort=False
    )

    OUT_PATH = OUT / "prediction_eval_summary.csv"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUT_PATH, index=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Prediction results tracking complete.")

if __name__ == "__main__":
    main()
