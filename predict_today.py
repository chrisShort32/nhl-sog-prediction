import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    ART_DIR = Path(ROOT / "model_artifacts_v2") 
    SLATE_CSV = Path(ROOT / "data_collection/todays_games.csv")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Starting today's prediction process...")
    # --- Load artifacts ---
    with open(ART_DIR / "feature_cols.json", "r") as f:
        FEATURE_COLS = json.load(f)

    models = {
        2: joblib.load(ART_DIR / "cal_lgbm_p_ge_2.joblib"),
        3: joblib.load(ART_DIR / "cal_lgbm_p_ge_3.joblib"),
        4: joblib.load(ART_DIR / "cal_lgbm_p_ge_4.joblib"),
        5: joblib.load(ART_DIR / "cal_lgbm_p_ge_5.joblib"),
    }

    player_latest = pd.read_parquet(ART_DIR / "player_latest_v2.parquet")

    # --- Load slate ---
    games_raw = pd.read_csv(SLATE_CSV)
    games_raw["game_date"] = pd.to_datetime(games_raw["game_date"], errors="coerce")
    games_raw["start_time_UTC"] = pd.to_datetime(games_raw["start_time_UTC"], utc=True, errors="coerce")

    for c in ["away_wins","away_losses","away_otl","home_wins","home_losses","home_otl"]:
        games_raw[c] = pd.to_numeric(games_raw[c], errors="coerce").fillna(0).astype(int)

    away_side = games_raw.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_wins": "team_wins_pre",
        "away_losses": "team_losses_pre",
        "away_otl": "team_otl_pre",
        "home_wins": "opponent_wins_pre",
        "home_losses": "opponent_losses_pre",
        "home_otl": "opponent_otl_pre",
    }).assign(is_home=0)

    home_side = games_raw.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_wins": "team_wins_pre",
        "home_losses": "team_losses_pre",
        "home_otl": "team_otl_pre",
        "away_wins": "opponent_wins_pre",
        "away_losses": "opponent_losses_pre",
        "away_otl": "opponent_otl_pre",
    }).assign(is_home=1)

    slate = pd.concat([away_side, home_side], ignore_index=True)

    slate["team_games_pre"] = slate["team_wins_pre"] + slate["team_losses_pre"] + slate["team_otl_pre"]
    slate["opponent_games_pre"] = slate["opponent_wins_pre"] + slate["opponent_losses_pre"] + slate["opponent_otl_pre"]

    slate["team_points_pct_pre"] = np.where(
        slate["team_games_pre"] > 0,
        (slate["team_wins_pre"] + 0.5 * slate["team_otl_pre"]) / slate["team_games_pre"],
        0.0
    )

    slate["opponent_points_pct_pre"] = np.where(
        slate["opponent_games_pre"] > 0,
        (slate["opponent_wins_pre"] + 0.5 * slate["opponent_otl_pre"]) / slate["opponent_games_pre"],
        0.0
    )

    slate["season"] = slate["season"].astype(str).str.slice(0, 4).astype(int)

    # --- build tonight rows ---
    teams_playing = set(slate["team"])
    player_latest = player_latest[player_latest["team"].isin(teams_playing)]

    tonight = player_latest.merge(slate, on=["season","team"], how="inner", suffixes=("", "_slate"))

    # overwrite game identity
    tonight["game_id"] = tonight["game_id_slate"]
    tonight["opponent"] = tonight["opponent_slate"]
    tonight["is_home"] = tonight["is_home_slate"]
    tonight["game_date"] = tonight["game_date_slate"]
    tonight["start_time_UTC"] = tonight["start_time_UTC_slate"]

    overwrite_cols = [
        "team_wins_pre","team_losses_pre","team_otl_pre",
        "opponent_wins_pre","opponent_losses_pre","opponent_otl_pre",
        "team_games_pre","opponent_games_pre",
        "team_points_pct_pre","opponent_points_pct_pre",
    ]
    for c in overwrite_cols:
        tonight[c] = tonight[f"{c}_slate"]

    # cleanup helper cols
    drop_cols = [c for c in tonight.columns if c.endswith("_slate")] + ["game_id_slate"]
    tonight = tonight.drop(columns=[c for c in drop_cols if c in tonight.columns])


    # ----------------------------
    # predict with v2 cal models
    # ----------------------------
    X_tonight = tonight[FEATURE_COLS].copy()

    for k, m in models.items():
        tonight[f"p_ge{k}"] = m.predict_proba(X_tonight)[:, 1]

    # --- write output ---
    today_str = datetime.now().strftime("%Y%m%d")
    out_path = Path(ROOT / f"predictions/preds_{today_str}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = ["game_id","player_id","player_name","team","opponent","is_home",
                "p_ge2","p_ge3","p_ge4","p_ge5"]
    out = tonight[out_cols].copy()
    out.to_csv(out_path, index=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Today's prediction process complete. Predictions saved to {out_path}")
    
if __name__ == "__main__":
    main()