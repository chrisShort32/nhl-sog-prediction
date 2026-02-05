# This Script Contains Feature Engineering Functions for Player Data
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    OUT = ROOT / "parquets"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting feature engineering process at {ts}...")

    # Get the data
    df = pd.read_parquet(OUT / "df_encoded_base.parquet")

    # Feature Engineering: SHOTS_ON_GOAL
    
    # Ensure sorted order
    df = df.sort_values(["player_id", "season", "game_date"]).copy()
    g = df.groupby(["player_id", "season"], sort=False)["shots_on_goal"]

    # Feature: Rolling average - Average number of shots on goal over previous 3, 5, 7, and 10 games
    # Feature: Rolling 'overs' - How many times a player hit a threshold (2, 3, 4) of SOG over previous 3, 5, 7, and 10 games
    windows = [3, 5, 7, 10]
    thresholds = [2, 3, 4]
    new_cols = {}
    
    for w in windows:
        # Rolling average
        new_cols[f"plr_roll{w}_shots"] = g.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )

    # Rolling overs (thresholds 2,3,4)
    for thr in thresholds:
        for w in windows:
            col_name = f"plr_roll{w}_over{thr}_shots_mean"
            new_cols[col_name] = g.transform(
                lambda s, thr=thr, w=w: (s >= thr).astype(int).shift(1).rolling(w, min_periods=1).mean()
            )
            col_sum = f"plr_roll{w}_over{thr}_shots"
            new_cols[col_sum] = g.transform(
                lambda s, thr=thr, w=w: (s >= thr).astype(int).shift(1).rolling(w, min_periods=1).sum()
            )
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    # Feature: Season to date average and overs
    # Players average SOG per game at this point in the season
    # Number of times a player hit a threshold (2, 3, 4) of SOG
    
    df = df.sort_values(["player_id", "season", "game_date"])
    g = df.groupby(["player_id", "season"], sort=False)["shots_on_goal"]

    # Season to date average - excluding current game
    new_cols = {}
    new_cols["plr_pre_avg_shots"] = g.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    
    # Season to date over - excluding current game
    for thr in [2,3,4]:
        col_name = f"plr_pre_over{thr}_shots"
        new_cols[col_name] = (
            g.transform(lambda s, thr=thr: s.ge(thr).astype(int).shift(1).expanding(min_periods=1).sum())
        )
        
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    # Feature: Home/Away splits for the previous features
    # Rolling average and over, Season to date average and over

    df = df.sort_values(["player_id", "season", "game_date"]).copy()

    windows = [3, 5, 7, 10]
    thresholds = [2, 3, 4]

    new_cols = {}

    # Helper: compute split series on subset, return aligned full-length series
    def split_roll_mean(df_sub, w):
        g = df_sub.groupby(["player_id", "season"], sort=False)["shots_on_goal"]
        return g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    def split_roll_over_sum(df_sub, w, thr):
        g = df_sub.groupby(["player_id", "season"], sort=False)["shots_on_goal"]
        return g.transform(lambda s, w=w, thr=thr: s.ge(thr).astype("int8").shift(1).rolling(w, min_periods=1).sum())

    def split_pre_avg(df_sub):
        g = df_sub.groupby(["player_id", "season"], sort=False)["shots_on_goal"]
        return g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    def split_pre_over_sum(df_sub, thr):
        g = df_sub.groupby(["player_id", "season"], sort=False)["shots_on_goal"]
        return g.transform(lambda s, thr=thr: s.ge(thr).astype("int8").shift(1).expanding(min_periods=1).sum())

    # --- HOME/AWAY rolling features ---
    for loc_flag, loc_name in [(1, "home"), (0, "away")]:
        idx = df.index[df["is_home"] == loc_flag]
        sub = df.loc[idx, ["player_id", "season", "shots_on_goal"]].copy()

        # Rolling average
        for w in windows:
            s = split_roll_mean(sub, w)
            new_cols[f"plr_roll{w}_shots_{loc_name}"] = s.reindex(df.index)

        # Rolling overs sum
        for thr in thresholds:
            for w in windows:
                s = split_roll_over_sum(sub, w, thr)
                new_cols[f"plr_roll{w}_over{thr}_shots_{loc_name}"] = s.reindex(df.index)

        # Season-to-date average
        s = split_pre_avg(sub)
        new_cols[f"plr_pre_avg_shots_{loc_name}"] = s.reindex(df.index)
        # Season-to-date overs sum
        for thr in thresholds:
            s = split_pre_over_sum(sub, thr)
            new_cols[f"plr_pre_over{thr}_shots_{loc_name}"] = s.reindex(df.index)

    # Attach all new columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Ffill within player-season, then fill remaining NaNs with 0
    home_away_cols = [c for c in df.columns if c.endswith("_home") or c.endswith("_away")]
    df[home_away_cols] = (
        df.groupby(["player_id", "season"], sort=False)[home_away_cols]
        .ffill()
        .fillna(0)
    )

    
    # Shot attempts features

    ATT_COL = "shot_attempts_total"

    df = df.sort_values(["player_id", "season", "game_date"]).copy()

    windows = [3, 5, 7, 10]
    new_cols = {}

    # overall group
    g_all = df.groupby(["player_id", "season"], sort=False)[ATT_COL]

    # Overall rolling averages
    for w in windows:
        col_all = f"plr_roll{w}_att"
        new_cols[col_all] = g_all.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
        )

    # Overall season-to-date average
    new_cols["plr_pre_avg_att"] = g_all.transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    # Helper functions for split computations (operate on subset df)
    def _split_roll_mean(df_sub: pd.DataFrame, w: int) -> pd.Series:
        g = df_sub.groupby(["player_id", "season"], sort=False)[ATT_COL]
        return g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    def _split_pre_avg(df_sub: pd.DataFrame) -> pd.Series:
        g = df_sub.groupby(["player_id", "season"], sort=False)[ATT_COL]
        return g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # Home/away split features
    for loc_flag, loc_name in [(1, "home"), (0, "away")]:
        idx = df.index[df["is_home"] == loc_flag]

        # Keep only what we need
        sub = df.loc[idx, ["player_id", "season", ATT_COL]].copy()

        # Rolling average split
        for w in windows:
            col_split = f"plr_roll{w}_att_{loc_name}"
            s = _split_roll_mean(sub, w)
            # Place on subset indices then expand to full index (NaN elsewhere)
            new_cols[col_split] = s.reindex(df.index)

        # Season-to-date average split
        col = f"plr_pre_avg_att_{loc_name}"
        s = _split_pre_avg(sub)
        new_cols[col] = s.reindex(df.index)

    # Attach once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Fill forward ONLY the attempt split cols we just made
    att_split_cols = [c for c in new_cols.keys() if c.endswith("_home") or c.endswith("_away")]
    df[att_split_cols] = (
        df.groupby(["player_id", "season"], sort=False)[att_split_cols]
        .ffill()
        .fillna(0)
    )

    
    # Special teams shots and attempts
    
    WINDOWS = [3, 5, 7, 10]
    
    df = df.sort_values(["player_id", "season", "game_date"])

    def add_roll_and_pre_avgs(df: pd.DataFrame, stat_col: str, prefix: str, windows=WINDOWS) -> pd.DataFrame:
        df = df.sort_values(["player_id", "season", "game_date"]).copy()
        new_cols = {}

        # Overall group (all games)
        g_all = df.groupby(["player_id", "season"], sort=False)[stat_col]

        # Overall rolling means + pre-average
        for w in windows:
            col_all = f"{prefix}_roll{w}"
            new_cols[col_all] = g_all.transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )

        new_cols[f"{prefix}_pre_avg"] = g_all.transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        # Helpers for split computations on subset
        def _split_roll_mean(df_sub: pd.DataFrame, w: int) -> pd.Series:
            g = df_sub.groupby(["player_id", "season"], sort=False)[stat_col]
            return g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

        def _split_pre_avg(df_sub: pd.DataFrame) -> pd.Series:
            g = df_sub.groupby(["player_id", "season"], sort=False)[stat_col]
            return g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

        # Home/away rolling + pre-average
        for loc_flag, loc_name in [(1, "home"), (0, "away")]:
            idx = df.index[df["is_home"] == loc_flag]
            sub = df.loc[idx, ["player_id", "season", stat_col]].copy()

            for w in windows:
                col_split = f"{prefix}_roll{w}_{loc_name}"
                s = _split_roll_mean(sub, w)
                new_cols[col_split] = s.reindex(df.index)

            col = f"{prefix}_pre_avg_{loc_name}"
            s = _split_pre_avg(sub)
            new_cols[col] = s.reindex(df.index)
        # Attach once
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # Ffill only the new split cols for THIS prefix (keeps behavior sane + faster)
        split_cols = [c for c in new_cols.keys() if c.endswith("_home") or c.endswith("_away")]
        df[split_cols] = (
            df.groupby(["player_id", "season"], sort=False)[split_cols]
            .ffill()
        )

        return df


    df = add_roll_and_pre_avgs(df, "pp_shots", "plr_pp_shots")
    df = add_roll_and_pre_avgs(df, "pp_attempts_total", "plr_pp_att")
    df = add_roll_and_pre_avgs(df, "pk_shots", "plr_pk_shots")
    df = add_roll_and_pre_avgs(df, "pk_attempts_total", "plr_pk_att")


    # Any remaining NaNs (first game / first home or away of season) -> 0
    new_feature_cols = [c for c in df.columns if c.startswith("plr_pp_") or c.startswith("plr_pk_")]
    df[new_feature_cols] = df[new_feature_cols].fillna(0)

    df.to_parquet(OUT / "df_feature_engineering.parquet", index=False)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Feature engineering complete at {ts}. Data saved to {OUT / 'df_feature_engineering.parquet'}")
    
if __name__ == "__main__":
    main()