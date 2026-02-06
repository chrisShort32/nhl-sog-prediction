# Some additional features added in v2:
# - Player rolling + season-to-date average PIM
# - Player rolling + season-to-date average TOI
# - Player rolling + season-to-date average for:
#   - Shots per 60 minutes of TOI
#   - Shot attempts per 60 minutes of TOI
#   - Shots per shift
#   - Shot attempts per shift
# - Team rolling + season-to-date average PIM
# - Opponent rolling + season-to-date average PIM
# - Team-level rolling + season-to-date average for shots, attempts, blocked shots (overall and home/away)
# - Opponent-level rolling + season-to-date average for shots, attempts, blocked shots (overall and home/away)


from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "parquets"
    OUT = ROOT / "model_artifacts_v2"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting misc features process at {ts}...")

    # Get the data
    df = pd.read_parquet(DATA / "df_team_strength_goals.parquet")
    
    df = df.sort_values(["player_id", "game_id"])

    df["plr_roll5_toi"] = (
        df.groupby("player_id")["toi_seconds"]
        .transform(lambda s: s.shift(1).rolling(5).mean())
    )

    df["plr_roll10_toi"] = (
        df.groupby("player_id")["toi_seconds"]
        .transform(lambda s: s.shift(1).rolling(10).mean())
    )
    
    df = df.sort_values(["player_id", "game_id"])

    df["plr_roll5_pim"] = (
        df.groupby("player_id")["pim"]
        .transform(lambda s: s.shift(1).rolling(5).mean())
    )

    df["plr_roll10_pim"] = (
        df.groupby("player_id")["pim"]
        .transform(lambda s: s.shift(1).rolling(10).mean())
    )

    df["plr_avg_pim_pre"] = (
        df.groupby("player_id")["pim"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    
    # Build team-game table
    team_game = (
        df.groupby(["team", "season", "game_id"], as_index=False)["pim"]
        .sum()
        .rename(columns={"pim": "team_pim_game"})
        .sort_values(["team", "game_id"])
    )

    # Rolling + season-to-date (pre-game)
    team_game["team_roll5_pim"] = (
        team_game.groupby("team")["team_pim_game"]
                .transform(lambda s: s.shift(1).rolling(5).mean())
    )

    team_game["team_roll10_pim"] = (
        team_game.groupby("team")["team_pim_game"]
                .transform(lambda s: s.shift(1).rolling(10).mean())
    )

    team_game["team_season_avg_pre_pim"] = (
        team_game.groupby(["team", "season"])["team_pim_game"]
                .transform(lambda s: s.shift(1).expanding().mean())
    )

    # Merge back (include team_pim_game too)
    df = df.merge(
        team_game[[
            "team", "season", "game_id",
            "team_pim_game", "team_roll5_pim", "team_roll10_pim", "team_season_avg_pre_pim"
        ]],
        on=["team", "season", "game_id"],
        how="left"
    )

    # Map opponent to opponent's team_pim_game for the same date/season
    opp_game = team_game.rename(columns={
        "team": "opponent",
        "team_pim_game": "opp_pim_game",
        "team_roll5_pim": "opp_roll5_pim",
        "team_roll10_pim": "opp_roll10_pim",
        "team_season_avg_pre_pim": "opp_season_avg_pre_pim",
    })

    df = df.merge(
        opp_game[[
            "opponent", "season", "game_id",
            "opp_pim_game", "opp_roll5_pim", "opp_roll10_pim", "opp_season_avg_pre_pim"
        ]],
        on=["opponent", "season", "game_id"],
        how="left"
    )
    
    ROLL_WINDOWS = (5, 10)

    team_cols = [
        "team_shots",
        "team_attempts",
        "team_attempts_blocked",
        "team_attempts_missed",
        "team_blocks",
    ]

    # --- Helper to compute rolling + avg (home/away subsets) ---
    def compute_homeaway_rollings(df_, col):
        df_ = df_.sort_values(["season", "team_id", "game_date"]).copy()
        g = df_.groupby(["season", "team_id"])[col]

        out = df_[["season", "team_id", "game_id"]].copy()
        out[f"{col}_rolling_5"] = g.transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
        out[f"{col}_rolling_10"] = g.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        out[f"{col}_avg"] = g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        return out


    # -----------------------------
    # 1) Build team-game table
    # -----------------------------
    team_games = (
        df.groupby(
            ["season", "team_id", "opponent_id", "game_id", "game_date", "is_home"],
            as_index=False
        )
        .agg({
            # already team-level -> take first (avoid double counting across players)
            "team_shots": "first",
            "team_shots_against": "first",

            # derived from player sums
            "shot_attempts_total": "sum",
            "shot_attempts_blocked": "sum",
            "shot_attempts_missed": "sum",
            "blocked_shots": "sum",
        })
        .rename(columns={
            "shot_attempts_total": "team_attempts",
            "shot_attempts_blocked": "team_attempts_blocked",
            "shot_attempts_missed": "team_attempts_missed",
            "blocked_shots": "team_blocks",
        })
    )

    # ensure clean sort
    team_games = team_games.sort_values(["season", "team_id", "game_date"]).copy()
    grouped = team_games.groupby(["season", "team_id"])

    # -----------------------------
    # 2) Overall rolling + season avg (pre-game)
    # -----------------------------
    for col in team_cols:
        team_games[f"{col}_rolling_5"] = (
            grouped[col]
            .rolling(5, min_periods=1).mean()
            .reset_index(level=[0,1], drop=True)
            .groupby([team_games["season"], team_games["team_id"]])
            .shift(1)
        )

        team_games[f"{col}_rolling_10"] = (
            grouped[col]
            .rolling(10, min_periods=1).mean()
            .reset_index(level=[0,1], drop=True)
            .groupby([team_games["season"], team_games["team_id"]])
            .shift(1)
        )

        team_games[f"{col}_avg"] = (
            grouped[col]
            .expanding().mean()
            .reset_index(level=[0,1], drop=True)
            .groupby([team_games["season"], team_games["team_id"]])
            .shift(1)
        )

    # -----------------------------
    # 3) Home / Away rollings (pre-game)
    # -----------------------------
    # NOTE: use == 1 / == 0 to avoid dtype surprises
    home = team_games.query("is_home == 1").copy()
    away = team_games.query("is_home == 0").copy()

    for col in team_cols:
        home_rolls = compute_homeaway_rollings(home, col).rename(columns={
            f"{col}_rolling_5":  f"{col}_home_rolling_5",
            f"{col}_rolling_10": f"{col}_home_rolling_10",
            f"{col}_avg":        f"{col}_home_avg",
        })

        away_rolls = compute_homeaway_rollings(away, col).rename(columns={
            f"{col}_rolling_5":  f"{col}_away_rolling_5",
            f"{col}_rolling_10": f"{col}_away_rolling_10",
            f"{col}_avg":        f"{col}_away_avg",
        })

        team_games = (
            team_games
            .merge(
                home_rolls[["season","team_id","game_id",
                            f"{col}_home_rolling_5", f"{col}_home_rolling_10", f"{col}_home_avg"]],
                on=["season","team_id","game_id"],
                how="left"
            )
            .merge(
                away_rolls[["season","team_id","game_id",
                            f"{col}_away_rolling_5", f"{col}_away_rolling_10", f"{col}_away_avg"]],
                on=["season","team_id","game_id"],
                how="left"
            )
        )

    # -----------------------------
    # 4) Opponent features (copy opponent's team-side rollings)
    # -----------------------------
    def generated_cols_for_metric(metric: str, windows=ROLL_WINDOWS):
        cols = [f"{metric}_avg"]
        cols += [f"{metric}_rolling_{w}" for w in windows]
        cols += [f"{metric}_home_avg", f"{metric}_away_avg"]
        cols += [f"{metric}_home_rolling_{w}" for w in windows]
        cols += [f"{metric}_away_rolling_{w}" for w in windows]
        return cols

    team_feature_cols = []
    for m in team_cols:
        team_feature_cols += generated_cols_for_metric(m)

    opp_merge = (
        team_games[["season", "game_id", "team_id"] + team_feature_cols]
        .rename(columns={"team_id": "opponent_id"})
        .rename(columns={c: c.replace("team_", "opp_") for c in team_feature_cols})
    )

    team_games = team_games.merge(
        opp_merge,
        on=["season", "game_id", "opponent_id"],
        how="left"
    )

    opp_feature_cols = [c.replace("team_", "opp_") for c in team_feature_cols]

    # -----------------------------
    # 5) Merge back into player-level df
    # -----------------------------
    df = df.merge(
        team_games[["season", "team_id", "game_id"] + team_feature_cols + opp_feature_cols],
        on=["season", "team_id", "game_id"],
        how="left"
    )

    WINDOWS = [3, 5, 7, 10] 

    df = df.sort_values(["player_id", "season", "game_date"]).copy()

    def add_roll_and_pre_avgs(df: pd.DataFrame, stat_col: str, prefix: str, windows=WINDOWS) -> pd.DataFrame:
        """
        Adds:
        - {prefix}_roll{w}: overall rolling mean (shifted by 1)
        - {prefix}_roll{w}_home / _away: home/away rolling mean (shifted by 1), then ffill within player-season
        - {prefix}_pre_avg: overall season-to-date mean (shifted by 1)
        - {prefix}_pre_avg_home / _away: home/away season-to-date mean (shifted by 1), then ffill within player-season
        """
        # overall rolling means
        for w in windows:
            df[f"{prefix}_roll{w}"] = (
                df.groupby(["player_id", "season"], group_keys=False)[stat_col]
                .apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            )

            # home/away rolling means (compute only on subset)
            for loc_flag, loc_name in [(1, "home"), (0, "away")]:
                col_split = f"{prefix}_roll{w}_{loc_name}"
                df[col_split] = np.nan
                mask = df["is_home"] == loc_flag
                df.loc[mask, col_split] = (
                    df.loc[mask]
                    .groupby(["player_id", "season"], group_keys=False)[stat_col]
                    .apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                )

        # overall season-to-date mean
        df[f"{prefix}_pre_avg"] = (
            df.groupby(["player_id", "season"], group_keys=False)[stat_col]
            .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

        # home/away season-to-date mean
        for loc_flag, loc_name in [(1, "home"), (0, "away")]:
            col = f"{prefix}_pre_avg_{loc_name}"
            df[col] = np.nan
            mask = df["is_home"] == loc_flag
            df.loc[mask, col] = (
                df.loc[mask]
                .groupby(["player_id", "season"], group_keys=False)[stat_col]
                .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
            )

        return df


    # ------------------------------------------------------------
    # 1) Raw-count features
    # ------------------------------------------------------------
    # rolling/season avg: attempts blocked, attempts missed
    df = add_roll_and_pre_avgs(df, "shot_attempts_blocked", "plr_blk_att")
    df = add_roll_and_pre_avgs(df, "shot_attempts_missed",  "plr_miss_att")

    # ------------------------------------------------------------
    # 2) Rate features (per TOI, per shift)
    # ------------------------------------------------------------
    # Guard against divide-by-zero
    toi = df["toi_seconds"].replace(0, np.nan)
    shf = df["shifts"].replace(0, np.nan)

    # per 60 minutes (shots/attempts per TOI)
    df["shots_per_toi60"]    = df["shots_on_goal"]      / toi * 3600.0
    df["att_per_toi60"]      = df["shot_attempts_total"]/ toi * 3600.0

    # per shift
    df["shots_per_shift"]    = df["shots_on_goal"]       / shf
    df["att_per_shift"]      = df["shot_attempts_total"] / shf

    # Replace any NaNs created by 0 TOI / 0 shifts with 0 for the *raw rate columns* themselves
    rate_cols = ["shots_per_toi60", "att_per_toi60", "shots_per_shift", "att_per_shift"]
    df[rate_cols] = df[rate_cols].fillna(0)

    # now add rolling + season-to-date (with home/away splits) for these rates
    df = add_roll_and_pre_avgs(df, "shots_per_toi60", "plr_shots_per_toi60")
    df = add_roll_and_pre_avgs(df, "att_per_toi60",   "plr_att_per_toi60")
    df = add_roll_and_pre_avgs(df, "shots_per_shift", "plr_shots_per_shift")
    df = add_roll_and_pre_avgs(df, "att_per_shift",   "plr_att_per_shift")

    # ------------------------------------------------------------
    # 3) Home/away ffill and fill remaining NaNs with 0
    # ------------------------------------------------------------
    split_cols = [c for c in df.columns if c.endswith("_home") or c.endswith("_away")]
    df[split_cols] = (
        df.groupby(["player_id", "season"], group_keys=False)[split_cols]
        .ffill()
        .fillna(0)
    )

    player_latest = (
        df.sort_values(["player_id", "game_id"])
            .groupby("player_id", as_index=False)
            .tail(1)
            .copy()
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Saving player_latest_v2.parquet at {ts}...")
    player_latest.to_parquet(OUT / "player_latest_v2.parquet", index=False)
    
    print(f"Saving df_model_v2.parquet at {ts}...")
    df.to_parquet(OUT / "df_model_v2.parquet", index=False)
    
if __name__ == "__main__":
    main()