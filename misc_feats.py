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
    OUT = ROOT / "model_artifacts_v6"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting misc features process at {ts}...")

    # Get the data
    df = pd.read_parquet(DATA / "df_team_strength_goals.parquet")
    
    df = df.sort_values(["player_id", "game_date", "game_id"])
    
    # Rolling TOI + PIM
    g_plr = df.groupby("player_id")

    df["plr_roll5_toi"]  = g_plr["toi_seconds"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["plr_roll10_toi"] = g_plr["toi_seconds"].transform(lambda s: s.shift(1).rolling(10).mean())

    df["plr_roll5_pim"]   = g_plr["pim"].transform(lambda s: s.shift(1).rolling(5).mean())
    df["plr_roll10_pim"]  = g_plr["pim"].transform(lambda s: s.shift(1).rolling(10).mean())
    df["plr_avg_pim_pre"] = g_plr["pim"].transform(lambda s: s.shift(1).expanding().mean())
    
    # Build team-game table
    team_game = (
        df.groupby(["team", "season", "game_id"], as_index=False)
        .agg({"pim": "sum", "game_date": "first"})
        .rename(columns={"pim": "team_pim_game"})
        .sort_values(["team", "game_date", "game_id"])
    )

    g_tpim = team_game.groupby(["team", "season"])["team_pim_game"]
    team_game["team_roll5_pim"]         = g_tpim.transform(lambda s: s.shift(1).rolling(5).mean())
    team_game["team_roll10_pim"]        = g_tpim.transform(lambda s: s.shift(1).rolling(10).mean())
    team_game["team_season_avg_pre_pim"] = g_tpim.transform(lambda s: s.shift(1).expanding().mean())

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
        "team_shots_against",
    ]

    # -----------------------------
    # 1) Build team-game table
    # -----------------------------
    team_games = (
        df.groupby(
            ["season", "team_id", "opponent_id", "game_id", "game_date", "is_home"],
            as_index=False,
        )
        .agg(
            team_shots           =("team_shots",           "first"),
            team_shots_against   =("team_shots_against",   "first"),
            team_attempts        =("shot_attempts_total",   "sum"),
            team_attempts_blocked=("shot_attempts_blocked", "sum"),
            team_attempts_missed =("shot_attempts_missed",  "sum"),
            team_blocks          =("blocked_shots",         "sum"),
        )
    )

    team_games = team_games.sort_values(["season", "team_id", "game_date", "game_id"]).copy()
    

    # -----------------------------
    # 2) Overall rolling + season avg (pre-game)
    # -----------------------------
    g_tg = team_games.groupby(["season", "team_id"])

    for col in team_cols:
        team_games[f"{col}_rolling_5"]  = g_tg[col].transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
        team_games[f"{col}_rolling_10"] = g_tg[col].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        team_games[f"{col}_avg"]        = g_tg[col].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # -----------------------------
    # 3) Home / Away rollings (pre-game)
    # -----------------------------
        # --- Helper to compute rolling + avg (home/away subsets) ---
    def compute_homeaway_rollings(df_sub: pd.DataFrame, cols: list) -> pd.DataFrame:
        df_sub = df_sub.sort_values(["season", "team_id", "game_date", "game_id"]).copy()
        g = df_sub.groupby(["season", "team_id"])

        out = df_sub[["season", "team_id", "game_id"]].copy()
        for col in cols:
            out[f"{col}_rolling_5"] = g[col].transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
            out[f"{col}_rolling_10"] = g[col].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
            out[f"{col}_avg"] = g[col].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        return out
    
    home_sub = team_games[team_games["is_home"] == 1].copy()
    away_sub = team_games[team_games["is_home"] == 0].copy()
    
    home_rolls = compute_homeaway_rollings(home_sub, team_cols)
    away_rolls = compute_homeaway_rollings(away_sub, team_cols)

    home_rolls = home_rolls.rename(columns={
        c: c.replace("_rolling_5",  "_home_rolling_5")
             .replace("_rolling_10", "_home_rolling_10")
             .replace("_avg",        "_home_avg")
        for c in home_rolls.columns if c not in ("season", "team_id", "game_id")
    })
    away_rolls = away_rolls.rename(columns={
        c: c.replace("_rolling_5",  "_away_rolling_5")
             .replace("_rolling_10", "_away_rolling_10")
             .replace("_avg",        "_away_avg")
        for c in away_rolls.columns if c not in ("season", "team_id", "game_id")
    })

    # Single merge per split
    team_games = (
        team_games
        .merge(home_rolls, on=["season", "team_id", "game_id"], how="left")
        .merge(away_rolls, on=["season", "team_id", "game_id"], how="left")
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

    df = df.sort_values(["player_id", "season", "game_date", "game_id"]).copy()

    def add_roll_and_pre_avgs(
        df: pd.DataFrame,
        stat_col: str,
        prefix: str,
        windows: list = WINDOWS,
    ) -> pd.DataFrame:
        """
        Adds rolling means and season-to-date means for stat_col, overall and
        split by home/away.
        """
        g_all = df.groupby(["player_id", "season"], sort=False)[stat_col]

        # Overall rolling + season-to-date
        for w in windows:
            df[f"{prefix}_roll{w}"] = g_all.transform(
                lambda s, w=w: s.shift(1).rolling(window=w, min_periods=1).mean()
            )

        df[f"{prefix}_pre_avg"] = g_all.transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )

        # Home / away splits — operate on subset, reindex to full frame
        for loc_flag, loc_name in [(1, "home"), (0, "away")]:
            mask = df["is_home"] == loc_flag
            sub  = df.loc[mask, ["player_id", "season", stat_col]].copy()
            g_sub = sub.groupby(["player_id", "season"], sort=False)[stat_col]

            for w in windows:
                col_split = f"{prefix}_roll{w}_{loc_name}"
                df[col_split] = (
                    g_sub.transform(lambda s, w=w: s.shift(1).rolling(window=w, min_periods=1).mean())
                    .reindex(df.index)          # NaN for rows that are the other location
                )

            col_pre = f"{prefix}_pre_avg_{loc_name}"
            df[col_pre] = (
                g_sub.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
                .reindex(df.index)
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
        df.groupby(["player_id", "season"], sort=False)[split_cols]
        .ffill()
        .fillna(0)
    )

    player_latest = (
        df.sort_values(["player_id", "game_date", "game_id"])
            .groupby("player_id", as_index=False)
            .tail(1)
            .copy()
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Saving player_latest_v6.parquet at {ts}...")
    player_latest.to_parquet(OUT / "player_latest_v6.parquet", index=False)
    
    print(f"Saving df_model_v6.parquet at {ts}...")
    df.to_parquet(OUT / "df_model_v6.parquet", index=False)
    
if __name__ == "__main__":
    main()