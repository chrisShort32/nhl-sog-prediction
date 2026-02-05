# This script is part one of team strength metrics
# It handles team level win/loss metrics for team and opponent
# Derives games played at the time of the current game
# Points percentage at the time of the current game
# Win and point percentage differentials
# Win ratios
# Rolling win and win percentage for 5 and 10 games
# Differentials rolling wins and win percentage
# Home and away splits for each of these categories
# Derived rest days
# Saved as: df_team_strength_wins_rest.parquet

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    OUT = ROOT / "parquets"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting team strength wins process at {ts}...")

    # Get the data
    df = pd.read_parquet(OUT / "df_feature_engineering.parquet")
    
    # Normalize team column names
    df = df.rename(columns={
        "opp_wins_pre": "opponent_wins_pre",
        "opp_losses_pre": "opponent_losses_pre",
        "opp_otl_pre": "opponent_otl_pre",
    })
    
    # Team games played pre-game
    df["team_games_pre"] = (
        df["team_wins_pre"] + df["team_losses_pre"] + df["team_otl_pre"]
    )

    # Opponent games played pre-game
    df["opponent_games_pre"] = (
        df["opponent_wins_pre"] + df["opponent_losses_pre"] + df["opponent_otl_pre"]
    )

    # Points percentage: (W + 0.5*OTL) / (W + L + OTL)
    df["team_points_pct_pre"] = np.where(
        df["team_games_pre"] > 0,
        (df["team_wins_pre"] + 0.5*df["team_otl_pre"]) / df["team_games_pre"],
        0.0
    ).round(3)

    df["opponent_points_pct_pre"] = np.where(
        df["opponent_games_pre"] > 0,
        (df["opponent_wins_pre"] + 0.5*df["opponent_otl_pre"]) / df["opponent_games_pre"],
        0.0
    ).round(3)
        
    # Win, points, and points% differentials
    df["win_diff_pre"] = (
        df["team_wins_pre"] - df["opponent_wins_pre"]
    )

    df["points_diff_pre"] = (
        (df["team_wins_pre"] + 0.5 * df["team_otl_pre"])
        - (df["opponent_wins_pre"] + 0.5 * df["opponent_otl_pre"])
    )

    df["points_pct_diff_pre"] = (
        df["team_points_pct_pre"] - df["opponent_points_pct_pre"]
    )
    
    # Win/loss ratios
    df["team_win_loss_ratio_pre"] = np.where(
        df["team_losses_pre"] > 0,
        df["team_wins_pre"] / df["team_losses_pre"],
        np.nan
    )
    df["opponent_win_loss_ratio_pre"] = np.where(
        df["opponent_losses_pre"] > 0,
        df["opponent_wins_pre"] / df["opponent_losses_pre"],
        np.nan
    )
    
    
    df["game_outcome"] = np.select(
    [
        df["team_win"] == 1,
        df["team_otl"] == 1,
        df["team_loss"] == 1,
    ],
    [
        "W",
        "OTL",
        "L",
    ],
    default="UNK"
)
    
    # collapse to one record per team per game
    team_games = (
        df.groupby(["season", "team_id", "game_id"], as_index=False)
        .agg({"game_outcome": "first", "team_games_pre": "first"})
    ).sort_values(["season", "team_id", "game_id"])
    
    
    # Mark wins
    team_games["team_win_game"] = (team_games["game_outcome"] == "W").astype(int)

    # Rolling window of previous 5 (exclude current)
    team_games["team_wins_last_5"] = (
        team_games.groupby(["season", "team_id"])["team_win_game"]
        .transform(lambda s: s.rolling(5, min_periods=1).sum().shift(1))
        .fillna(0)
    )

    # Denominator = min(prior games, 5)
    denom = team_games["team_games_pre"].clip(upper=5)

    team_games["team_win_pct_last_5"] = np.where(
        denom.eq(0), 0, team_games["team_wins_last_5"] / denom
    ).round(3)

    # Merge back to original df
    df = df.merge(
        team_games[["season","team_id","game_id","team_win_game","team_wins_last_5","team_win_pct_last_5"]],
        on=["season","team_id","game_id"],
        how="left"
    )
    
    # Rolling window of previous 10 (exclude current)
    team_games["team_wins_last_10"] = (
        team_games.groupby(["season", "team_id"])["team_win_game"]
        .transform(lambda s: s.rolling(10, min_periods=1).sum().shift(1))
        .fillna(0)
    )

    # Denominator = min(prior games, 10)
    denom10 = team_games["team_games_pre"].clip(upper=10)

    team_games["team_win_pct_last_10"] = np.where(
        denom10.eq(0),
        0,
        team_games["team_wins_last_10"] / denom10
    ).round(3)
    
    df = df.merge(
        team_games[[
            "season","team_id","game_id",
            "team_wins_last_10","team_win_pct_last_10"
        ]],
        on=["season","team_id","game_id"],
        how="left"
    )
    
    # Rename team_games columns to opponent versions for merge
    opp_stats = team_games[[
        "season", "team_id", "game_id",
        "team_wins_last_5", "team_win_pct_last_5",
        "team_wins_last_10", "team_win_pct_last_10"
    ]].rename(columns={
        "team_id": "opponent_id",
        "team_wins_last_5": "opp_wins_last_5",
        "team_win_pct_last_5": "opp_win_pct_last_5",
        "team_wins_last_10": "opp_wins_last_10",
        "team_win_pct_last_10": "opp_win_pct_last_10"
    })

    # Merge opponent stats back into df
    df = df.merge(
        opp_stats,
        on=["season", "opponent_id", "game_id"],
        how="left"
    )
    
    # Momentum differentials - positive = heating up, negative = slumping
    df["team_wins_diff_5v10"] = (df["team_wins_last_5"] - df["team_wins_last_10"])
    df["team_win_pct_diff_5v10"] = (df["team_win_pct_last_5"] - df["team_win_pct_last_10"]).round(3)

    df["opp_wins_diff_5v10"] = (df["opp_wins_last_5"] - df["opp_wins_last_10"])
    df["opp_win_pct_diff_5v10"] = (df["opp_win_pct_last_5"] - df["opp_win_pct_last_10"]).round(3)
    
    # Momentum differentials - between team and opponent - normalized (-1, 1)
    df["momentum_diff_5v10"] = (
        (df["team_wins_diff_5v10"] + df["team_win_pct_diff_5v10"]) - (df["opp_wins_diff_5v10"] + df["opp_win_pct_diff_5v10"])
    ).round(3) / 2
    
    # Wins differential between team and opponent
    df["team_vs_opp_wins_diff_5"] = (
        df["team_wins_last_5"] - df["opp_wins_last_5"]
    )
    df["team_vs_opp_wins_diff_10"] = (
        df["team_wins_last_10"] - df["opp_wins_last_10"]
    )

    # Win percentage differential between team and opponent
    df["team_vs_opp_win_pct_diff_5"] = (
        df["team_win_pct_last_5"] - df["opp_win_pct_last_5"]
    ).round(3)
    df["team_vs_opp_win_pct_diff_10"] = (
        df["team_win_pct_last_10"] - df["opp_win_pct_last_10"]
    ).round(3)
    
    
    # Compute rolling home/away form features (wins + win%) for each team.
        # Assumes df has columns: ['season', 'team_id', 'is_home', 'game_date', 'game_outcome'].
        # Returns a DataFrame with new columns merged in:
        # team_home_wins_last_5, team_home_win_pct_last_5, ...
        # team_away_wins_last_5, team_away_win_pct_last_5, etc.
    def add_home_away_form_features(df):

        # Prepare base (one row per team per home/away game)
        team_games_homeaway = (
            df.groupby(["season", "team_id", "is_home", "game_date"], as_index=False)
            .agg({"game_outcome": "first"})
            .sort_values(["season", "team_id", "is_home", "game_date"])
        )

        # mark wins
        team_games_homeaway["team_win_game"] = (team_games_homeaway["game_outcome"] == "W").astype(int)

        # Rolling windows (5 and 10)
        for window in [5, 10]:
            col_wins = f"wins_last_{window}"
            col_win_pct = f"win_pct_last_{window}"

            team_games_homeaway[col_wins] = (
                team_games_homeaway.groupby(["season","team_id","is_home"])["team_win_game"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum().shift(1))
                .fillna(0)
            )

            denom = (
                team_games_homeaway.groupby(["season","team_id","is_home"]).cumcount().clip(upper=window)
            )
            team_games_homeaway[col_win_pct] = np.where(
                denom.eq(0),
                0,
                team_games_homeaway[col_wins] / denom
            ).round(3)

        # Pivot out home vs away into separate columns
        home = (
            team_games_homeaway.query("is_home == 1")[[
                "season","team_id","game_date","wins_last_5","win_pct_last_5","wins_last_10","win_pct_last_10"
            ]]
            .rename(columns={
                "wins_last_5":"team_home_wins_last_5",
                "win_pct_last_5":"team_home_win_pct_last_5",
                "wins_last_10":"team_home_wins_last_10",
                "win_pct_last_10":"team_home_win_pct_last_10"
            })
        )

        away = (
            team_games_homeaway.query("is_home == 0")[[
                "season","team_id","game_date","wins_last_5","win_pct_last_5","wins_last_10","win_pct_last_10"
            ]]
            .rename(columns={
                "wins_last_5":"team_away_wins_last_5",
                "win_pct_last_5":"team_away_win_pct_last_5",
                "wins_last_10":"team_away_wins_last_10",
                "win_pct_last_10":"team_away_win_pct_last_10"
            })
        )

        # Merge both home/away splits back to main df on team/date
        df = df.merge(home, on=["season","team_id","game_date"], how="left")
        df = df.merge(away, on=["season","team_id","game_date"], how="left")



        # opponent context
        # Flip is_home (if team is home=1, opponent is away=0)
        df["opp_is_home"] = 1 - df["is_home"]

        # Merge opponent home/away form using the same team_games_homeaway table
        opp_merge = (
            team_games_homeaway[[
                "season","team_id","is_home","game_date",
                "wins_last_5","win_pct_last_5","wins_last_10","win_pct_last_10"
            ]]
            .rename(columns={
                "team_id":"opponent_id",
                "is_home":"opp_is_home",
                "wins_last_5":"opp_wins_last_5_homeaway",
                "win_pct_last_5":"opp_win_pct_last_5_homeaway",
                "wins_last_10":"opp_wins_last_10_homeaway",
                "win_pct_last_10":"opp_win_pct_last_10_homeaway"
            })
        )

        df = df.merge(
            opp_merge,
            on=["season","opponent_id","opp_is_home","game_date"],
            how="left"
        )

        return df

    df = add_home_away_form_features(df)
    
    
    # rest days
    # collapse to one row per team per game
    team_games = (
        df.groupby(["season","team_id","game_date"], as_index=False)
        .agg({"game_outcome":"first"})
        .sort_values(["season","team_id","game_date"])
    )

    # compute rest safely
    team_games["team_days_rest"] = (
        team_games.groupby(["season","team_id"])["game_date"]
            .diff()
            .dt.days
            .fillna(0)
            .astype(int)
    )

    # same for opponents
    opp_games = (
        df.groupby(["season","opponent_id","game_date"], as_index=False)
        .agg({"game_outcome":"first"})
        .rename(columns={"opponent_id":"team_id"})  # reuse same logic
        .sort_values(["season","team_id","game_date"])
    )
    opp_games["opp_days_rest"] = (
        opp_games.groupby(["season","team_id"])["game_date"]
            .diff()
            .dt.days
            .fillna(0)
            .astype(int)
    )

    # merge both back to df
    df = df.merge(
        team_games[["season","team_id","game_date","team_days_rest"]],
        on=["season","team_id","game_date"],
        how="left"
    )
    df = df.merge(
        opp_games[["season","team_id","game_date","opp_days_rest"]],
        left_on=["season","opponent_id","game_date"],
        right_on=["season","team_id","game_date"],
        suffixes=("","_drop"),
        how="left"
    ).drop(columns="team_id_drop")

    # final differential
    df["rest_diff"] = df["team_days_rest"] - df["opp_days_rest"]

    df.to_parquet(OUT / "df_team_strength_wins_rest.parquet", index=False)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Team strength wins process complete at {ts}. Data saved to {OUT / 'df_team_strength_wins_rest.parquet'}")
    
    
if __name__ == "__main__":
    main()