# Note: In the data collection, goals counted towards teams that won or lost in a shootout are excluded
# This means teams will have one less goal for each shootout win they posted
# Teams will have one less goal against for every shootout loss they posted

# This notebook is part 2 of team strength
# It handles team strength in the context of goals for and against both the team and its opponent
# Rolling and season average goals for and against
# Rolling and season average goal differential
# Home and away splits for each


import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    OUT = ROOT / "parquets"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting team strength goals process at {ts}...")

    # Get the data
    df = pd.read_parquet(OUT / "df_team_strength_wins_rest.parquet")
    
    # Rolling and season average goals for and against

    # Collapse to one row per team per game
    team_games = (
        df.groupby(["season", "team_id", "opponent_id", "game_id", "game_date", "is_home"], as_index=False)
        .agg({
            "team_goals": "first",
            "team_goals_against": "first",
        })
    )

    # Sort and group for rolling stats
    team_games = team_games.sort_values(["season", "team_id", "game_date"])
    grouped = team_games.groupby(["season", "team_id"])

    # Rolling averages and season averages
    team_games["team_goals_rolling_5"] = (
        grouped["team_goals"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )
    team_games["team_goals_rolling_10"] = (
        grouped["team_goals"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )
    team_games["team_goals_avg"] = (
        grouped["team_goals"]
        .expanding()
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )

    team_games["team_goals_against_rolling_5"] = (
        grouped["team_goals_against"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )
    team_games["team_goals_against_rolling_10"] = (
        grouped["team_goals_against"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )
    team_games["team_goals_against_avg"] = (
        grouped["team_goals_against"]
        .expanding()
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )

    # Merge back into player-level df
    df = df.merge(
        team_games[
            [
                "season", "team_id", "game_id",
                "team_goals_rolling_5", "team_goals_rolling_10", "team_goals_avg",
                "team_goals_against_rolling_5", "team_goals_against_rolling_10", "team_goals_against_avg"
            ]
        ],
        on=["season", "team_id", "game_id"],
        how="left"
    )


    # Rolling and season average team goal differential
    # Compute per-game differential
    team_games["team_goal_diff"] = (
        team_games["team_goals"] - team_games["team_goals_against"]
    )

    # Rolling and season averages (shifted by 1)
    grouped = team_games.groupby(["season", "team_id"])

    team_games["team_goal_diff_rolling_5"] = (
        grouped["team_goal_diff"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )

    team_games["team_goal_diff_rolling_10"] = (
        grouped["team_goal_diff"]
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )

    team_games["team_goal_diff_avg"] = (
        grouped["team_goal_diff"]
        .expanding()
        .mean()
        .reset_index(level=[0,1], drop=True)
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )

    team_games["team_goals_cumulative_pre"] = (
        grouped["team_goals"].cumsum()
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )
    team_games["team_goals_against_cumulative_pre"] = (
        grouped["team_goals_against"].cumsum()
        .groupby([team_games["season"], team_games["team_id"]])
        .shift(1)
    )

    # Opponent rolling goals and season average
    # Create opponent dataframe
    opp_merge = (
        team_games[[
            "season", "team_id", "game_id", "game_date",
            "team_goals", "team_goals_against", "team_goal_diff",
            "team_goals_rolling_5", "team_goals_rolling_10", "team_goals_avg",
            "team_goals_against_rolling_5", "team_goals_against_rolling_10", "team_goals_against_avg",
            "team_goal_diff_rolling_5", "team_goal_diff_rolling_10", "team_goal_diff_avg",
            "team_goals_cumulative_pre", "team_goals_against_cumulative_pre",
        ]]
        .rename(columns={
            "team_id": "opponent_id",
            "team_goals": "opp_goals_for",
            "team_goals_against": "opp_goals_against",
            "team_goal_diff": "opp_goal_diff",
            "team_goals_rolling_5": "opp_goals_for_rolling_5",
            "team_goals_rolling_10": "opp_goals_for_rolling_10",
            "team_goals_avg": "opp_goals_for_avg",
            "team_goals_against_rolling_5": "opp_goals_against_rolling_5",
            "team_goals_against_rolling_10": "opp_goals_against_rolling_10",
            "team_goals_against_avg": "opp_goals_against_avg",
            "team_goal_diff_rolling_5": "opp_goal_diff_rolling_5",
            "team_goal_diff_rolling_10": "opp_goal_diff_rolling_10",
            "team_goal_diff_avg": "opp_goal_diff_avg",
            "team_goals_cumulative_pre": "opp_goals_cumulative",
            "team_goals_against_cumulative_pre": "opp_goals_against_cumulative",
        })
    )

    # Merge back to main team_games on opponent_id
    team_games = team_games.merge(
        opp_merge,
        on=["season", "game_id", "opponent_id"],
        how="left",
        suffixes=("", "_opp")
    )

    cols_to_merge = [
        "season", "team_id", "game_id",
        "team_goal_diff", "team_goal_diff_rolling_5", "team_goal_diff_rolling_10", "team_goal_diff_avg",
        "team_goals_cumulative_pre", "team_goals_against_cumulative_pre",
        "opp_goals_for_rolling_5", "opp_goals_against_rolling_5", "opp_goal_diff_rolling_5",
        "opp_goals_for_rolling_10", "opp_goals_against_rolling_10", "opp_goal_diff_rolling_10",
        "opp_goals_for_avg", "opp_goals_against_avg", "opp_goal_diff_avg",
        "opp_goals_cumulative", "opp_goals_against_cumulative"
    ]

    df = df.merge(
        team_games[cols_to_merge],
        on=["season", "team_id", "game_id"],
        how="left"
    )
    
    
    # Ensure clean sorting
    team_games = team_games.sort_values(["season", "team_id", "game_date"]).copy()

    # Helper for rolling features
    def compute_homeaway_rollings(df_, col):
        df_ = df_.sort_values(["season", "team_id", "game_date"]).copy()
        g = df_.groupby(["season", "team_id"])[col]

        out = df_[["season", "team_id", "game_id"]].copy()
        out[f"{col}_rolling_5"]  = g.transform(lambda s: s.shift(1).rolling(5,  min_periods=1).mean())
        out[f"{col}_rolling_10"] = g.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        out[f"{col}_avg"]        = g.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        return out


    # Compute for home and away separately
    home = team_games.query("is_home == 1").copy()
    away = team_games.query("is_home == 0").copy()

    home_rolls = compute_homeaway_rollings(home, "team_goals")
    away_rolls = compute_homeaway_rollings(away, "team_goals")

    # Rename columns to indicate context
    home_rolls = home_rolls.rename(columns={
        "team_goals_rolling_5": "team_goals_home_rolling_5",
        "team_goals_rolling_10": "team_goals_home_rolling_10",
        "team_goals_avg": "team_goals_home_avg",
    })
    away_rolls = away_rolls.rename(columns={
        "team_goals_rolling_5": "team_goals_away_rolling_5",
        "team_goals_rolling_10": "team_goals_away_rolling_10",
        "team_goals_avg": "team_goals_away_avg",
    })

    # Merge back
    team_games = (
        team_games
        .merge(home_rolls[["season", "team_id", "game_id",
                        "team_goals_home_rolling_5", "team_goals_home_rolling_10", "team_goals_home_avg"]],
            on=["season", "team_id", "game_id"], how="left")
        .merge(away_rolls[["season", "team_id", "game_id",
                        "team_goals_away_rolling_5", "team_goals_away_rolling_10", "team_goals_away_avg"]],
            on=["season", "team_id", "game_id"], how="left")
    )

    # --- Ensure correct ordering ---
    team_games = team_games.sort_values(["season", "team_id", "game_date"]).copy()
    
    # --- Master loop over each metric ---
    for col in ["team_goals_against", "team_goal_diff"]:

        # HOME subset
        home = team_games.query("is_home == True").copy()
        home_rolls = compute_homeaway_rollings(home, col)
        home_rolls = home_rolls.rename(columns={
            f"{col}_rolling_5":  f"{col}_home_rolling_5",
            f"{col}_rolling_10": f"{col}_home_rolling_10",
            f"{col}_avg":        f"{col}_home_avg",
        })

        # AWAY subset
        away = team_games.query("is_home == False").copy()
        away_rolls = compute_homeaway_rollings(away, col)
        away_rolls = away_rolls.rename(columns={
            f"{col}_rolling_5":  f"{col}_away_rolling_5",
            f"{col}_rolling_10": f"{col}_away_rolling_10",
            f"{col}_avg":        f"{col}_away_avg",
        })

        # --- Merge both back safely ---
        team_games = (
            team_games
            .merge(home_rolls, on=["season", "team_id", "game_id"], how="left")
            .merge(away_rolls, on=["season", "team_id", "game_id"], how="left")
        )

    opp_merge = (
        team_games[
            [
                "season","game_id","team_id",
                # team-side home/away stats to copy over
                "team_goals_home_rolling_5","team_goals_home_rolling_10","team_goals_home_avg",
                "team_goals_away_rolling_5","team_goals_away_rolling_10","team_goals_away_avg",
                "team_goals_against_home_rolling_5","team_goals_against_home_rolling_10","team_goals_against_home_avg",
                "team_goals_against_away_rolling_5","team_goals_against_away_rolling_10","team_goals_against_away_avg",
                "team_goal_diff_home_rolling_5","team_goal_diff_home_rolling_10","team_goal_diff_home_avg",
                "team_goal_diff_away_rolling_5","team_goal_diff_away_rolling_10","team_goal_diff_away_avg",
            ]
        ]
        .rename(
            columns={
                "team_id": "opponent_id",
                "team_goals_home_rolling_5": "opp_goals_home_rolling_5",
                "team_goals_home_rolling_10": "opp_goals_home_rolling_10",
                "team_goals_home_avg": "opp_goals_home_avg",
                "team_goals_away_rolling_5": "opp_goals_away_rolling_5",
                "team_goals_away_rolling_10": "opp_goals_away_rolling_10",
                "team_goals_away_avg": "opp_goals_away_avg",
                "team_goals_against_home_rolling_5": "opp_goals_against_home_rolling_5",
                "team_goals_against_home_rolling_10": "opp_goals_against_home_rolling_10",
                "team_goals_against_home_avg": "opp_goals_against_home_avg",
                "team_goals_against_away_rolling_5": "opp_goals_against_away_rolling_5",
                "team_goals_against_away_rolling_10": "opp_goals_against_away_rolling_10",
                "team_goals_against_away_avg": "opp_goals_against_away_avg",
                "team_goal_diff_home_rolling_5": "opp_goal_diff_home_rolling_5",
                "team_goal_diff_home_rolling_10": "opp_goal_diff_home_rolling_10",
                "team_goal_diff_home_avg": "opp_goal_diff_home_avg",
                "team_goal_diff_away_rolling_5": "opp_goal_diff_away_rolling_5",
                "team_goal_diff_away_rolling_10": "opp_goal_diff_away_rolling_10",
                "team_goal_diff_away_avg": "opp_goal_diff_away_avg",
            }
        )
    )
    team_games = team_games.merge(
        opp_merge,
        on=["season","game_id","opponent_id"],
        how="left",
    )
    
    # --- Columns that already exist in df ---
    already_in_df = set(df.columns)

    # --- Columns to bring from team_games ---
    cols_to_merge = [
        "season", "team_id", "game_id",
    ] + [
        c for c in team_games.columns
        if c not in already_in_df  # exclude duplicates
    ]

    # --- Perform merge ---
    df = df.merge(
        team_games[cols_to_merge],
        on=["season", "team_id", "game_id"],
        how="left"
    )
    
    
    # Make sure no duplicate rows
    df = df.drop_duplicates(subset=["season", "game_id", "team_id", "player_id"])
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Team strength goals process complete at {ts}. Data saved to {OUT / 'df_team_strength_goals.parquet'}")
    
    df.to_parquet(OUT / "df_team_strength_goals.parquet", index=False)
    
if __name__ == "__main__":
    main()
    

