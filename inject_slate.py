# inject_slate.py
# Appends skeleton rows for tonight's games into player_data.parquet.
# This ensures that when the feature pipeline runs, the .shift(1) rolling
# calculations on tonight's skeleton rows incorporate last night's actual stats.
#
# Must run AFTER new_data.py and BEFORE encode_categorical.py.
#
# The skeleton rows contain identity/schedule columns but zero values for
# all box-score stats. They are flagged with is_skeleton=True so they can
# be stripped before training.

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


def main() -> None:
    ROOT = Path(__file__).resolve().parent
    OUT = ROOT / "parquets"
    SLATE_CSV = ROOT / "data_collection/todays_games.csv"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Starting slate injection...")

    # ------------------------------------------------------------------
    # 1. Load existing player data (output of new_data.py)
    # ------------------------------------------------------------------
    df = pd.read_parquet(OUT / "player_data.parquet")

    # Flag existing rows so we can distinguish them later
    if "is_skeleton" not in df.columns:
        df["is_skeleton"] = False

    # Remove any skeleton rows left over from a previous run
    df = df[~df["is_skeleton"]].copy()

    # ------------------------------------------------------------------
    # 2. Load tonight's slate
    # ------------------------------------------------------------------
    slate_raw = pd.read_csv(SLATE_CSV)
    slate_raw["game_date"] = pd.to_datetime(slate_raw["game_date"], errors="coerce")
    slate_raw["start_time_UTC"] = pd.to_datetime(
        slate_raw["start_time_UTC"], utc=True, errors="coerce"
    )

    # Skip if slate is empty
    if slate_raw.empty:
        print(f"[{ts}] No games on today's slate. Nothing to inject.")
        df.to_parquet(OUT / "player_data.parquet", index=False)
        return

    # Build a team abbreviation -> team_id lookup from existing data
    team_id_lookup = (
        df[["team", "team_id"]]
        .drop_duplicates()
        .set_index("team")["team_id"]
        .to_dict()
    )

    # Build one row per team-side (away + home) from the slate
    away = slate_raw.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_wins": "team_win_pre", "away_losses": "team_loss_pre", "away_otl": "team_otl_pre",
        "home_wins": "opp_win_pre", "home_losses": "opp_loss_pre", "home_otl": "opp_otl_pre",
    }).assign(is_home=0)

    home = slate_raw.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_wins": "team_win_pre", "home_losses": "team_loss_pre", "home_otl": "team_otl_pre",
        "away_wins": "opp_win_pre", "away_losses": "opp_loss_pre", "away_otl": "opp_otl_pre",
    }).assign(is_home=1)

    slate = pd.concat([away, home], ignore_index=True)

    # Map abbreviations to numeric IDs
    slate["team_id"] = slate["team"].map(team_id_lookup)
    slate["opponent_id"] = slate["opponent"].map(team_id_lookup)

    # ------------------------------------------------------------------
    # 3. Build full-season roster
    #    Every player who has appeared for a team this season gets a
    #    skeleton row. We use each player's most recent game to pull
    #    their current team assignment (handles mid-season trades).
    #    This over-generates rows for injured/inactive players, but
    #    those extra predictions are harmless and get filtered downstream.
    # ------------------------------------------------------------------
    season_val = df["season"].max()

    season_df = df[df["season"] == season_val].copy()

    # Each player's most recent appearance determines their current team
    roster_pool = (
        season_df
        .sort_values(["player_id", "game_date", "start_time_UTC", "game_id"])
        .groupby("player_id", as_index=False)
        .tail(1)
    )

    # Keep only the identity columns we need to build skeleton rows
    IDENTITY_COLS = [
        "player_id", "player_name", "first_name", "last_name", "name",
        "team", "team_id", "position", "headshot_url",
    ]
    # Only keep columns that actually exist
    identity_cols_present = [c for c in IDENTITY_COLS if c in roster_pool.columns]
    roster = roster_pool[identity_cols_present].drop_duplicates(subset=["player_id"])

    # ------------------------------------------------------------------
    # 4. Cross-join roster with slate to create skeleton rows
    # ------------------------------------------------------------------
    # For each team on the slate, get its players
    skeleton_rows = []

    for _, game_row in slate.iterrows():
        team_abbr = game_row["team"]
        team_players = roster[roster["team"] == team_abbr].copy()

        if team_players.empty:
            print(f"  WARNING: No recent players found for {team_abbr}, skipping.")
            continue

        # Stamp each player row with tonight's game info
        team_players["game_id"] = game_row["game_id"]
        team_players["season"] = game_row["season"]
        team_players["game_date"] = game_row["game_date"]
        team_players["start_time_UTC"] = game_row["start_time_UTC"]
        team_players["opponent"] = game_row["opponent"]
        team_players["is_home"] = game_row["is_home"]
        team_players["venue"] = game_row.get("venue", np.nan)
        team_players["opponent_id"] = game_row["opponent_id"]

        skeleton_rows.append(team_players)

    if not skeleton_rows:
        print(f"[{ts}] Could not build any skeleton rows. Saving unchanged data.")
        df.to_parquet(OUT / "player_data.parquet", index=False)
        return

    skeletons = pd.concat(skeleton_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # 5. Fill box-score / stat columns with 0
    #    These are the raw stats that the pipeline reads. Setting them to 0
    #    is safe because every feature computation uses .shift(1), so these
    #    zeros are never included in any rolling window — they sit at the
    #    tail and get shifted past.
    # ------------------------------------------------------------------
    BOX_SCORE_COLS = [
        # Player box score
        "shots_on_goal", "blocked_shots", "goals", "assists", "points",
        "plus_minus", "power_play_goals", "hits", "pim",
        "shifts", "giveaways", "takeaways", "hits_taken",
        # Shot attempts (from PBP)
        "shot_attempts_total", "shot_attempts_blocked", "shot_attempts_missed",
        "pp_shots", "pp_shots_blocked", "pp_shots_missed", "pp_attempts_total",
        "pk_shots", "pk_shots_blocked", "pk_shots_missed", "pk_attempts_total",
        # Team-level game stats
        "team_shots", "team_goals", "team_shots_against", "team_goals_against",
        # Win/loss outcome flags
        "team_win", "team_loss", "team_otl",
    ]

    for col in BOX_SCORE_COLS:
        if col not in skeletons.columns:
            skeletons[col] = 0

    # TOI as "0:00" string (encode_categorical will convert to seconds)
    if "toi" not in skeletons.columns:
        skeletons["toi"] = "0:00"

    # Mark as skeleton
    skeletons["is_skeleton"] = True

    # ------------------------------------------------------------------
    # 6. Align columns and append
    # ------------------------------------------------------------------
    # Add any columns present in df but missing in skeletons (fill with NaN)
    for col in df.columns:
        if col not in skeletons.columns:
            skeletons[col] = np.nan

    # Keep only columns that exist in df (plus is_skeleton)
    skeletons = skeletons[[c for c in df.columns if c in skeletons.columns]].copy()

    # Align dtypes to match original df so parquet doesn't choke on mixed types
    for col in skeletons.columns:
        if col in df.columns:
            try:
                skeletons[col] = skeletons[col].astype(df[col].dtype)
            except (ValueError, TypeError):
                # Force Timestamp -> string for columns stored as strings in the original
                if df[col].dtype == object and hasattr(skeletons[col], "dt"):
                    skeletons[col] = skeletons[col].astype(str)

    combined = pd.concat([df, skeletons], ignore_index=True)

    n_skeletons = len(skeletons)
    n_games = slate["game_id"].nunique()
    print(f"[{ts}] Injected {n_skeletons} skeleton rows for {n_games} games.")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    combined.to_parquet(OUT / "player_data.parquet", index=False)

    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts2}] Slate injection complete.")


if __name__ == "__main__":
    main()