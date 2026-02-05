# This notebook takes the raw data from the scrape, cleans the data, and encodes categorical data
# Drops duplicate or redundant columns -- first_name, last_name, name
# Reorders the columns for better readability
# Encodes position
# Encodes TOI -- to seconds
# Extracts hour from game start time -- encodes into 3 buckets [early, primetime, late] for game start times
# Extracts and encodes day, month, year from game date
# Calculates days since season started and normalized days since start
# Saves df as -- df_encoded_base.parquet


from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    OUT = ROOT / "parquets"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Starting categorical encoding process...")
    
    # get the data
    df = pd.read_parquet(OUT / "player_data.parquet")
    # Convert to proper season format
    df["season"] = df["game_id"].astype(str).str[:4].astype(int)
    # Drop duplicate/redundant columns
    df = df.drop(columns=["first_name", "last_name", "name"])

    # Encode position
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[["position"]])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["position"]))
    df_encoded = pd.concat([df, encoded_df], axis=1)

    ## Create cumulative wins, losses, and OTL at the time of the game
    team_games = (
        df.groupby(["season", "game_id", "team"], as_index=False)
        [["game_date", "start_time_UTC", "team_win", "team_loss", "team_otl", "opponent"]]
        .max()
    )

    team_games["game_date"] = pd.to_datetime(team_games["game_date"], errors="coerce")
    team_games["start_time_UTC"] = pd.to_datetime(team_games["start_time_UTC"], errors="coerce")

    team_games = team_games.sort_values(["season", "team", "game_id"])

    for col, out in [("team_win", "team_wins_pre"),
                    ("team_loss", "team_losses_pre"),
                    ("team_otl", "team_otl_pre")]:
        team_games[out] = (
            team_games.groupby(["season", "team"])[col]
            .apply(lambda s: s.shift(1).cumsum())
            .reset_index(level=[0,1], drop=True)
            .fillna(0)
            .astype(int)
        )

    opp_pre = team_games[["season", "game_id", "team", "team_wins_pre", "team_losses_pre", "team_otl_pre"]].copy()
    opp_pre = opp_pre.rename(columns={
        "team": "opponent",
        "team_wins_pre": "opp_wins_pre",
        "team_losses_pre": "opp_losses_pre",
        "team_otl_pre": "opp_otl_pre",
    })

    team_games = team_games.merge(
        opp_pre,
        on=["season", "game_id", "opponent"],
        how="left"
    )

    df_encoded = df_encoded.merge(
        team_games[["season", "game_id", "team",
                "team_wins_pre", "team_losses_pre", "team_otl_pre",
                "opp_wins_pre", "opp_losses_pre", "opp_otl_pre"]],
        on=["season", "game_id", "team"],
        how="left"
    )

    # Encode TOI -- convert xx:xx to total seconds of ice time
    def convert_to_seconds(s):
        try:
            mins, secs = map(int, s.split(":"))
            return mins*60 + secs
        except Exception:
            return pd.NA

    df_encoded["toi_seconds"] = df_encoded["toi"].apply(convert_to_seconds).astype("Int64")

    # Encode start time -- extract hour
    df_encoded["start_time_UTC"] = pd.to_datetime(df_encoded["start_time_UTC"], errors="coerce", utc=True)
    df_encoded["game_start_hour"] = df_encoded["start_time_UTC"].dt.hour

    # Encode start times - early, prime time, late
    def categorize_start_hour(h):
        if h == 0:
            return "prime_time"
        elif h in [1,2,3]:
            return "late"
        elif h in range(13,24):
            return "early"
        else:
            return "unknown"

    df_encoded["game_start_bucket"] = df_encoded["game_start_hour"].apply(categorize_start_hour)

    # One hot encode buckets
    df_encoded = pd.get_dummies(df_encoded, columns=["game_start_bucket"], prefix="start")
    df_encoded = df_encoded.copy()
    # Encode game date
    df_encoded["game_date"] = pd.to_datetime(df_encoded["game_date"], errors="coerce")
    # Get month
    df_encoded["game_month"] = df_encoded["game_date"].dt.month
    # Get day
    df_encoded["game_weekday"] = df_encoded["game_date"].dt.weekday
    # Get days since start
    df_encoded["days_since_start"] = (
        df_encoded.groupby("season")["game_date"]
        .transform(lambda x: (x - x.min()).dt.days)
    )
    # Get normalized days since start
    df_encoded["day_of_season_normalized"] = (
        df_encoded.groupby("season")["days_since_start"]
        .transform(lambda x: x / x.max())
    )

    df_encoded.to_parquet(OUT / "df_encoded_base.parquet", index=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Categorical encoding complete. Data saved to {OUT / 'df_encoded_base.parquet'}")
    
if __name__ == "__main__":
    main()