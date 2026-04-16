import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAYER_DATA = ROOT / "data_collection"
OUT = ROOT / "dashboard_data/latest"
V4_DATA = ROOT / "model_artifacts_v4/player_latest_v4.parquet"

def preprocess_data():
    old_pbp_df = pd.read_csv(PLAYER_DATA / "2022-2026_pbp.csv")
    old_box_df = pd.read_csv(PLAYER_DATA / "2022-2026_box.csv")
    update_pbp_df = pd.read_csv(PLAYER_DATA / "update_pbp.csv")
    update_box_df = pd.read_csv(PLAYER_DATA / "update_box.csv")
    
    old_df = pd.merge(
        old_box_df, old_pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )
    
    update_df = pd.merge(
        update_box_df, update_pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )
    
    df = pd.concat([old_df, update_df], ignore_index=True)
    df["logo_path"] = "dashboard_data/team_logos/" + df["team"] + "_dark.svg"
    df = df[df["season"] > 20242025]
    
    df.to_parquet(OUT / "processed_player_data.parquet")
    print(f"Processed {len(df)} rows and saved to processed_player_data.parquet")
    # do we need processed_player_data anymore? doesnt v3 have all the relevant info and then some?
    # it probably doesnt cost much to leave it just in case
    
    v4_df = pd.read_parquet(V4_DATA)
    v4_df.to_parquet(OUT / "player_latest_v4.parquet", index=False)
    print(f"Copied {len(v4_df)} rows from player_latest_v4.parquet to player_latest_v4.parquet")

    try:
        todays_games = pd.read_csv(ROOT / "data_collection/todays_games.csv")
    except pd.errors.EmptyDataError:
        todays_games = pd.DataFrame()
        print("No games today — CSV is empty")
    todays_games.to_parquet(OUT / "todays_games.parquet")

if __name__ == "__main__":
    preprocess_data()