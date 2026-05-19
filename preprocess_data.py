import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAYER_DATA = ROOT / "data_collection"
OUT = ROOT / "dashboard_data/latest"
V5_DATA = ROOT / "model_artifacts_v5/player_latest_v5.parquet"

def preprocess_data():
    old_pbp_df = pd.read_csv(PLAYER_DATA / "2022-2026_pbp.csv")
    old_box_df = pd.read_csv(PLAYER_DATA / "2022-2026_box.csv")
    update_pbp_df = pd.read_csv(PLAYER_DATA / "update_pbp.csv")
    update_box_df = pd.read_csv(PLAYER_DATA / "update_box.csv")
    playoff_pbp_df = pd.read_csv(PLAYER_DATA / "playoff_pbp.csv")
    playoff_box_df = pd.read_csv(PLAYER_DATA / "playoff_box.csv")
    
    old_df = pd.merge(
        old_box_df, old_pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )
    old_df["is_playoffs"] = 0
    
    update_df = pd.merge(
        update_box_df, update_pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )
    update_df["is_playoffs"] = 0
    playoff_df = pd.merge(
        playoff_box_df,
        playoff_pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )
    playoff_df["is_playoffs"] = 1
    
    df = pd.concat([old_df, update_df, playoff_df], ignore_index=True)
    df["logo_path_dark"] = "dashboard_data/team_logos/" + df["team"] + "_dark.svg"
    df["logo_path"] = "dashboard_data/team_logos/" + df["team"] + ".svg"
    df = df[df["season"] > 20242025]
    
    ## Fix team change Arizona Coyotes to Utah Mammoth
    df.loc[df["team_id"] == 53, ["team_id", "team"]] = [68, "UTA"]
    df.loc[df["opponent_id"] == 53, ["opponent_id", "opponent"]] = [68, "UTA"]
    df.loc[df["team_id"] == 59, ["team_id", "team"]] = [68, "UTA"]
    df.loc[df["opponent_id"] == 59, ["opponent_id", "opponent"]] = [68, "UTA"]
    
    
    df.to_parquet(OUT / "processed_player_data.parquet")
    print(f"Processed {len(df)} rows and saved to processed_player_data.parquet")


    v5_df = pd.read_parquet(V5_DATA)
    v5_df.to_parquet(OUT / "player_latest_v5.parquet", index=False)
    print(f"Copied {len(v5_df)} rows from player_latest_v5.parquet to player_latest_v5.parquet")

    try:
        todays_games = pd.read_csv(ROOT / "data_collection/todays_games.csv")
    except pd.errors.EmptyDataError:
        todays_games = pd.DataFrame()
        print("No games today — CSV is empty")
    todays_games.to_parquet(OUT / "todays_games.parquet")

if __name__ == "__main__":
    preprocess_data()