import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLAYER_DATA = ROOT / "data_collection"
OUT = ROOT / "dashboard_data/latest"

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
    df["logo_path"] = "dashboard_data/team_logos/" + df["team"] + ".svg"
    df = df[df["season"] > 20242025]
    
    df.to_parquet(OUT / "processed_player_data.parquet")
    print(f"Processed {len(df)} rows and saved to processed_player_data.parquet")

if __name__ == "__main__":
    preprocess_data()