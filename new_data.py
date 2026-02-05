# Script to take in new data and prepare it for feature engineering
from pathlib import Path
import pandas as pd
from datetime import datetime

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data_collection"
    OUT = ROOT / "parquets"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Starting new data pipeline...")
    
    # Load data
    box_df = pd.read_csv(DATA / "2022-2026_box.csv")
    pbp_df = pd.read_csv(DATA / "2022-2026_pbp.csv")
    update_box_df = pd.read_csv(DATA / "update_box.csv")
    update_pbp_df = pd.read_csv(DATA / "update_pbp.csv")

    # Merge PBP and box
    df = pd.merge(
        box_df,
        pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )

    # Merge updated PBP and box
    update_df = pd.merge(
        update_box_df,
        update_pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )

    # Fix team change Arizona Coyotes to Utah Mammoth
    df.loc[df["team_id"] == 53, ["team_id", "team"]] = [68, "UTA"]
    df.loc[df["opponent_id"] == 53, ["opponent_id", "opponent"]] = [68, "UTA"]
    df.loc[df["team_id"] == 59, ["team_id", "team"]] = [68, "UTA"]
    df.loc[df["opponent_id"] == 59, ["opponent_id", "opponent"]] = [68, "UTA"]

    # Combine dataframes
    df = pd.concat([df, update_df], ignore_index=True)

    # Save to parquet
    df.to_parquet(OUT / "player_data.parquet", index=False)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] New data pipeline complete. Data saved to {OUT / 'player_data.parquet'}")
    
if __name__ == "__main__":
    main()