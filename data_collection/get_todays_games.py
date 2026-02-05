import requests
from datetime import datetime
from pathlib import Path
import csv, json

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_ROOT / "todays_games.csv"

def get_games():
    today = datetime.now().date().isoformat()
    url = f"https://api-web.nhle.com/v1/score/{today}"
    r = requests.get(url)
    data = r.json()
    
    game_info = []
    games = data.get("games", [])
    if not games:
        print("No games found for today.")
        return
    
    for game in games:
        game_id = game.get("id")
        season = game.get("season")
        game_date = game.get("gameDate")
        away_team = game.get("awayTeam", {}).get("abbrev")
        away_team_record = game.get("awayTeam", {}).get("record")
        wins, losses, otl = away_team_record.split("-") if away_team_record else ("0", "0", "0")
        home_team = game.get("homeTeam", {}).get("abbrev")
        home_team_record = game.get("homeTeam", {}).get("record")
        wins_h, losses_h, otl_h = home_team_record.split("-") if home_team_record else ("0", "0", "0")
        start_utc = game.get("startTimeUTC")
        
        game_info.append({
            "game_id": game_id,
            "season": season,
            "game_date": game_date,
            "away_team": away_team,
            "away_wins": wins,
            "away_losses": losses,
            "away_otl": otl,
            "home_team": home_team,
            "home_wins": wins_h,
            "home_losses": losses_h,
            "home_otl": otl_h,
            "start_time_UTC": start_utc
        })
        
    # Write all player rows to CSV
    header_written = False
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as out_fh:
        writer = None
        if not header_written:
            if game_info:
                writer = csv.DictWriter(out_fh, fieldnames=game_info[0].keys())
                writer.writeheader()
                header_written = True
        for info in game_info:
            writer.writerow(info)
            
    print(f"Done! Wrote today's games to CSV: {OUTPUT_FILE}")
    
    with open(PROJECT_ROOT / "new_game_ids.json", "w") as f:
        json.dump({"new_game_ids": [game["game_id"] for game in game_info]}, f, indent=2   )
    
if __name__ == "__main__":
    get_games()
                
               