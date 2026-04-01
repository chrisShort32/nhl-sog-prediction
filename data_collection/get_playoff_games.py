import requests
from datetime import date, timedelta
import json
import time

def get_game_ids_for_season(seasons, series):
    """
    Loops through each season in the given range and collects all NHL playoff game IDs.
    Returns a list of unique game IDs (playoffs only).
    """
    
    game_ids = set()
    
    for season in seasons:
        for letter in series:
            url = f"https://api-web.nhle.com/v1/schedule/playoff-series/{season}/{letter}"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()

                    game_count = 0
                    for game in data.get("games", []):
                        game_ids.add(game["id"])
                        game_count += 1

                    print(f"{season}/{letter}: {game_count} games in series")

                else:
                    print(f"HTTP {resp.status_code} for {season}")

            except Exception as e:
                print(f"Failed for {season}: {e}")

            time.sleep(0.25)  # prevent rate limiting

    return sorted(list(game_ids))


print("Fetching game IDs...")
seasons = [20222023, 20232024, 20242025]
series = [
        'A','B','C','D','E','F','G','H', ## R1
        'I','J','K','L', ## R2
        'M','N', ## R3
        'O' ## R4 -- SCF
    ]
game_ids = get_game_ids_for_season(seasons, series)
print(f"Found {len(game_ids)} playoff games")

# Save to file
with open("playoff_game_ids.json", "w") as f:
    json.dump({"playoff_game_ids": game_ids}, f, indent=2)

print("Saved game IDs to playoff_game_ids.json")
