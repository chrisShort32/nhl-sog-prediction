import requests
from datetime import date, timedelta
import json
import time

def get_game_ids_for_season(start_date, end_date):
    """
    Loops through each date in the given range and collects all NHL game IDs.
    Returns a list of unique game IDs (regular season only).
    """
    game_ids = set()
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()

                # Iterate over all gameWeeks, not just the first one
                for week in data.get("gameWeek", []):
                    for game in week.get("games", []):
                        if game.get("gameType") == 2:  # 2 = regular season
                            game_ids.add(game["id"])

                print(f"{date_str}: {len(game_ids)} total games so far")

            else:
                print(f"HTTP {resp.status_code} for {date_str}")

        except Exception as e:
            print(f"Failed for {date_str}: {e}")

        current_date += timedelta(days=1)
        time.sleep(0.25)  # Prevent rate limiting

    return sorted(list(game_ids))



season_start = date(2022, 10, 1)
season_end = date(2026, 1, 7)

print("Fetching game IDs...")
game_ids = get_game_ids_for_season(season_start, season_end)
print(f"Found {len(game_ids)} regular-season games")

# Save to file
with open("nhl_game_ids_2022-2026.json", "w") as f:
    json.dump({"2022-2026": game_ids}, f, indent=2)

print("Saved game IDs to nhl_game_ids_2022-2026.json")
