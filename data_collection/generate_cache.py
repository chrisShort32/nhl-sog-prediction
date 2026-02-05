import os, time, json, requests
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "update_game_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_request(url):
    """Fetch JSON data with caching to local disk."""
    fname = os.path.join(CACHE_DIR, url.split("/")[-2] + "_" + url.split("/")[-1].replace("/", "_"))
    if os.path.exists(fname):
        with open(fname, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass  # Invalid cache → re-fetch

    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        print(f"Request failed ({r.status_code}) for {url}")
        return {}
    try:
        data = r.json()
    except Exception:
        print(f"Invalid JSON for {url}")
        return {}
    with open(fname, "w") as f:
        json.dump(data, f)
    time.sleep(0.2)
    return data


# Boxscore
def get_boxscore_data(game_id):
    """Fetch boxscore data and extract skater info (forwards + defense)."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    return cached_request(url)

    
# Play-by-play
def get_play_by_play_from_game_id(game_id):
    """Fetch raw play-by-play data for the game."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    return cached_request(url)
   