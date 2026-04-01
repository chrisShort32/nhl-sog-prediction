import os, time, json, requests
import pathlib


BASE_DIR = pathlib.Path(__file__).resolve().parent

def cached_request(url, game_cache):
    """Fetch JSON data with caching to local disk."""
    fname = os.path.join(game_cache, url.split("/")[-2] + "_" + url.split("/")[-1].replace("/", "_"))
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
        
        game_state = data.get("gameState")
        if game_state == 'FUT':
            return {}
    except Exception:
        print(f"Invalid JSON for {url}")
        return {}
    with open(fname, "w") as f:
        json.dump(data, f)
    time.sleep(0.2)
    return data


# Boxscore
def get_boxscore_data(game_id, game_cache):
    """Fetch boxscore data and extract skater info (forwards + defense)."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    os.makedirs(game_cache, exist_ok=True)
        
    return cached_request(url, game_cache)

    
# Play-by-play
def get_play_by_play_from_game_id(game_id, game_cache):
    """Fetch raw play-by-play data for the game."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"    
    os.makedirs(game_cache, exist_ok=True)
        
    return cached_request(url, game_cache)
   