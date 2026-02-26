import json
import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    raise ValueError("Please set the ODDS_API_KEY environment variable.")
SPORT = "icehockey_nhl"
MARKETS = "player_shots_on_goal_alternate"  # "player_shots_on_goal" -- also available
REGIONS = "us"
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"

URL_BASE = "https://api.the-odds-api.com/v4/sports"

CACHE_DIR = "pipeline/betting_lines_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

#Event ids can be found in the id field in the response of the /events endpoint (see /v4/sports/{sports}/events).
#If the event has expired (not receiving updates due to completion or cancellation), a HTTP 404 status code will be returned.
EVENT_URL = URL_BASE + "/{SPORT}/events/?apiKey={API_KEY}"
# Get odds for event id
ODDS_URL = URL_BASE + "/{SPORT}/events/{eventId}/odds/?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&dateFormat={DATE_FORMAT}&oddsFormat={ODDS_FORMAT}"

def get_event_ids():
    url = EVENT_URL.format(SPORT=SPORT, API_KEY=API_KEY)
    r = requests.get(url)
    data = r.json()
    event_ids = [event['id'] for event in data]
    return event_ids, data

def get_odds_for_event(event_id: str):
    url = ODDS_URL.format(SPORT=SPORT, API_KEY=API_KEY, REGIONS=REGIONS, MARKETS=MARKETS, DATE_FORMAT=DATE_FORMAT, ODDS_FORMAT=ODDS_FORMAT, eventId=event_id)
    r = requests.get(url)
    data = r.json()
    return data

def main() -> None:
    event_ids, data = get_event_ids()
    with open(os.path.join(CACHE_DIR, "event_ids.json"), "w") as f:
        json.dump(event_ids, f, indent=2)
    with open(os.path.join(CACHE_DIR, "events_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"Found {len(event_ids)} events.")
    for event_id in event_ids:
        print(f"Fetching odds for event {event_id}...")
        odds_data = get_odds_for_event(event_id)
        with open(os.path.join(CACHE_DIR, f"odds_{event_id}.json"), "w") as f:
            json.dump(odds_data, f, indent=2)
            
    print("Done.")

if __name__ == "__main__":
    main()
    
    