import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo


ALT_MARKET_KEY = "player_shots_on_goal_alternate"
LOCAL_TZ = ZoneInfo("America/Chicago")


TEAM_ABBR = {
    "Anaheim Ducks": "ANA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    "Utah Mammoth": "UTA",
}


def team_to_abbr(name: str) -> str:
    return TEAM_ABBR.get(name, name)  # fallback to full name if unknown


def american_to_implied_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return None


def implied_prob_to_american(p: Optional[float]) -> Optional[int]:
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        odds = - (p / (1 - p)) * 100
    else:
        odds = ((1 - p) / p) * 100
    return int(round(odds))


def parse_iso_utc(s: Optional[str]) -> Optional[datetime]:
    """
    Parse '2026-01-14T01:10:00Z' -> aware datetime in UTC.
    """
    if not s:
        return None
    try:
        # 'Z' => UTC
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def is_today_local(commence_time_utc: Optional[str]) -> bool:
    dt_utc = parse_iso_utc(commence_time_utc)
    if dt_utc is None:
        return False
    dt_local = dt_utc.astimezone(LOCAL_TZ)
    today_local = datetime.now(LOCAL_TZ).date()
    return dt_local.date() == today_local


def parse_events_to_alt_rows_today(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Per-book alt rows, but ONLY for games whose commence_time is today in America/Chicago.
    """
    alt_rows: List[Dict[str, Any]] = []

    for ev in events:
        commence_time = ev.get("commence_time")

        # Filter out tomorrow/other days
        if not is_today_local(commence_time):
            continue

        event_id = ev.get("id", "")
        home_team = team_to_abbr(ev.get("home_team", ""))
        away_team = team_to_abbr(ev.get("away_team", ""))

        for book in ev.get("bookmakers", []) or []:
            book_key = book.get("key", "")
            markets = book.get("markets", []) or []

            alt_market = next((m for m in markets if m.get("key") == ALT_MARKET_KEY), None)
            if not alt_market:
                continue

            for out in alt_market.get("outcomes", []) or []:
                player = out.get("description") or out.get("player") or ""
                point = out.get("point")
                price = out.get("price")
                if not player or point is None:
                    continue

                try:
                    pt = float(point)
                except (TypeError, ValueError):
                    continue

                alt_rows.append({
                    "event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "book_key": book_key,
                    "player_name": player,
                    "point": pt,          # e.g. 1.5, 2.5, 3.5...
                    "price": price,
                    "imp_prob": american_to_implied_prob(price),
                })

    return alt_rows


def aggregate_alt_wide_mincols(alt_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Output one row per (event_id, player_name, home_team, away_team) with:
      odds_2p = Over 1.5
      odds_3p = Over 2.5
      odds_4p = Over 3.5
      odds_5p = Over 4.5
    Minimal columns only.
    """
    VALID_K = {2, 3, 4, 5}

    def point_to_kplus(pt: float) -> Optional[int]:
        # Keep only x.5 lines
        if abs(pt - (int(pt) + 0.5)) > 1e-9:
            return None
        return int(pt + 0.5)  # 1.5->2, 2.5->3, 3.5->4, 4.5->5, ...

    # collect implied probs per (event, player, matchup, k)
    bucket = defaultdict(lambda: {"probs": [], "meta": None})

    for r in alt_rows:
        k = point_to_kplus(r["point"])
        if k is None or k not in VALID_K:
            continue

        key = (r["event_id"], r["player_name"], r["home_team"], r["away_team"], k)
        b = bucket[key]

        if b["meta"] is None:
            b["meta"] = {
                "player_name": r["player_name"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "k": k,
            }

        p = r.get("imp_prob")
        if p is not None:
            b["probs"].append(p)

    # compute consensus odds per (event, player, k)
    per_point = []
    for _, b in bucket.items():
        avg_p = (sum(b["probs"]) / len(b["probs"])) if b["probs"] else None
        per_point.append({
            "player_name": b["meta"]["player_name"],
            "home_team": b["meta"]["home_team"],
            "away_team": b["meta"]["away_team"],
            "k": b["meta"]["k"],
            "avg_price": implied_prob_to_american(avg_p),
        })

    # pivot wide to minimal cols
    wide = {}
    for r in per_point:
        key = (r["player_name"], r["home_team"], r["away_team"])
        if key not in wide:
            wide[key] = {
                "player_name": r["player_name"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "odds_2p": None,
                "odds_3p": None,
                "odds_4p": None,
                "odds_5p": None,
            }

        wide[key][f"odds_{r['k']}p"] = r["avg_price"]

    out = list(wide.values())
    out.sort(key=lambda x: (x["home_team"], x["away_team"], x["player_name"]))
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"No rows for {path}, skipping.")
        return
    fieldnames = ["player_name", "home_team", "away_team", "odds_2p", "odds_3p", "odds_4p", "odds_5p"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# -------------------------
# Main: load cache, filter today, aggregate, write
# -------------------------
CACHE_DIR = Path("betting_lines_cache")

def main() -> None:
    all_events: List[Dict[str, Any]] = []
    for file in CACHE_DIR.glob("odds_*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to load {file.name}: {e}")
            continue

        if isinstance(data, list):
            all_events.extend(data)
        elif isinstance(data, dict):
            all_events.append(data)

    print(f"Total events loaded (raw): {len(all_events)}")

    alt_rows = parse_events_to_alt_rows_today(all_events)
    print(f"Alt rows (today only, per-book): {len(alt_rows)}")

    alt_wide = aggregate_alt_wide_mincols(alt_rows)
    print(f"Alt wide rows (today only): {len(alt_wide)}")
    today_str = datetime.now().strftime("%Y%m%d")
    write_csv(f"betting_lines/betting_lines_{today_str}.csv", alt_wide)
    print(f"Wrote betting_lines_{today_str}.csv")

if __name__ == "__main__":
    main()