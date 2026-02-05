import json, csv
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
GAME_CACHE = PROJECT_ROOT / "update_game_cache"
OUTPUT_FILE = PROJECT_ROOT / "update_pbp.csv"

def get_pbp_data(game_id):
    pbp_path = GAME_CACHE / f"{game_id}_play-by-play"
    pbp = json.loads(pbp_path.read_text(encoding="utf-8"))
    
    return pbp

def player_info(pbp):
    
    # Get player info -- ignore goalies
    roster_spots = pbp.get("rosterSpots")
    players = {}
    for player in roster_spots:
        if player.get("positionCode") == "G":
            continue
        
        pid = str(player.get("playerId"))
        
        players[pid]= { 
        "season": str(pbp.get("season")),
        "game_id": str(pbp.get("id")),
        "team_id": player.get("teamId"),
        "player_id": pid,
        "first_name": player.get("firstName").get("default"),
        "last_name": player.get("lastName").get("default"),
        "player_name": player.get("firstName").get("default") + " " + player.get("lastName").get("default"),
        "shot_attempts_total": 0,
        "shot_attempts_blocked": 0,
        "shot_attempts_missed": 0,
        "hits_taken": 0,
        "on_pp": 0,
        "on_pk": 0,
        "pp_shots": 0,
        "pp_shots_blocked": 0,
        "pp_shots_missed": 0,
        "pp_attempts_total": 0,
        "pk_shots": 0,
        "pk_shots_blocked": 0,
        "pk_shots_missed": 0,
        "pk_attempts_total": 0,
        }
        
    return players

def scrape_plays(pbp, players, goalie_games):
    
    events = pbp.get("plays", [])
    home_team = pbp.get("homeTeam").get("id")
    away_team = pbp.get("awayTeam").get("id")
    
    for event in events:
        ## ignore shootout stats
        if event.get("periodDescriptor").get("periodType") == "SO":
            continue
        
        ## get the event type
        event_type = event.get("typeDescKey")
        
        ## get the situation code -- 1451 = home pp, 1541 = away pp 
        situation = event.get("situationCode")
       
        ## determine if situation is power play
        if situation in ["1451", "1351", "1560"]:
            home_pp = True
        elif situation in ["1541", "1531", "0651"]:
            away_pp = True
        else:
            home_pp = False
            away_pp = False
        
       
        if event_type == "blocked-shot":
            shooter = str(event.get("details").get("shootingPlayerId"))
            if not shooter or shooter not in players:
                goalie_games.append({
                    "game_id": pbp.get("id"),
                    "goalie_id": shooter,
                    "event_type": event_type
                })
                continue
            ## determine which team is on pp and pk if any
            pp_team = ''
            pk_team = ''
            if home_pp:
                pp_team = home_team
                pk_team = away_team
            elif away_pp:
                pp_team = away_team
                pk_team = home_team
                

            ## increment shots attempts and blocked attempts
            players[shooter]["shot_attempts_blocked"] += 1
            players[shooter]["shot_attempts_total"] += 1
            
            ## increment pp shot attempts, flag on pp
            if players[shooter]["team_id"] == pp_team:
                players[shooter]["pp_shots_blocked"] += 1
                players[shooter]["pp_attempts_total"] += 1
                players[shooter]["on_pp"] = 1
            ## increment pk shot attempts, flag on pk
            elif players[shooter]["team_id"] == pk_team:  
                players[shooter]["pk_shots_blocked"] += 1
                players[shooter]["pk_attempts_total"] += 1
                players[shooter]["on_pk"] = 1
                
            
        elif event_type == "shot-on-goal":
            shooter = str(event.get("details").get("shootingPlayerId"))
            if not shooter or shooter not in players:
                goalie_games.append({
                    "game_id": pbp.get("id"),
                    "goalie_id": shooter,
                    "event_type": event_type
                })
                continue
            ## determine which team is on pp and pk if any
            pp_team = ''
            pk_team = ''
            if home_pp:
                pp_team = home_team
                pk_team = away_team
            elif away_pp:
                pp_team = away_team
                pk_team = home_team
                
            ## increment shots attempts
            players[shooter]["shot_attempts_total"] += 1
            
            ## increment pp shot attempts, flag on pp
            if players[shooter]["team_id"] == pp_team:
                players[shooter]["pp_shots"] += 1
                players[shooter]["pp_attempts_total"] += 1
                players[shooter]["on_pp"] = 1
            ## increment pk shot attempts, flag on pk
            elif players[shooter]["team_id"] == pk_team:  
                players[shooter]["pk_shots"] += 1
                players[shooter]["pk_attempts_total"] += 1
                players[shooter]["on_pk"] = 1
            
        elif event_type == "goal":
            shooter = str(event.get("details").get("scoringPlayerId"))
            if not shooter or shooter not in players:
                goalie_games.append({
                    "game_id": pbp.get("id"),
                    "goalie_id": shooter,
                    "event_type": event_type
                })
                continue
            ## determine which team is on pp and pk if any
            pp_team = ''
            pk_team = ''
            if home_pp:
                pp_team = home_team
                pk_team = away_team
            elif away_pp:
                pp_team = away_team
                pk_team = home_team

            ## increment shots attempts
            players[shooter]["shot_attempts_total"] += 1
            
            ## increment pp shot attempts, flag on pp
            if players[shooter]["team_id"] == pp_team:
                players[shooter]["pp_shots"] += 1
                players[shooter]["pp_attempts_total"] += 1
                players[shooter]["on_pp"] = 1
            ## increment pk shot attempts, flag on pk
            elif players[shooter]["team_id"] == pk_team:  
                players[shooter]["pk_shots"] += 1
                players[shooter]["pk_attempts_total"] += 1
                players[shooter]["on_pk"] = 1
        
        elif event_type == "missed-shot":
            shooter = str(event.get("details").get("shootingPlayerId"))
            if not shooter or shooter not in players:
                goalie_games.append({
                    "game_id": pbp.get("id"),
                    "goalie_id": shooter,
                    "event_type": event_type
                })
                continue
            ## determine which team is on pp and pk if any
            pp_team = ''
            pk_team = ''
            if home_pp:
                pp_team = home_team
                pk_team = away_team
            elif away_pp:
                pp_team = away_team
                pk_team = home_team

            ## increment shots attempts and missed shots
            players[shooter]["shot_attempts_missed"] += 1
            players[shooter]["shot_attempts_total"] += 1
            
            ## increment pp shot attempts, flag on pp
            if players[shooter]["team_id"] == pp_team:
                players[shooter]["pp_shots_missed"] += 1
                players[shooter]["pp_attempts_total"] += 1
                players[shooter]["on_pp"] = 1
            ## increment pk shot attempts, flag on pk
            elif players[shooter]["team_id"] == pk_team:  
                players[shooter]["pk_shots_missed"] += 1
                players[shooter]["pk_attempts_total"] += 1
                players[shooter]["on_pk"] = 1
                
        elif event_type == "hit":
            hittee = str(event.get("details").get("hitteePlayerId"))
            if not hittee or hittee not in players:
                goalie_games.append({
                    "game_id": pbp.get("id"),
                    "goalie_id": hittee,
                    "event_type": event_type
                })
                continue
            players[hittee]["hits_taken"] += 1
            
    return players         

# --- Main run ---

GOALIE_FILE = Path("update_goalie_event_games.csv")


def gather_pbp_game_ids() -> list[str]:
    """Return sorted game_ids for which a cached play-by-play file exists."""
    if not GAME_CACHE.exists():
        raise FileNotFoundError(f"Cache directory not found: {GAME_CACHE.resolve()}")

    # glob returns Path objects that already exist
    game_ids = sorted({p.name.split("_")[0] for p in GAME_CACHE.glob("*_play-by-play")})
    return game_ids


def write_pbp_csv(game_ids: Iterable[str]) -> None:
    game_ids = list(game_ids)
    print(f"Found {len(game_ids)} valid games to process.")
    print(f"Processing {len(game_ids)} games")

    header_written = False
    goalie_games: list[dict] = []

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as out_fh:
        writer = None

        for i, gid in enumerate(game_ids, start=1):
            print(f"[{i}/{len(game_ids)}] Parsing {gid}...")
            try:
                pbp = get_pbp_data(gid)
                roster = player_info(pbp)
                players = scrape_plays(pbp, roster, goalie_games)

                if not players:
                    continue

                if not header_written:
                    # players is a dict; grab one row dict to define header
                    header = list(next(iter(players.values())).keys())
                    writer = csv.DictWriter(out_fh, fieldnames=header)
                    writer.writeheader()
                    header_written = True

                writer.writerows(players.values())

            except Exception as e:
                print(f"Error processing {gid}: {e}")

    # Write goalie event log (optional)
    if goalie_games:
        with GOALIE_FILE.open("w", newline="", encoding="utf-8") as out_fh:
            w = csv.DictWriter(out_fh, fieldnames=goalie_games[0].keys())
            w.writeheader()
            w.writerows(goalie_games)
        print(f"Logged {len(goalie_games)} goalie-related shooting events → {GOALIE_FILE}")
    else:
        print("No goalie shooting events detected.")

    print(f"\nDone! Wrote combined CSV: {OUTPUT_FILE}")


def main() -> None:
    game_ids = gather_pbp_game_ids()
    write_pbp_csv(game_ids)


if __name__ == "__main__":
    main()

