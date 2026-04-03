import json, csv
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent


def get_boxscore_data(game_id, game_cache):
    box_path = game_cache / f"{game_id}_boxscore"
    box = json.loads(box_path.read_text(encoding="utf-8"))

    # --- Detect if shootout ---
    is_shootout = box.get("periodDescriptor", {}).get("periodType") == "SO"
    is_overtime = box.get("periodDescriptor", {}).get("periodType") == "OT"
       
    # --- Get game metadata ---
    season = box.get("season")
    game_date = box.get("gameDate")
    
    game_meta_data = {
        "venue_location": box.get("venueLocation", {}).get("default"),
        "venue": box.get("venue", {}).get("default"),
        "start_time_UTC": box.get("startTimeUTC"),
    }

    # --- Team-level setup ---
    team_data = {}
    for side in ["homeTeam", "awayTeam"]:
        team = box.get(side, {})
        if side == "homeTeam":
            op = box.get("awayTeam", {})
        else:
            op = box.get("homeTeam", {})
        
        team_id = team.get("id")
        team_abbrev = team.get("abbrev")
        team_logo = team.get("logo")
        team_score_box = team.get("score", 0)
        team_sog = team.get("sog", 0)
        team_sog_against = op.get("sog")
        opponent_score_box = op.get("score", 0)
        opponent_id = op.get("id")
        opponent = op.get("abbrev")
        opponent_logo = op.get("logo")
        team_win = 0
        team_otl = 0
        team_loss = 0
        opponent_win = 0
        opponent_otl = 0
        opponent_loss = 0
        
        if team_score_box < opponent_score_box:
            if is_overtime or is_shootout:
                team_otl = 1
                opponent_win = 1
            else:
                team_loss = 1
                opponent_win = 1
        else:
            if is_overtime or is_shootout:
                team_win = 1
                opponent_otl = 1
            else:
                team_win = 1
                opponent_loss = 1
                
        
        team_data[side] = {
            "id": team_id,
            "abbrev": team_abbrev,
            "logo": team_logo,
            "score_box": team_score_box,
            "team_shots": team_sog,
            "team_shots_against": team_sog_against,
            "team_goals_against": opponent_score_box,
            "team_win": team_win,
            "team_otl": team_otl,
            "team_loss": team_loss,
            "opponent_win": opponent_win,
            "opponent_otl": opponent_otl,
            "opponent_loss": opponent_loss,
            "opponent_id": opponent_id,
            "opponent": opponent,
            "opponent_logo": opponent_logo
        }

    # --- Adjust scores if shootout ---
    if is_shootout:
        true_goals = min(team_data["homeTeam"]["score_box"], team_data["awayTeam"]["score_box"])
        team_data["homeTeam"]["score_true"] = true_goals
        team_data["awayTeam"]["score_true"] = true_goals
    else:
        team_data["homeTeam"]["score_true"] = team_data["homeTeam"]["score_box"]
        team_data["awayTeam"]["score_true"] = team_data["awayTeam"]["score_box"]

    # --- Player-level info (forwards + defense) ---
    player_info = []
    for side in ["homeTeam", "awayTeam"]:
        team_id = team_data[side]["id"]
        team_abbrev = team_data[side]["abbrev"]
        opponent_id = team_data[side]["opponent_id"]
        opponent = team_data[side]["opponent"]
        team_score_true = team_data[side]["score_true"]
        team_shots = team_data[side]["team_shots"]
        team_shots_against = team_data[side]["team_shots_against"]
        if side == "homeTeam":
            team_goals_against = team_data["awayTeam"]["score_true"]
            is_home = 1
        else:
            team_goals_against = team_data["homeTeam"]["score_true"]
            is_home = 0
        team_players = box.get("playerByGameStats", {}).get(side, {})
        for group in ["forwards", "defense"]:
            for player in team_players.get(group, []):
                pid = player.get("playerId")
                if not pid:
                    continue
                player_info.append({
                    "season": str(season),
                    "game_id": str(game_id),
                    "game_date": game_date,
                    "player_id": str(pid),
                    "name": player["name"]["default"], # this is: (first initial).(last name)
                    "position": player.get("position"),
                    "team_id": team_id,
                    "team": team_abbrev,
                    "team_logo": team_data[side]["logo"],
                    "opponent_id": opponent_id,
                    "opponent": opponent,
                    "opponent_logo": team_data[side]["opponent_logo"],
                    "is_home": is_home,
                    "shots_on_goal": player.get("sog", 0),
                    "blocked_shots": player.get("blockedShots", 0),
                    "goals": player.get("goals", 0),
                    "assists": player.get("assists", 0),
                    "points": player.get("points", 0),
                    "plus_minus": player.get("plusMinus", 0),
                    "power_play_goals": player.get("powerPlayGoals", 0),
                    "hits": player.get("hits", 0),
                    "pim": player.get("pim", 0),
                    "toi": player.get("toi", 0),
                    "shifts": player.get("shifts", 0),
                    "giveaways": player.get("giveaways", 0),
                    "takeaways": player.get("takeaways", 0),
                    "team_shots": team_shots,
                    "team_goals": team_score_true,
                    "team_shots_against": team_shots_against,
                    "team_goals_against": team_goals_against,
                    "team_win": team_data[side]["team_win"],
                    "team_otl": team_data[side]["team_otl"],
                    "team_loss": team_data[side]["team_loss"],
                    "opponent_win": team_data[side]["opponent_win"],
                    "opponent_otl": team_data[side]["opponent_otl"],
                    "opponent_loss": team_data[side]["opponent_loss"],

                    **game_meta_data
                
                })
    return player_info


# --- Main run ---

def gather_game_ids(game_cache) -> list[str]:
    """Return sorted game_ids for which a cached play-by-play file exists."""
    if not game_cache.exists():
            raise FileNotFoundError(f"Cache directory not found: {game_cache.resolve()}")

    # glob returns Path objects that already exist
    game_ids = sorted({p.name.split("_")[0] for p in game_cache.glob("*_boxscore")})
    return game_ids


def write_boxscore_csv(game_ids: Iterable[str], game_cache, output_file) -> None:
    game_ids = list(game_ids)
    print(f"Found {len(game_ids)} valid games with boxscore")
    print(f"Processing {len(game_ids)} games")

    header_written = False
    with output_file.open("w", newline="", encoding="utf-8") as out_fh:
        writer = None

        for i, gid in enumerate(game_ids, start=1):
            print(f"[{i}/{len(game_ids)}] Parsing {gid}...")
            try:
                player_info = get_boxscore_data(gid, game_cache)
                if not player_info:
                    continue

                if not header_written:
                    header = list(player_info[0].keys())
                    writer = csv.DictWriter(out_fh, fieldnames=header)
                    writer.writeheader()
                    header_written = True

                writer.writerows(player_info)

            except Exception as e:
                print(f"Error processing {gid}: {e}")

    print(f"\nDone! Wrote combined CSV: {output_file}")


def main(playoffs) -> None:
    if playoffs:
        game_cache = PROJECT_ROOT / "playoff_game_cache"
        output_file = PROJECT_ROOT / "playoff_box.csv"
    else:
        game_cache = PROJECT_ROOT / "update_game_cache"
        output_file = PROJECT_ROOT / "update_box.csv"
    game_ids = gather_game_ids(game_cache)
    write_boxscore_csv(game_ids, game_cache, output_file)


if __name__ == "__main__":
    main()