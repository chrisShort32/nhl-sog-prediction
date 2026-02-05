import json, os
from generate_cache import get_boxscore_data, get_play_by_play_from_game_id

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_all_game_ids(filename):
    with open(os.path.join(BASE_DIR, filename), "r") as f:
        return json.load(f)


def process_all_games():
    all_game_ids = load_all_game_ids(filename="new_game_ids.json")
    game_ids = all_game_ids["new_game_ids"]
    total_new = 0
 
    print(f"\nProcessing ({len(game_ids)} games)...")
    

    for gid in game_ids:
        print(f"Game {gid}")
        try:
            stats = get_boxscore_data(gid)
            pbp = get_play_by_play_from_game_id(gid)
            if not stats or not pbp:
                print(f"No data for {gid}, skipping.")
                continue


        except Exception as e:
            print(f"Error with {gid}: {e}")

        total_new += 1

    print(f"\nFinished! {total_new} new games processed.")


if __name__ == "__main__":
    process_all_games()