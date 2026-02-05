from .the_big_loop import process_all_games as fetch_game_data
from .parse_box_score import main as parse_box_scores
from .parse_play_by_play import main as parse_play_by_plays
from .get_todays_games import get_games as get_todays_games
from .get_lines import main as fetch_betting_lines
from .aggregate_lines import main as aggregate_betting_lines

def run_step(name, func):
    print(f"\n--- {name} ---")
    try:
        func()
    except Exception as e:
        print(f"Error during {name}: {e}")

def main():
    run_step("Processing yesterdays game data", fetch_game_data)

    run_step("Parsing boxscore data", parse_box_scores)

    run_step("Parsing play-by-play data", parse_play_by_plays)

    run_step("Getting today's games", get_todays_games)

    run_step("Fetching betting lines", fetch_betting_lines)

    run_step("Aggregating betting lines", aggregate_betting_lines)

    print("\nData collection and processing complete.")
    
if __name__ == "__main__":
    main()