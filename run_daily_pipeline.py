from datetime import datetime
import traceback
import time

from new_data import main as new_data
from encode_categorical import main as encode_categorical
from feat_eng_player import main as feature_engineering_player
from team_strength_wins import main as team_strength_wins
from team_strength_goals import main as team_strength_goals
from misc_feats import main as misc_feats
from predict_today import main as predict_today
from suggest_bets import main as suggest_bets
from prediction_results_bets import main as prediction_results_bets
from prediction_results_all import main as prediction_results_all
from export_dashboard_parquets import main as export_dashboard_parquets
from preprocess_data import preprocess_data as preprocess_data

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_step(name: str, func) -> None:
    print(f"[{ts()}] Starting step: {name}...")
    start = time.time()
    try:
        func()
        duration = time.time() - start
        print(f"[{ts()}] Completed step: {name} ({duration:.2f}s).")
    except Exception:
        print(f"[{ts()}] ERROR in step: {name}")
        traceback.print_exc()
        raise
    
    
def main() -> None:
    run_step("New Data Collection", new_data)
    run_step("Categorical Encoding", encode_categorical)
    run_step("Feature Engineering - Player", feature_engineering_player)
    run_step("Team Strength - Wins", team_strength_wins)
    run_step("Team Strength - Goals", team_strength_goals)
    run_step("Miscellaneous Features", misc_feats)
    run_step("Prediction Results - All", prediction_results_all)
    run_step("Prediction Results - Bets", prediction_results_bets)
    run_step("Today's Predictions", predict_today)
    run_step("Suggest Bets", suggest_bets)
    run_step("Export Dashboard Parquets", export_dashboard_parquets)
    run_step("Preprocess Data for Dashboard", preprocess_data)
    
if __name__ == "__main__":
    main()