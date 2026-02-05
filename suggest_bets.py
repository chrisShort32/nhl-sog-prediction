# This script is dedicated to comparing predictions to betting lines and deciding which bets to take

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    
    PRED_DIR = Path(ROOT /"predictions")
    LINES_DIR = Path(ROOT /"betting_lines")
    OUT = Path(ROOT /"suggested_bets")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting suggest bets process at {ts}...")
    
    today_str = datetime.now().strftime("%Y%m%d")
    pred_file = PRED_DIR/(f"preds_{today_str}.csv")
    predictions = pd.read_csv(pred_file)
    
    line_file = LINES_DIR/f"betting_lines_{today_str}.csv"
    betting_lines = pd.read_csv(line_file)

    predictions = predictions[predictions["player_id"] != 8483678]  # Ignore Elias Pettersson the Defenseman
    merged = predictions.merge(
        betting_lines,
        on=["player_name"],
        how="inner"
    )


    # Compute implied probability from odds

    VIG = 0.06

    def american_to_prob(odds):
        if pd.isna(odds):
            return np.nan
        odds = float(odds)
        if odds < 0:
            return abs(odds) / (abs(odds) + 100.0)
        return 100.0 / (odds + 100.0)
    
    def american_to_dec(odds):
        if pd.isna(odds):
            return np.nan
        odds = float(odds)
        if odds < 0:
            return 1 - (100.0/odds)
        return 1 + (odds/100.0)

    def prob_to_american(p):
        if pd.isna(p):
            return np.nan
        p = float(p)
        p = min(max(p, 1e-6), 1 - 1e-6)
        if p >= 0.5:
            return -100.0 * p / (1.0 - p)
        return 100.0 * (1.0 - p) / p

    def estimate_under_odds(over_odds, vig=VIG):
        """Estimate under odds from over odds using a vig assumption."""
        p_over_imp = american_to_prob(over_odds)
        if pd.isna(p_over_imp):
            return np.nan
        p_under_imp = (1.0 + vig) - p_over_imp
        p_under_imp = min(max(p_under_imp, 1e-6), 1 - 1e-6)
        return prob_to_american(p_under_imp)


    for col in ["odds_2p","odds_3p","odds_4p", "odds_5p"]:
        if col in merged.columns:
            merged[f"imp_{col[-2:]}"] = merged[col].apply(american_to_prob)


    # Compute edge from model predictions
    merged["edge_2p"] = merged["p_ge2"] - merged["imp_2p"]
    merged["edge_3p"] = merged["p_ge3"] - merged["imp_3p"]
    merged["edge_4p"] = merged["p_ge4"] - merged["imp_4p"]
    merged["edge_5p"] = merged["p_ge5"] - merged["imp_5p"]

    merged["diff_p2p3"] = (merged["p_ge2"] - merged["p_ge3"]).clip(lower=0)
    merged["diff_p3p4"] = (merged["p_ge3"] - merged["p_ge4"]).clip(lower=0)

    eps = 1e-9
    p2 = merged["p_ge2"].clip(eps,1)
    p3 = merged["p_ge3"].clip(eps,1)
    p4 = merged["p_ge4"].clip(eps,1)

    merged["r_3_2"] = p3/p2
    merged["r_4_3"] = p4/p3

    merged["lr_3_2"] = np.log(p3/p2)
    merged["lr_4_3"] = np.log(p4/p3)
    
    RULES = {
        # --- global ---
        "vig": 0.06,

        # value = strong +EV on OVER
        "value_edge_min": 0.06,
        "value_p_min": 0.50,
        "value_odds_max": 160,

        # singles
        "single_p_min": 0.53,
        "single_odds_min": -150,
        "single_odds_max": 160,
        "single_edge_min": 0,

        # parlay fodder = very high p, price not too insane
        "parlay_p_min": 0.65,          
        "parlay_odds_min": -200,
        "parlay_odds_max": 100,

        # under
        "under_edge_min": 0,          
        "under_p_under_min": 0.53,
        "under_odds_min": -150,
        "under_odds_max": 160,

        # avoid = no action (used when lines are bad / model is unsure)
        "avoid_over_edge_max": -0.12,     # over is meaningfully overpriced
        "avoid_mid_p_low": 0.42,
        "avoid_mid_p_high": 0.54,
    }
    
    MARKETS = ["2p", "3p", "4p", "5p"]
    rows = []

    for m in MARKETS:
        k = m[0]  # '2','3','4','5'

        p_col = f"p_ge{k}"
        odds_col = f"odds_{m}"
        imp_col = f"imp_{m}"
        edge_col = f"edge_{m}"

        tmp = merged[[
            "game_id", "player_id", "player_name", "team", "opponent", "is_home",
            p_col, odds_col, imp_col, edge_col
        ]].copy()

        tmp.rename(columns={
            p_col: "p_over",
            odds_col: "odds_over",
            imp_col: "imp_over",
            edge_col: "edge_over",
        }, inplace=True)

        tmp["market"] = m
        tmp["threshold"] = int(k)

        # Model under prob
        tmp["p_under"] = 1.0 - tmp["p_over"]

        # Estimate under odds + implied under prob
        tmp["odds_under_est"] = tmp["odds_over"].apply(lambda x: estimate_under_odds(x, vig=RULES["vig"]))
        tmp["imp_under_est"] = tmp["odds_under_est"].apply(american_to_prob)

        # Under "edge" (estimated)
        tmp["edge_under_est"] = tmp["p_under"] - tmp["imp_under_est"]

        rows.append(tmp)

    markets_long = pd.concat(rows, ignore_index=True)
    
    # Conditions
    is_under = (
        (markets_long["edge_under_est"] >= RULES["under_edge_min"]) &
        (markets_long["p_under"] >= RULES["under_p_under_min"]) &
        (markets_long["odds_under_est"] >= RULES["under_odds_min"]) &
        (markets_long["odds_under_est"] <= RULES["under_odds_max"])
    )

    is_value = (
        (markets_long["edge_over"] >= RULES["value_edge_min"]) &
        (markets_long["p_over"] >= RULES["value_p_min"]) &
        (markets_long["odds_over"] <= RULES["value_odds_max"])
    )

    is_parlay = (
        (markets_long["p_over"] >= RULES["parlay_p_min"]) &
        (markets_long["odds_over"] >= RULES["parlay_odds_min"]) &
        (markets_long["odds_over"] <= RULES["parlay_odds_max"])
    )

    is_single = (
        (markets_long["p_over"] >= RULES["single_p_min"]) &
        (markets_long["odds_over"] >= RULES["single_odds_min"]) &
        (markets_long["odds_over"] <= RULES["single_odds_max"]) &
        (markets_long["edge_over"] >= RULES["single_edge_min"]) 
    )

    is_avoid = (
        (markets_long["edge_over"] <= RULES["avoid_over_edge_max"]) |
        (markets_long["p_over"].between(RULES["avoid_mid_p_low"], RULES["avoid_mid_p_high"]))
    )

    markets_long["bet_type"] = np.select(
        [is_under, is_value, is_single, is_parlay, is_avoid],
        ["under", "value", "single", "parlay", "avoid"],
        default="lean"
    )

    markets_long["side"] = np.where(markets_long["bet_type"] == "under", "under", "over")

    markets_long["bet_odds"] = np.where(
        markets_long["side"] == "under",
        markets_long["odds_under_est"],
        markets_long["odds_over"]
    )
        
    markets_long["bet_odds_d"] = markets_long["bet_odds"].apply(american_to_dec)


    markets_long["bet_imp"] = np.where(
        markets_long["side"] == "under",
        markets_long["imp_under_est"],
        markets_long["imp_over"]
    )

    markets_long["bet_p"] = np.where(
        markets_long["side"] == "under",
        markets_long["p_under"],
        markets_long["p_over"]
    )

    markets_long["bet_edge"] = np.where(
        markets_long["side"] == "under",
        markets_long["edge_under_est"],
        markets_long["edge_over"]
    )

    feat_cols = [
        "game_id",
        "player_id",
        "player_name",
        "diff_p2p3",
        "diff_p3p4",
        "r_3_2",
        "r_4_3",
        "lr_3_2",
        "lr_4_3",
    ]

    markets_long = markets_long.merge(
        merged[feat_cols],
        on=["game_id", "player_id", "player_name"],
        how="left"
    )
    
    final = markets_long.copy()

    # optional: sort for readability
    final = final.sort_values(
        ["game_id", "bet_type", "market", "bet_edge", "bet_p"],
        ascending=[True, True, True, False, False]
    ).reset_index(drop=True)
    
    today_str = datetime.now().strftime("%Y%m%d")
    out_path = f"{OUT}/suggested_bets_full_{today_str}.csv"
    final.to_csv(out_path, index=False)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Suggest bets process complete at {ts}. Data saved to {out_path}")
    
if __name__ == "__main__":
    main()