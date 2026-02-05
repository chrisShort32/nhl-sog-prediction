from datetime import datetime
import pandas as pd
from pathlib import Path
import numpy as np

def main() -> None:
    ROOT = Path(__file__).resolve().parent
    PRED_DIR = Path(ROOT / "predictions")
    BETS_DIR = Path(ROOT / "suggested_bets")
    OUT = Path(ROOT / "eval_outputs")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Starting tracking of betting results...")
    
    update_box_df = pd.read_csv(ROOT / "data_collection/update_box.csv")
    update_pbp_df = pd.read_csv(ROOT / "data_collection/update_pbp.csv")

    # Merge PBP and box
    df = pd.merge(
        update_box_df,
        update_pbp_df,
        on=["season", "game_id", "team_id", "player_id"],
        how="inner"
    )
    
    pred_files = list(PRED_DIR.glob("preds_*.csv"))
    pred_dfs = [pd.read_csv(f) for f in pred_files]
    predictions = pd.concat(pred_dfs, ignore_index=True)
    
    bet_files = list(BETS_DIR.glob("suggested_bets_full_*.csv"))
    bet_dfs = [pd.read_csv(f) for f in bet_files]
    bets = pd.concat(bet_dfs, ignore_index=True)
    
    actuals = (
        df[["game_id", "player_id", "player_name", "team", "shots_on_goal"]]
        .copy().rename(columns={"shots_on_goal": "actual_sog"})
    )
    
    pred_eval = predictions.merge(
        actuals[["game_id", "player_id", "player_name", "team", "actual_sog"]],
        on=["game_id", "player_id", "player_name", "team"],
        how="left",
    )
    
    bet_eval = bets.copy()
    bet_eval = bet_eval.merge(
        actuals[["game_id", "player_id", "player_name", "team", "actual_sog"]],
        on=["game_id", "player_id", "player_name", "team"],
        how="left",
    )
    

    
    def american_to_dec(odds):
        if pd.isna(odds):
            return np.nan
        odds = float(odds)
        if odds < 0:
            return 1 - (100/odds)
        return 1 + (odds/100)
        
    bet_eval["bet_odds_d"] = bet_eval["bet_odds"].apply(american_to_dec)


    ACTIONABLE = {"value", "single", "parlay", "under"}
    
    bet_eval = bet_eval[bet_eval["actual_sog"].notna()].copy()
    full_bet_eval = bet_eval.copy()
    bet_eval = bet_eval[bet_eval["bet_type"].isin(ACTIONABLE)].copy()
    
    bet_eval["side"] = bet_eval["side"].fillna("over")

    bet_eval["hit"] = np.where(
        bet_eval["side"].eq("over"),
        bet_eval["actual_sog"] >= bet_eval["threshold"],   # over wins
        bet_eval["actual_sog"] <  bet_eval["threshold"],   # under wins
    )

    full_bet_eval["hit"] = np.where(
        full_bet_eval["side"].eq("over"),
        full_bet_eval["actual_sog"] >= full_bet_eval["threshold"],
        full_bet_eval["actual_sog"] < full_bet_eval["threshold"],
    )

    def profit_1u(hit, odds):
        # Void / missing -> 0 profit (stake returned)
        if pd.isna(hit) or pd.isna(odds):
            return 0.0

        odds = float(odds)
        if not bool(hit):
            return -1.0

        # Win
        if odds > 0:
            return odds / 100.0
        else:
            return 100.0 / abs(odds)

    bet_eval["profit"] = [
        profit_1u(h, o) for h, o in zip(bet_eval["hit"], bet_eval["bet_odds"])
    ]

    full_bet_eval["profit"] = [
        profit_1u(h, o) for h, o in zip(full_bet_eval["hit"], full_bet_eval["bet_odds"])
    ]
    
    summary = {
        "bets": len(bet_eval[bet_eval["actual_sog"].notna()]),
        "hit_rate": bet_eval.loc[bet_eval["actual_sog"].notna(), "hit"].mean(),
        "profit_units": bet_eval["profit"].sum(),
        "roi_per_bet": bet_eval.loc[bet_eval["actual_sog"].notna(), "profit"].mean(),
    }

    by_type = (
        bet_eval.groupby("bet_type")
            .agg(
                bets=("profit","size"),
                profit=("profit","sum"),
                roi=("profit","mean"),
                hit_rate=("hit","mean"),
            )
            .sort_values("profit", ascending=False)
    )
    
    by_market = (
        bet_eval.groupby("threshold")
            .agg(
                bets=("hit","size"),
                hit_rate=("hit","mean"),
                avg_odds=("bet_odds_d","mean"),
                profit=("profit","sum"),
                roi=("profit","mean"),
            )
            .sort_index()
    )
    
    unders = (
        bet_eval[bet_eval["bet_type"] == "under"]
            .sort_values(["threshold", "bet_edge"], ascending=[True, False])
            .loc[:, [
                "game_id",
                "player_id",
                "player_name",
                "team",
                "opponent",
                "market",
                "threshold",
                "bet_type",
                "p_over",
                "p_under",
                "odds_under_est",
                "bet_odds_d",
                "imp_under_est",
                "edge_under_est",
                "actual_sog",
                "hit",
                "profit",
            ]]
    )
    
    overs = (
        bet_eval[bet_eval["bet_type"] != "under"]
            .sort_values(["threshold", "bet_edge"], ascending=[True, False])
            .loc[:, [
                "game_id",
                "player_id",
                "player_name",
                "team",
                "opponent",
                "market",
                "threshold",
                "bet_type",
                "p_over",
                "p_under",
                "odds_over",
                "bet_odds_d",
                "imp_over",
                "edge_over",
                "actual_sog",
                "hit",
                "profit",
            ]]
    )

    merged_overs = overs.merge(
        pred_eval,
        on=["game_id", "player_id", "player_name", "team", "opponent","actual_sog"],
        how="left",
    )


    eps = 1e-9
    p2 = merged_overs["p_ge2"].clip(eps,1)
    p3 = merged_overs["p_ge3"].clip(eps,1)
    p4 = merged_overs["p_ge4"].clip(eps,1)

    def ratio_buckets(merged):
        r = merged.to_numpy()
        qs = np.array([0.10, 0.21, 0.35, 0.51, 0.69, 1.00])
        quantile_targets = np.quantile(r, qs)
        
        unique_vals = np.unique(r)
        bucket_edges = np.array([
            unique_vals[np.searchsorted(unique_vals, q, side="left")]
            for q in quantile_targets
        ])
        
        bucket_edges = np.unique(bucket_edges)
        
        idx = np.searchsorted(bucket_edges, r, side="right")
        idx = np.clip(idx, 0, len(bucket_edges) - 1)
        floor_edges = np.r_[r.min(), bucket_edges[:-1]]
        
        return floor_edges[idx]
        
    merged_overs["r_3_2"] = p3/p2
    merged_overs["r32_bucket"] = ratio_buckets(merged_overs["r_3_2"] )
    merged_overs["r_4_3"] = p4/p3
    merged_overs["r43_bucket"] = ratio_buckets(merged_overs["r_4_3"])

    merged_overs["lr_3_2"] = np.log(p3/p2)
    merged_overs["lr_4_3"] = np.log(p4/p3)


    merged_overs.to_csv(OUT / "advanced_over_results.csv", index=False)
    
    merged_unders = unders.merge(
        pred_eval,
        on=["game_id", "player_id", "player_name", "team", "opponent","actual_sog"],
        how="left",
    )

    merged_unders["diff_p2p3"] = (merged_unders["p_ge2"] - merged_unders["p_ge3"]).clip(lower=0)
    merged_unders["diff_p3p4"] = (merged_unders["p_ge3"] - merged_unders["p_ge4"]).clip(lower=0)
    merged_unders["diff23_bucket"] = ratio_buckets(merged_unders["diff_p2p3"])
    merged_unders["diff34_bucket"] = ratio_buckets(merged_unders["diff_p3p4"])

    merged_unders.to_csv(OUT / "advanced_under_results.csv", index=False, header=True)
    
    full_bet_eval = full_bet_eval.merge(
        pred_eval,
        on=["game_id", "player_id","player_name", "team", "opponent", "is_home", "actual_sog"],
        how="left",
    )

    eps = 1e-9
    p2 = full_bet_eval["p_ge2"].clip(eps,1)
    p3 = full_bet_eval["p_ge3"].clip(eps,1)
    p4 = full_bet_eval["p_ge4"].clip(eps,1)
    p5 = full_bet_eval["p_ge5"].clip(eps,1)

    full_bet_eval["r_3_2"] = p3/p2
    full_bet_eval["r32_bucket"] = ratio_buckets(full_bet_eval["r_3_2"] )
    full_bet_eval["r_4_3"] = p4/p3
    full_bet_eval["r43_bucket"] = ratio_buckets(full_bet_eval["r_4_3"])

    full_bet_eval["diff_p2p3"] = (p2-p3).clip(lower=0)
    full_bet_eval["diff_p3p4"] = (p3-p4).clip(lower=0)
    full_bet_eval["diff_p4p5"] = (p4-p5).clip(lower=0)
    full_bet_eval["diff23_bucket"] = ratio_buckets(full_bet_eval["diff_p2p3"])
    full_bet_eval["diff34_bucket"] = ratio_buckets(full_bet_eval["diff_p3p4"])
    full_bet_eval["diff45_bucket"] = ratio_buckets(full_bet_eval["diff_p4p5"])


    full_bet_eval.to_csv(OUT / "full_bet_eval.csv", index=False)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Betting results tracking complete.")
    
    
    
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    run_date = pd.Timestamp.today().date()

    # 1) Summary dict -> 1-row df
    summary_df = pd.DataFrame([{
        "run_date": run_date,
        "run_ts": run_ts,
        "section": "summary",
        **summary
    }])

    # 2) by_type -> df
    by_type_df = by_type.reset_index().copy()
    by_type_df.insert(0, "section", "by_type")
    by_type_df.insert(0, "run_ts", run_ts)
    by_type_df.insert(0, "run_date", run_date)

    # 3) by_market -> df
    by_market_df = by_market.reset_index().copy()
    by_market_df.insert(0, "section", "by_market")
    by_market_df.insert(0, "run_ts", run_ts)
    by_market_df.insert(0, "run_date", run_date)

    # Combine into ONE CSV (overwrite each run)
    out_df = pd.concat([summary_df, by_type_df, by_market_df], ignore_index=True, sort=False)

    out_path = OUT / "betting_eval_summary.csv"
    out_df.to_csv(out_path, index=False)
    
if __name__ == "__main__":
    main()