"""
Transfer-window shortlist generator

Two ways to search (run from the project root):

1. Replacement search: players who play like a target
   python3 src/shortlist.py "Jarrod Bowen" --max-age 24 --max-value 30 --exclude-league "Premier League"

2. Role search: best players in a role, ranked by a feature
   python3 src/shortlist.py --role 2 --max-age 23 --sort assists_per90

Filters (all optional):
   --min-age --max-age              age (latest season)
   --min-value --max-value          current market value in EUR millions
   --max-ratio R                    value / model value; 0.5 = priced at half what
                                    the model expects (run value_model.py first)
   --max-contract-years N           contract ends within N years from today
   --sub-position "Centre-Back"     Transfermarkt position (repeat for several)
   --min-height --max-height        cm
   --foot left|right|both
   --league / --exclude-league      repeat for several
   --position DF|MF|FW  --min-minutes  --n

Add --out shortlist.csv to save the results.

Market value, contract and club are snapshots from July 2026
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, CLUSTER_NAMES
from similarity import MODEL_FILE, add_roles, build_profiles, find_player

ENRICHED_FILE = "data/processed/player_clusters_enriched.csv"
GAPS_FILE = "data/processed/player_value_gaps.csv"   # from value_model.py
TM_COLS = ["tm_sub_position", "height_in_cm", "foot", "value_latest",
           "contract_expiration_date", "tm_current_club"]

LOW_MINUTES = 1500        
HYBRID_GAP = 0.3          
SET_PIECE_CM = 3.0        
SET_PIECE_MF = 4.5        
CENTRAL_MID = ("Central Midfield", "Defensive Midfield")

# Default ranking feature for a role search (matches the cluster ids in features.py)
DEFAULT_SORT = {
    0: "crosses_per90", 1: "goals_per90", 2: "assists_per90",
    3: "shots_per90", 4: "interceptions_per90", 5: "tackles_won_per90",
}

def attach_tm(profiles, df, mode):
    cols = [c for c in TM_COLS if c in df.columns]
    if mode == "season":
        out = profiles.copy()      
    else:
        last = df.sort_values("season").groupby("player").tail(1).set_index("player")[cols]
        out = profiles.merge(last, left_on="player", right_index=True, how="left")
    out = out.reset_index(drop=True)
    out["contract_expiration_date"] = pd.to_datetime(out["contract_expiration_date"])
    out["contract_years"] = ((out["contract_expiration_date"] - pd.Timestamp.today().normalize())
                             .dt.days / 365.25).round(2)
    out["value_m"] = (out["value_latest"] / 1e6).round(1)
    return attach_gaps(out, mode)


def attach_gaps(profiles, mode):
    if not os.path.exists(GAPS_FILE):
        return profiles
    g = pd.read_csv(GAPS_FILE)
    g["_wg"] = g["log_gap"] * g["minutes"]
    keys = ["player", "season"] if mode == "season" else ["player"]
    agg = g.groupby(keys)[["_wg", "minutes"]].sum()
    agg["value_ratio"] = np.exp(agg["_wg"] / agg["minutes"]).round(2)
    return profiles.merge(agg[["value_ratio"]].reset_index(), on=keys, how="left")


def risk_flags(row):
    flags = []
    if row["n_seasons"] == 1:
        flags.append("one season only")
    if row["minutes"] < LOW_MINUTES:
        flags.append("low minutes")
    if row["role_gap"] <= HYBRID_GAP:
        flags.append(f"hybrid ({row['role_2_name']})")

    sub = row.get("tm_sub_position")
    if pd.notna(sub):
        if sub in CENTRAL_MID and row["crosses_per90"] >= SET_PIECE_CM:
            flags.append("crosses may be set pieces")
    elif row["position_group"] == "MF" and row["crosses_per90"] >= SET_PIECE_MF:
        flags.append("crosses may be set pieces")

    cy = row.get("contract_years")
    if pd.notna(cy):
        if cy < 0:
            flags.append("contract expired per July 2026 data (verify)")
        elif cy <= 1:
            flags.append("contract ends within 12 months")
    if pd.isna(row.get("value_latest")):
        flags.append("no market value")
    if pd.isna(row.get("age")):
        flags.append("age unknown")
    return "; ".join(flags)


def filter_pool(profiles, min_age=None, max_age=None, leagues=None,
                exclude_leagues=None, position=None, min_minutes=0, roles=None,
                min_value=None, max_value=None, max_contract_years=None,
                sub_positions=None, min_height=None, max_height=None, foot=None,
                max_ratio=None):
    keep = profiles["minutes"] >= min_minutes
    if min_age is not None:
        keep &= profiles["age"] >= min_age
    if max_age is not None:
        keep &= profiles["age"] <= max_age
    if leagues:
        keep &= profiles["league"].isin(leagues)
    if exclude_leagues:
        keep &= ~profiles["league"].isin(exclude_leagues)
    if position:
        keep &= profiles["position_group"] == position
    if roles is not None:
        keep &= profiles["role_1"].isin(roles)
    if min_value is not None:
        keep &= profiles["value_m"] >= min_value
    if max_value is not None:
        keep &= profiles["value_m"] <= max_value
    if max_contract_years is not None:   # future contracts only; expired ones are excluded
        keep &= profiles["contract_years"].between(0, max_contract_years)
    if sub_positions:
        keep &= profiles["tm_sub_position"].isin(sub_positions)
    if min_height is not None:
        keep &= profiles["height_in_cm"] >= min_height
    if max_height is not None:
        keep &= profiles["height_in_cm"] <= max_height
    if foot:
        keep &= profiles["foot"] == foot
    if max_ratio is not None:            # priced at most this multiple of the model value
        keep &= profiles["value_ratio"] <= max_ratio
    return keep


OUT_COLS = ["player", "team", "league", "age", "tm_sub_position", "height_in_cm",
            "role_1_name", "minutes", "value_m", "contract_years"]


def tidy(out, extra):
    out = out.copy()
    out["age"] = out["age"].astype("Int64")
    out["flags"] = out.apply(risk_flags, axis=1)
    ratio = ["value_ratio"] if "value_ratio" in out.columns else []
    cols = OUT_COLS + ratio + extra + ["flags"]
    return out[cols].rename(columns={
        "tm_sub_position": "sub_position", "height_in_cm": "height",
        "value_m": "value_eur_m"})


def replacement_shortlist(profiles, Xs, idx, n=15, **filters):
    dist = np.linalg.norm(Xs - Xs[idx], axis=1)
    keep = filter_pool(profiles, **filters)
    keep &= profiles["player"] != profiles.loc[idx, "player"]
    out = profiles[keep].copy()
    out["distance"] = dist[keep.values]
    return tidy(out.nsmallest(n, "distance"), ["distance"])


def role_shortlist(profiles, role, sort_by=None, n=15, **filters):
    sort_by = sort_by or DEFAULT_SORT[role]
    keep = filter_pool(profiles, roles=[role], **filters)
    return tidy(profiles[keep].nlargest(n, sort_by), [sort_by])


def main():
    ap = argparse.ArgumentParser(description="Transfer shortlist generator")
    ap.add_argument("player", nargs="?", help="target player (replacement search)")
    ap.add_argument("--role", type=int, help="role id 0-5 (role search)")
    ap.add_argument("--sort", choices=FEATURES, help="ranking feature for role search")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--mode", choices=["combined", "season"], default="combined")
    ap.add_argument("--min-age", type=int)
    ap.add_argument("--max-age", type=int)
    ap.add_argument("--min-value", type=float, help="EUR millions")
    ap.add_argument("--max-value", type=float, help="EUR millions")
    ap.add_argument("--max-ratio", type=float,
                    help="value vs model: 0.5 = priced at half of what the model expects")
    ap.add_argument("--max-contract-years", type=float)
    ap.add_argument("--sub-position", action="append", help="e.g. 'Centre-Back'; repeatable")
    ap.add_argument("--min-height", type=int)
    ap.add_argument("--max-height", type=int)
    ap.add_argument("--foot", choices=["left", "right", "both"])
    ap.add_argument("--league", action="append", help="repeat to allow several")
    ap.add_argument("--exclude-league", action="append")
    ap.add_argument("--position", choices=["DF", "MF", "FW"])
    ap.add_argument("--min-minutes", type=int, default=0)
    ap.add_argument("--out", help="save results to this CSV path")
    args = ap.parse_args()

    if (args.player is None) == (args.role is None):
        sys.exit("Give either a target player or --role (not both).")

    try:
        df = pd.read_csv(ENRICHED_FILE)
    except FileNotFoundError:
        sys.exit(f"{ENRICHED_FILE} not found. Run: python3 src/join_transfermarkt.py")
    profiles = attach_tm(build_profiles(df, args.mode), df, args.mode)
    profiles, Xs = add_roles(profiles, joblib.load(MODEL_FILE))

    filters = dict(min_age=args.min_age, max_age=args.max_age,
                   leagues=args.league, exclude_leagues=args.exclude_league,
                   position=args.position, min_minutes=args.min_minutes,
                   min_value=args.min_value, max_value=args.max_value,
                   max_contract_years=args.max_contract_years,
                   sub_positions=args.sub_position, min_height=args.min_height,
                   max_height=args.max_height, foot=args.foot,
                   max_ratio=args.max_ratio)

    if args.player:
        matches = find_player(profiles, args.player)
        if matches.empty:
            sys.exit(f"No player matching '{args.player}'")
        if matches["player"].nunique() > 1:
            print("Several players match, be more specific:")
            print(matches[["player", "team", "season"]].drop_duplicates().to_string(index=False))
            sys.exit()
        idx = matches.sort_values("season").index[-1]
        t = profiles.loc[idx]
        val = f"€{t['value_m']:.0f}M" if pd.notna(t["value_m"]) else "no value"
        print(f"\nTarget: {t['player']} ({t['team']}, {t['league']}, age {t['age']:.0f}, "
              f"{t['tm_sub_position']}, {val}; {t['role_1_name']}, second role {t['role_2_name']})")
        res = replacement_shortlist(profiles, Xs, idx, n=args.n, **filters)
    else:
        print(f"\nRole search: {CLUSTER_NAMES[args.role]}")
        res = role_shortlist(profiles, args.role, args.sort, n=args.n, **filters)

    print(f"\nShortlist ({len(res)} players):")
    print(res.round(2).to_string(index=False))
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
