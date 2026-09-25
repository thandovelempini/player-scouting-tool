"""Similar-player search and role assignment.

Usage (from the project root):
    python3 src/similarity.py "Declan Rice"
    python3 src/similarity.py "Kai Havertz" --n 15 --league "Serie A"
    python3 src/similarity.py "Jarrod Bowen" --mode season --season 2025/26
"""
import argparse
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, CLUSTER_NAMES

DATA_FILE = "data/processed/player_clusters.csv"
MODEL_FILE = "models/clustering.joblib"


def norm(text):
    """Lowercase and strip accents so 'Guéhi' matches 'guehi'."""
    text = unicodedata.normalize("NFKD", str(text))
    return text.encode("ascii", "ignore").decode().lower().strip()


def build_profiles(df, mode="combined"):
    """One row per player ('combined': minutes-weighted average of both
    seasons) or one row per player-season ('season')."""
    if mode == "season":
        out = df.reset_index(drop=True).copy()
        out["n_seasons"] = 1
        return out

    d = df.sort_values("season")
    w = d["minutes"]
    by_player = d["player"]
    weighted = (
        d[FEATURES].mul(w, axis=0).groupby(by_player).sum()
        .div(w.groupby(by_player).sum(), axis=0)
    )
    # latest season's team, league and age (age rises by 1 each season)
    last_cols = ["team", "league", "position_group"] + (["age"] if "age" in d else [])
    last = d.groupby("player").tail(1).set_index("player")[last_cols]
    out = weighted.join(last)
    out["n_seasons"] = d.groupby("player")["season"].nunique()
    out["season"] = d.groupby("player")["season"].agg(lambda s: "+".join(sorted(s)))
    out["minutes"] = w.groupby(by_player).sum()
    return out.reset_index()


def add_roles(profiles, models):
    """Scale features, then find each player's closest and second-closest role."""
    X = models["imputer"].transform(profiles[FEATURES])
    Xs = models["scaler"].transform(X)
    dist = models["kmeans"].transform(Xs)
    order = np.argsort(dist, axis=1)
    rows = np.arange(len(profiles))

    profiles = profiles.copy()
    profiles["role_1"] = order[:, 0]
    profiles["role_2"] = order[:, 1]
    profiles["role_gap"] = dist[rows, order[:, 1]] - dist[rows, order[:, 0]]
    profiles["role_1_name"] = profiles["role_1"].map(CLUSTER_NAMES)
    profiles["role_2_name"] = profiles["role_2"].map(CLUSTER_NAMES)
    return profiles, Xs


def find_player(profiles, query, season=None):
    q = norm(query)
    names = profiles["player"].map(norm)
    mask = names == q
    if not mask.any():
        mask = names.str.contains(q, regex=False)
    matches = profiles[mask]
    if season is not None:
        matches = matches[matches["season"] == season]
    return matches


def similar_players(profiles, Xs, idx, n=10, league=None, position=None,
                    role=None, min_minutes=0, season=None):
    dist = np.linalg.norm(Xs - Xs[idx], axis=1)
    keep = profiles["player"] != profiles.loc[idx, "player"]
    keep &= profiles["minutes"] >= min_minutes
    if league:
        keep &= profiles["league"] == league
    if position:
        keep &= profiles["position_group"] == position
    if role is not None:
        keep &= profiles["role_1"] == role
    if season:
        keep &= profiles["season"] == season

    out = profiles[keep].copy()
    out["distance"] = dist[keep.values]
    cols = ["player", "team", "league", "season", "position_group",
            "role_1_name", "minutes", "distance"]
    return out.nsmallest(n, "distance")[cols]


def main():
    ap = argparse.ArgumentParser(description="Find similar players")
    ap.add_argument("player")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mode", choices=["combined", "season"], default="combined",
                    help="combined = average both seasons (default)")
    ap.add_argument("--season", help="e.g. 2025/26 (filters results; in season "
                                     "mode also picks the target's season)")
    ap.add_argument("--league")
    ap.add_argument("--position", choices=["DF", "MF", "FW"])
    ap.add_argument("--role", type=int, help="cluster id of results (0-5)")
    ap.add_argument("--min-minutes", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(DATA_FILE)
    models = joblib.load(MODEL_FILE)
    profiles, Xs = add_roles(build_profiles(df, args.mode), models)

    target_season = args.season if args.mode == "season" else None
    matches = find_player(profiles, args.player, target_season)
    if matches.empty:
        sys.exit(f"No player matching '{args.player}'")
    if matches["player"].nunique() > 1:
        print("Several players match, be more specific:")
        print(matches[["player", "team", "season"]].drop_duplicates().to_string(index=False))
        sys.exit()
    if len(matches) > 1:  # season mode without --season: use the latest season
        matches = matches.sort_values("season").tail(1)

    idx = matches.index[0]
    t = profiles.loc[idx]
    print(f"\n{t['player']} ({t['team']}, {t['league']}, {t['season']}, "
          f"{t['position_group']}, {t['minutes']:.0f} min)")
    print(f"  Closest role:  {t['role_1_name']}")
    print(f"  Second role:   {t['role_2_name']}  (gap {t['role_gap']:.2f}; "
          f"below ~0.3 = hybrid)")
    print("  " + ", ".join(f"{f.replace('_per90', '')} {t[f]:.2f}" for f in FEATURES))

    res = similar_players(
        profiles, Xs, idx, n=args.n, league=args.league, position=args.position,
        role=args.role, min_minutes=args.min_minutes,
        season=args.season if args.mode == "combined" else None,
    )
    print(f"\nMost similar players ({args.mode} profiles):")
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
