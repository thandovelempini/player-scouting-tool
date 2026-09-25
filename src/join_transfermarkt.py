"""
Join Transfermarkt market values, contracts, heights and sub-positions onto
the clustered players
"""
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

CLUSTERS = "data/processed/player_clusters.csv"
TM_DIR = "data/raw/transfermarkt"
OUT = "data/processed/player_clusters_enriched.csv"
REVIEW = "data/processed/tm_match_review.csv"
OVERRIDES = "data/raw/transfermarkt/manual_overrides.csv"   # player, birth_year, tm_player_id

SNAPSHOT = pd.Timestamp("2026-07-06")   # Transfermarkt dataset is current to this date
FUZZY_CUTOFF = 0.88
SEASON_DATES = {                        # (season start, season end)
    "2024/25": (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-06-30")),
    "2025/26": (pd.Timestamp("2025-07-01"), pd.Timestamp("2026-06-30")),
}


def norm(text):
    """Lowercase, strip accents and punctuation: 'Guéhi' -> 'guehi'."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_transfermarkt():
    tm = pd.read_csv(f"{TM_DIR}/players.csv", parse_dates=["date_of_birth", "contract_expiration_date"])
    tm = tm[tm["last_season"] >= 2024].copy()          # active in the last two seasons
    tm["birth_year"] = tm["date_of_birth"].dt.year
    tm["nkey"] = tm["name"].map(norm)
    return tm


def club_names(tm, val):
    recent = val[pd.to_datetime(val["date"]) >= "2024-07-01"]
    names = {}
    for pid, club in zip(recent["player_id"], recent["current_club_name"]):
        names.setdefault(pid, set()).add(norm(club))
    for pid, club in zip(tm["player_id"], tm["current_club_name"]):
        names.setdefault(pid, set()).add(norm(club))
    return names


def club_score(team, clubs):
    t = norm(team)
    best = 0.0
    for c in clubs:
        r = SequenceMatcher(None, t, c).ratio()
        if t and (set(t.split()) <= set(c.split()) or set(c.split()) <= set(t.split())):
            r = max(r, 0.9)
        best = max(best, r)
    return best


def match_players(players, tm, clubs):
    by_key = {k: g for k, g in tm.groupby(["nkey", "birth_year"])}
    by_year = {y: g for y, g in tm.groupby("birth_year")}
    rows = []
    for name, by, team in players[["player", "birth_year", "team"]].itertuples(index=False):
        key = norm(name)
        method, pid, score = "unmatched", None, None

        g = by_key.get((key, by))
        if g is not None:
            method = "exact" if len(g) == 1 else "exact_multiple"
            if len(g) == 1:
                pid = g.iloc[0]["player_id"]
            else:  
                scored = sorted(((club_score(team, clubs.get(p, set())), p) for p in g["player_id"]), reverse=True)
                if scored[0][0] >= 0.6 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.1):
                    pid = scored[0][1]
                else:
                    pid = g.sort_values(["last_season", "market_value_in_eur"], ascending=False).iloc[0]["player_id"]
                    method = "exact_multiple_unsure"
        else:
            best = []
            for year in (by, by - 1, by + 1):           
                for _, c in by_year.get(year, tm.iloc[0:0]).iterrows():
                    a, b = set(key.split()), set(c["nkey"].split())
                    s = SequenceMatcher(None, key, c["nkey"]).ratio()
                    if a and (a <= b or b <= a) and min(len(a), len(b)) >= 1 and len(key) >= 4:
                        s = max(s, 0.9)                 
                    if s >= FUZZY_CUTOFF:
                        best.append((s - (0.02 if year != by else 0), c["player_id"]))
                if best:
                    break
            if best:
                best.sort(reverse=True)
                unique = len(best) == 1 or best[0][0] - best[1][0] >= 0.05
                method = "fuzzy" if unique else "fuzzy_ambiguous"
                score, pid = best[0][0], best[0][1]
                if not unique:
                    pid = None
            else:
                last = key.split()[-1] if key else ""
                cands = []
                if len(last) >= 5:
                    for _, c in by_year.get(by, tm.iloc[0:0]).iterrows():
                        s = SequenceMatcher(None, last, c["nkey"].split()[-1]).ratio()
                        if s >= 0.85:
                            cands.append((s, c["player_id"]))
                if len(cands) == 1:
                    method, score, pid = "surname", cands[0][0], cands[0][1]
        rows.append({"player": name, "birth_year": by, "team": team, "method": method,
                     "player_id": pid, "score": score})
    return pd.DataFrame(rows)


def apply_overrides(matches):
    import os
    if not os.path.exists(OVERRIDES):
        return matches
    ov = pd.read_csv(OVERRIDES)
    for _, r in ov.iterrows():
        mask = (matches["player"] == r["player"]) & (matches["birth_year"] == r["birth_year"])
        matches.loc[mask, "player_id"] = r["tm_player_id"] if pd.notna(r["tm_player_id"]) else None
        matches.loc[mask, "method"] = "manual" if pd.notna(r["tm_player_id"]) else "unmatched"
    print(f"Applied {len(ov)} manual overrides from {OVERRIDES}")
    return matches


def attach_values(enriched, val):
    val = val.dropna(subset=["market_value_in_eur"]).copy()
    val["date"] = pd.to_datetime(val["date"]).astype("datetime64[ns]")
    val["player_id"] = val["player_id"].astype("int64")
    val = val.sort_values("date")
    for label, idx in (("value_start", 0), ("value_end", 1)):
        parts = []
        for season, dates in SEASON_DATES.items():
            sub = enriched[(enriched["season"] == season) & enriched["player_id"].notna()].copy()
            sub["player_id"] = sub["player_id"].astype("int64")
            sub["_asof"] = pd.Series(dates[idx], index=sub.index).astype("datetime64[ns]")
            sub["_row"] = sub.index
            m = pd.merge_asof(
                sub.sort_values("_asof"),
                val[["player_id", "date", "market_value_in_eur"]],
                left_on="_asof", right_on="date", by="player_id", direction="backward")
            parts.append(m.set_index("_row")["market_value_in_eur"])
        enriched[label] = pd.concat(parts)
    return enriched


def disambiguate_names(enriched):
    return enriched

def main():
    df = pd.read_csv(CLUSTERS)
    if not {"birth_year"} <= set(df.columns):
        raise SystemExit("player_clusters.csv has no birth_year. Add age and birth_year in create_player_features.py and rerun the pipeline.")
    tm = load_transfermarkt()
    val = pd.read_csv(f"{TM_DIR}/player_valuations.csv")

    uniq = df[["player", "birth_year", "team"]].drop_duplicates()
    matches = apply_overrides(match_players(uniq, tm, club_names(tm, val)))

    static_cols = ["player_id", "sub_position", "height_in_cm", "foot", "contract_expiration_date",
                   "country_of_citizenship", "current_club_name", "current_club_domestic_competition_id",
                   "market_value_in_eur", "highest_market_value_in_eur"]
    static = tm[static_cols].rename(columns={
        "sub_position": "tm_sub_position", "current_club_name": "tm_current_club",
        "current_club_domestic_competition_id": "tm_current_league",
        "market_value_in_eur": "value_latest", "highest_market_value_in_eur": "value_peak"})

    enriched = (df.merge(matches[["player", "birth_year", "team", "method", "player_id"]],
                         on=["player", "birth_year", "team"], how="left")
                  .merge(static, on="player_id", how="left"))
    enriched = attach_values(enriched, val)
    enriched = disambiguate_names(enriched)
    enriched["contract_years_left"] = ((enriched["contract_expiration_date"] - SNAPSHOT).dt.days / 365.25).round(2)

    # Report
    n = len(enriched)
    print("Match method (share of player-season rows):")
    print((enriched["method"].value_counts(normalize=True) * 100).round(1).to_string())
    print(f"\nRows with a value at season end:   {enriched['value_end'].notna().mean():.1%}")
    print(f"Rows with a contract expiry date:  {enriched['contract_expiration_date'].notna().mean():.1%}")
    print(f"Rows with height:                  {enriched['height_in_cm'].notna().mean():.1%}")

    rev = (enriched[enriched["method"].isin(["fuzzy", "surname", "fuzzy_ambiguous", "unmatched", "exact_multiple", "exact_multiple_unsure"])]
           .groupby(["player", "birth_year", "method"], as_index=False)
           .agg(team=("team", "last"), minutes=("minutes", "sum"), tm_id=("player_id", "first")))
    rev.sort_values(["method", "minutes"], ascending=[True, False]).to_csv(REVIEW, index=False)
    print(f"\nWrote {len(rev)} rows to {REVIEW} for a quick manual check (most-played first).")

    enriched.drop(columns=["method"]).assign(tm_match=enriched["method"]).to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({n} rows)")


if __name__ == "__main__":
    main()
