"""
Market-value model: which players are priced below (or above) what their
output, age, league, position and club strength would suggest?
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from features import FEATURES

INPUT = "data/processed/player_clusters_enriched.csv"
OUTPUT = "data/processed/player_value_gaps.csv"

# points_per_match = team's points per game while the player is on the pitch
NUMERIC = FEATURES + ["minutes", "height_in_cm", "points_per_match"]
CATEGORICAL = ["league", "sub_pos"]


def make_model():
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        # value rises then collapses with age; a spline lets the curve bend freely
        ("age", SplineTransformer(n_knots=7, degree=3, knots="quantile"), ["age"]),
    ])
    return Pipeline([("prep", pre), ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 20)))])


def main():
    df = pd.read_csv(INPUT).dropna(subset=["value_end"]).reset_index(drop=True)
    df["sub_pos"] = df["tm_sub_position"].fillna(df["position_group"])
    df["log_value"] = np.log(df["value_end"])
    X, y = df[NUMERIC + CATEGORICAL + ["age"]], df["log_value"]

    # Out-of-fold predictions, grouped by player
    oof = np.zeros(len(df))
    for train, test in GroupKFold(n_splits=5).split(X, y, groups=df["player"]):
        oof[test] = make_model().fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])

    mae = mean_absolute_error(y, oof)
    print(f"Rows: {len(df)}")
    print(f"Out-of-fold R2 (log value): {r2_score(y, oof):.3f}")
    print(f"Typical error: x{np.exp(mae):.2f} (MAE {mae:.2f} in log terms)")

    df["predicted_value"] = np.exp(oof)
    df["value_ratio"] = df["value_end"] / df["predicted_value"]     
    df["log_gap"] = y - oof
    df["gap_z"] = (df["log_gap"] / df["log_gap"].std()).round(2)

    # Are gaps persistent? Same player, two seasons
    w = df.pivot_table(index="player", columns="season", values="log_gap").dropna()
    print(f"Gap correlation across seasons ({len(w)} players): "
          f"{w['2024/25'].corr(w['2025/26']):.2f}")

    # What drives value (standardised coefficients, numeric features)
    final = make_model().fit(X, y)
    coefs = pd.Series(final.named_steps["ridge"].coef_[:len(NUMERIC)], index=NUMERIC)
    print("\nStandardised drivers of log value (age is modelled as a curve, so not listed):")
    print(coefs.sort_values(ascending=False).round(2).to_string())

    out = df[["player", "team", "league", "season", "age", "sub_pos", "minutes",
              "role", "value_end", "predicted_value", "value_ratio", "log_gap", "gap_z"]].copy()
    out["value_end_m"] = (out["value_end"] / 1e6).round(2)
    out["predicted_value_m"] = (out["predicted_value"] / 1e6).round(2)
    out["value_ratio"] = out["value_ratio"].round(2)
    out.drop(columns=["value_end", "predicted_value"]).to_csv(OUTPUT, index=False)

    cols = ["player", "team", "league", "age", "sub_pos", "value_end_m",
            "predicted_value_m", "value_ratio"]
    latest = out[(out["season"] == "2025/26") & (out["minutes"] >= 1500)]
    print("\nMost underpriced, 2025/26, age <= 28, min 1500 min (recruitable targets):")
    print(latest[latest["age"] <= 28].nsmallest(12, "log_gap")[cols].to_string(index=False))
    print("\nMost overpriced, 2025/26, min 1500 min (young players are priced on potential):")
    print(latest.nlargest(8, "log_gap")[cols].to_string(index=False))

    both = out.groupby("player").filter(lambda g: len(g) == 2 and (g["minutes"] >= 900).all())
    avg = (both.groupby("player").agg(team=("team", "last"), age=("age", "max"),
                                      sub_pos=("sub_pos", "last"), avg_gap=("log_gap", "mean"),
                                      value_m=("value_end_m", "last"))
                .assign(ratio=lambda d: np.exp(d["avg_gap"]).round(2)))
    print("\nPriced below the model in BOTH seasons (age <= 28, more reliable):")
    print(avg[avg["age"] <= 28].nsmallest(10, "avg_gap")
            [["team", "age", "sub_pos", "value_m", "ratio"]].to_string())
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
