import pandas as pd
from features import FEATURES, CLUSTER_NAMES

df = pd.read_csv("data/processed/player_clusters.csv")
df["role"] = df["cluster"].map(CLUSTER_NAMES)

for c in sorted(df["cluster"].unique()):
    top = (df[df["cluster"] == c]
           .nsmallest(10, "distance_to_centroid")
           [["player", "team", "league", "season", "position_group", "distance_to_centroid"]])
    print(f"\nCluster {c}: {CLUSTER_NAMES[c]}")
    print(top.round(2).to_string(index=False))

print("\nCluster size by league (%):")
print((pd.crosstab(df["role"], df["league"], normalize="columns") * 100).round(1).to_string())

print("\nCluster size by season (%):")
print((pd.crosstab(df["role"], df["season"], normalize="columns") * 100).round(1).to_string())

print("\nCluster profiles (z-scores):")
z = (df[FEATURES] - df[FEATURES].mean()) / df[FEATURES].std()
print(z.groupby(df["role"]).mean().round(2).to_string())

print("\nFeature means by season:")
print(df.groupby("season")[FEATURES].mean().round(2).to_string())

