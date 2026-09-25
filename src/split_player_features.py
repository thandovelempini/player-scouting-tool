import pandas as pd

INPUT_FILE = "data/processed/player_features.csv"

OUTFIELD_FILE = "data/processed/outfield_features.csv"
GOALKEEPER_FILE = "data/processed/goalkeeper_features.csv"

df = pd.read_csv(INPUT_FILE)

outfield_df = df[
    df["position_group"].isin(["DF", "MF", "FW"])
].copy()

goalkeeper_df = df[
    df["position_group"] == "GK"
].copy()

outfield_df.to_csv(
    OUTFIELD_FILE,
    index=False
)

goalkeeper_df.to_csv(
    GOALKEEPER_FILE,
    index=False
)

print("Outfield dataset:")
print("Records:", len(outfield_df))
print("Players:", outfield_df["player"].nunique())

print("\nOutfield positions:")
print(
    outfield_df["position_group"].value_counts()
)

print("\nGoalkeeper dataset:")
print("Records:", len(goalkeeper_df))
print("Players:", goalkeeper_df["player"].nunique())

print("\nGoalkeeper positions:")
print(
    goalkeeper_df["position_group"].value_counts()
)