import pandas as pd

INPUT_FILE = "data/processed/player_profiles_clean.csv"
OUTPUT_FILE = "data/processed/player_profiles_analysis.csv"

df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))

df["minutes"] = pd.to_numeric(
    df["minutes"],
    errors="coerce"
)

analysis_df = df[
    df["minutes"] >= 900
].copy()

print("Records after 900-minutes:", len(analysis_df))
print("Players after threshold:", analysis_df["player"].nunique())

print("\nRecords by season:")
print(
    analysis_df["season"].value_counts()
)

print("\nRecords by position:")
print(
    analysis_df["position_group"].value_counts()
)

print("\nRecords by league:")
print(
    analysis_df["league"].value_counts()
)

analysis_df.to_csv(
    OUTPUT_FILE,
    index=False
)