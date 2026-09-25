import pandas as pd

INPUT_FILE = "data/processed/player_profiles_analysis.csv"
OUTPUT_FILE = "data/processed/player_features.csv"

df = pd.read_csv(INPUT_FILE)

df["minutes"] = pd.to_numeric(
    df["minutes"],
    errors="coerce"
)

count_features = [
    "goals",
    "assists",
    "non_penalty_goals",
    "penalty_goals",
    "shots",
    "shots_on_target",
    "crosses",
    "interceptions",
    "tackles_won",
    "fouls",
    "fouls_drawn",
    "offsides",
    "yellow_cards",
    "red_cards",
    "second_yellow_cards"
]

for column in count_features:
    if column in df.columns:
        df[f"{column}_per90"] = (
            df[column] / df["minutes"] * 90
        )

rate_features = [
    "goals_per_shot",
    "goals_per_shot_on_target",
    "shot_on_target_pct",
    "goal_difference_per90",
    "on_off_difference",
    "points_per_match"
]

feature_columns = [
    "player",
    "team",
    "league",
    "season",
    "position",
    "position_group",
    "minutes",
    "age",
    "birth_year"
]

feature_columns += [
    f"{column}_per90"
    for column in count_features
    if column in df.columns
]

feature_columns += [
    column
    for column in rate_features
    if column in df.columns
]

features_df = df[feature_columns].copy()

numeric_columns = features_df.select_dtypes(
    include="number"
).columns

features_df[numeric_columns] = features_df[
    numeric_columns
].apply(
    pd.to_numeric,
    errors="coerce"
)

print("Feature dataset shape:", features_df.shape)

print("\nFeature of columns:")
print(features_df.columns.tolist())

print("\nMissing values:")
print(
    features_df.isna().sum()[
        features_df.isna().sum() > 0
    ]
)

print("\nPosition groups:")
print(
    features_df["position_group"].value_counts()
)

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)