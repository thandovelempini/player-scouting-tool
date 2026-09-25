import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

data_2024 = pd.read_csv(
    f"{RAW_DIR}/players_data-2024_2025.csv"
)

data_2025 = pd.read_csv(
    f"{RAW_DIR}/players_data-2025_2026.csv"
)

print("\n2024/25:")
print("Shape:", data_2024.shape)

print("\n2025/26:")
print("Shape:", data_2025.shape)

common_columns = sorted(
    set(data_2024.columns) &
    set(data_2025.columns)
)

print("Number of common columns:", len(common_columns))

for column in common_columns:
    print(column)

data_2024 = data_2024[common_columns].copy()
data_2025 = data_2025[common_columns].copy()

data_2024["Season"] = "2024/25"
data_2025["Season"] = "2025/26"

players = pd.concat(
    [data_2024, data_2025],
    ignore_index=True
)

# Combined data
print("\nShape:", players.shape)


print(players["Season"].value_counts())

# Duplicates
duplicate_columns = [
    "Player",
    "Squad",
    "Comp",
    "Season"
]

duplicates = players.duplicated(
    subset=duplicate_columns
)

print(
    "Duplicate player-season records:",
    duplicates.sum()
)

# Missing values
missing = (
    players.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing)

output_path = (
    f"{PROCESSED_DIR}/player_profiles_raw.csv"
)

players.to_csv(
    output_path,
    index=False
)

