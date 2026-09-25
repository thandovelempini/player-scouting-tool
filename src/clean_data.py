import pandas as pd
import os

INPUT_FILE = "data/processed/player_profiles_raw.csv"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:")
print(df.shape)

# Remove duplicate FBREF columns 
metadata_duplicates = [
    "Age_stats_keeper",
    "Age_stats_misc",
    "Age_stats_playing_time",
    "Age_stats_shooting",

    "Born_stats_keeper",
    "Born_stats_misc",
    "Born_stats_playing_time",
    "Born_stats_shooting",

    "Comp_stats_keeper",
    "Comp_stats_misc",
    "Comp_stats_playing_time",
    "Comp_stats_shooting",

    "Nation_stats_keeper",
    "Nation_stats_misc",
    "Nation_stats_playing_time",
    "Nation_stats_shooting",

    "Pos_stats_keeper",
    "Pos_stats_misc",
    "Pos_stats_playing_time",
    "Pos_stats_shooting",

    "Rk_stats_keeper",
    "Rk_stats_misc",
    "Rk_stats_playing_time",
    "Rk_stats_shooting",

    "90s_stats_keeper",
    "90s_stats_misc",
    "90s_stats_playing_time",
    "90s_stats_shooting",

    "MP_stats_keeper",
    "MP_stats_playing_time",

    "Min_stats_keeper",
    "Min_stats_playing_time",

    "Starts_stats_keeper",
    "Starts_stats_playing_time",

    "Gls_stats_shooting",

    "CrdY_stats_misc",
    "CrdR_stats_misc",

    "PK_stats_shooting",
    "PKatt_stats_keeper",
    "PKatt_stats_shooting"
]

df = df.drop(
    columns=metadata_duplicates,
    errors="ignore"
)

# Rename columns 
rename_columns = {
    "Player": "player",
    "Nation": "nation",
    "Pos": "position",
    "Squad": "team",
    "Comp": "league",
    "Age": "age",
    "Born": "birth_year",
    "Season": "season",

    "MP": "matches",
    "Starts": "starts",
    "Min": "minutes",
    "90s": "nineties",

    "Gls": "goals",
    "Ast": "assists",
    "G+A": "goals_assists",
    "G-PK": "non_penalty_goals",
    "PK": "penalty_goals",
    "PKatt": "penalty_attempts",

    "CrdY": "yellow_cards",
    "CrdR": "red_cards",
    "2CrdY": "second_yellow_cards",

    "Sh": "shots",
    "SoT": "shots_on_target",
    "SoT%": "shot_on_target_pct",
    "Sh/90": "shots_per90",
    "SoT/90": "shots_on_target_per90",
    "G/Sh": "goals_per_shot",
    "G/SoT": "goals_per_shot_on_target",

    "Fls": "fouls",
    "Fld": "fouls_drawn",
    "Off": "offsides",
    "Crs": "crosses",
    "Int": "interceptions",
    "TklW": "tackles_won",

    "OG": "own_goals",

    "Min%": "minutes_pct",
    "Mn/MP": "minutes_per_match",
    "Mn/Start": "minutes_per_start",
    "Mn/Sub": "minutes_per_sub",

    "Subs": "substitute_appearances",
    "Compl": "complete_matches",
    "unSub": "unused_substitutions",

    "W": "wins",
    "D": "draws",
    "L": "losses",

    "PPM": "points_per_match",
    "onG": "team_goals_with_player",
    "onGA": "team_goals_against_with_player",
    "+/-": "goal_difference_with_player",
    "+/-90": "goal_difference_per90",
    "On-Off": "on_off_difference",

    "GA": "goals_against",
    "GA90": "goals_against_per90",
    "Saves": "saves",
    "Save%": "save_pct",
    "SoTA": "shots_on_target_against",
    "CS": "clean_sheets",
    "CS%": "clean_sheet_pct",
    "PKA": "penalties_against",
    "PKm": "penalties_missed",
    "PKsv": "penalties_saved"
}

df = df.rename(
    columns=rename_columns
)

# Remove ranking columns 
df = df.drop(
    columns=["Rk"],
    errors="ignore"
)

# Standardise league names
league_names = {
    "eng Premier League": "Premier League",
    "es La Liga": "La Liga",
    "it Serie A": "Serie A",
    "de Bundesliga": "Bundesliga",
    "fr Ligue 1": "Ligue 1"
}

df["league"] = df["league"].replace(league_names)

def assign_position_group(position):
    if pd.isna(position):
        return "Unknown"
    primary = position.split(",")[0].strip()
    return primary if primary in {"GK", "DF", "MF", "FW"} else "Unknown"

df["position_group"] = df["position"].apply(assign_position_group)

# Position groups
print(df["position_group"].value_counts())

# Leagues
print(df["league"].value_counts())

# Seasons
print(df["season"].value_counts())

# Dataset shape
print("\nDataset shape:")
print(df.shape)

# Unique players
print("\nUnique players:")
print(df["player"].nunique())

print("\nPlayer-season records:")
print(len(df))

# Dataset types
print("\nData types:")
print(df.dtypes)

missing = df.isna().sum()
missing_pct = df.isna().mean() * 100

missing_summary = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct.round(2)
})

print("\nMissing values:")
print(
    missing_summary[
        missing_summary["missing_count"] > 0
    ].sort_values(
        "missing_pct",
        ascending=False
    )
)

print("\nLeague counts:")
print(
    df.groupby(["season", "league"])
    .size()
)

print("\nNumeric summary:")
print(df.describe())

output_file = (
    f"{OUTPUT_DIR}/player_profiles_clean.csv"
)

df.to_csv(
    output_file,
    index=False
)