import soccerdata as sd
import os

SEASONS = ["2425", "2526"]
LEAGUE = "Big 5 European Leagues Combined"

os.makedirs("data/raw", exist_ok=True)

fbref = sd.FBref(leagues=LEAGUE, seasons=SEASONS)

stat_types = [
    "standard",
    "shooting",
    "misc",
    "playing_time"
]

data = {}

for stat_type in stat_types:
    print(f"{stat_type}")

    df = fbref.read_player_season_stats(
        stat_type=stat_type
    )

    data[stat_type] = df

    print(f"Shape: {df.shape}")
    print(df.index.names)
    print(df.index[:5])
    print(df.columns.tolist())

    df.to_csv(f"data/raw/{stat_type}.csv")

