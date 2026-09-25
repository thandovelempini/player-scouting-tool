import pandas as pd
import requests

df = pd.read_csv("data/powerbi/player_profiles.csv")

teams = df[["team", "badge_url"]].drop_duplicates()

headers = {
    "User-Agent": "Mozilla/5.0"
}

results = []

for _, row in teams.iterrows():
    try:
        r = requests.get(
            row["badge_url"],
            headers=headers,
            timeout=10
        )

        results.append({
            "team": row["team"],
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type"),
            "works": r.status_code == 200
        })

    except Exception as e:
        results.append({
            "team": row["team"],
            "status": "ERROR",
            "content_type": "",
            "works": False
        })

check = pd.DataFrame(results)

print("\nBADGES THAT DON'T LOAD:\n")
print(
    check[check["works"] == False]
    .to_string(index=False)
)