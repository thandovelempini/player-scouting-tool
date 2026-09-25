import os

import joblib
import numpy as np
import pandas as pd

from features import FEATURES
from shortlist import ENRICHED_FILE, attach_tm
from similarity import MODEL_FILE, add_roles, build_profiles

OUT_DIR = "data/powerbi"
TOP_N = 30
PCT_COLS = [f.replace("_per90", "") + "_pct" for f in FEATURES]
TM_OUT = ["tm_sub_position", "height_in_cm", "foot", "value_m", "value_ratio",
          "contract_expiration_date", "contract_years", "tm_current_club"]

BADGE_FILES = {

    # =========================
    # PREMIER LEAGUE — 2025/26
    # =========================
    "Arsenal": ("history/2025-26/England - Premier League", "Arsenal FC.png"),
    "Aston Villa": ("history/2025-26/England - Premier League", "Aston Villa.png"),
    "Bournemouth": ("history/2025-26/England - Premier League", "AFC Bournemouth.png"),
    "Brentford": ("history/2025-26/England - Premier League", "Brentford FC.png"),
    "Brighton": ("history/2025-26/England - Premier League", "Brighton & Hove Albion.png"),
    "Chelsea": ("history/2025-26/England - Premier League", "Chelsea FC.png"),
    "Crystal Palace": ("history/2025-26/England - Premier League", "Crystal Palace.png"),
    "Everton": ("history/2025-26/England - Premier League", "Everton FC.png"),
    "Fulham": ("history/2025-26/England - Premier League", "Fulham FC.png"),
    "Leeds United": ("history/2025-26/England - Premier League", "Leeds United.png"),
    "Liverpool": ("history/2025-26/England - Premier League", "Liverpool FC.png"),
    "Manchester City": ("history/2025-26/England - Premier League", "Manchester City.png"),
    "Manchester Utd": ("history/2025-26/England - Premier League", "Manchester United.png"),
    "Newcastle United": ("history/2025-26/England - Premier League", "Newcastle United.png"),
    "Newcastle Utd": ("history/2025-26/England - Premier League", "Newcastle United.png"),
    "Nott'ham Forest": ("history/2025-26/England - Premier League", "Nottingham Forest.png"),
    "Nottingham Forest": ("history/2025-26/England - Premier League", "Nottingham Forest.png"),
    "Sunderland": ("history/2025-26/England - Premier League", "Sunderland AFC.png"),
    "Tottenham": ("history/2025-26/England - Premier League", "Tottenham Hotspur.png"),
    "Tottenham Hotspur": ("history/2025-26/England - Premier League", "Tottenham Hotspur.png"),
    "West Ham": ("history/2025-26/England - Premier League", "West Ham United.png"),
    "West Ham United": ("history/2025-26/England - Premier League", "West Ham United.png"),
    "Wolves": ("history/2025-26/England - Premier League", "Wolverhampton Wanderers.png"),

    # 2024/25 PL clubs
    "Ipswich Town": ("history/2024-25/England - Premier League", "Ipswich Town.png"),
    "Leicester City": ("history/2024-25/England - Premier League", "Leicester City.png"),
    "Southampton": ("history/2024-25/England - Premier League", "Southampton FC.png"),
    "Burnley": ("history/2025-26/England - Premier League", "Burnley FC.png"),


    # =========================
    # LA LIGA — 2025/26
    # =========================
    "Alavés": ("history/2025-26/Spain - LaLiga", "Deportivo Alavés.png"),
    "Athletic Club": ("history/2025-26/Spain - LaLiga", "Athletic Bilbao.png"),
    "Atlético Madrid": ("history/2025-26/Spain - LaLiga", "Atlético de Madrid.png"),
    "Barcelona": ("history/2025-26/Spain - LaLiga", "FC Barcelona.png"),
    "Betis": ("history/2025-26/Spain - LaLiga", "Real Betis Balompié.png"),
    "Real Betis": ("history/2025-26/Spain - LaLiga", "Real Betis Balompié.png"),
    "Celta Vigo": ("history/2025-26/Spain - LaLiga", "Celta de Vigo.png"),
    "Elche": ("history/2025-26/Spain - LaLiga", "Elche CF.png"),
    "Espanyol": ("history/2025-26/Spain - LaLiga", "RCD Espanyol Barcelona.png"),
    "Getafe": ("history/2025-26/Spain - LaLiga", "Getafe CF.png"),
    "Girona": ("history/2025-26/Spain - LaLiga", "Girona FC.png"),
    "Levante": ("history/2025-26/Spain - LaLiga", "Levante UD.png"),
    "Leganés": ("history/2024-25/Spain - LaLiga", "CD Leganés.png"),
    "Mallorca": ("history/2025-26/Spain - LaLiga", "RCD Mallorca.png"),
    "Osasuna": ("history/2025-26/Spain - LaLiga", "CA Osasuna.png"),
    "Oviedo": ("history/2025-26/Spain - LaLiga", "Real Oviedo.png"),
    "Rayo Vallecano": ("history/2025-26/Spain - LaLiga", "Rayo Vallecano.png"),
    "Real Madrid": ("history/2025-26/Spain - LaLiga", "Real Madrid.png"),
    "Real Sociedad": ("history/2025-26/Spain - LaLiga", "Real Sociedad.png"),
    "Sevilla": ("history/2025-26/Spain - LaLiga", "Sevilla FC.png"),
    "Valencia": ("history/2025-26/Spain - LaLiga", "Valencia CF.png"),
    "Valladolid": ("history/2024-25/Spain - LaLiga", "Real Valladolid CF.png"),
    "Villarreal": ("history/2025-26/Spain - LaLiga", "Villarreal CF.png"),
    "Las Palmas": ("history/2024-25/Spain - LaLiga", "UD Las Palmas.png"),


    # =========================
    # SERIE A — 2025/26
    # =========================
    "Atalanta": ("history/2025-26/Italy - Serie A", "Atalanta BC.png"),
    "Bologna": ("history/2025-26/Italy - Serie A", "Bologna FC 1909.png"),
    "Cagliari": ("history/2025-26/Italy - Serie A", "Cagliari Calcio.png"),
    "Como": ("history/2025-26/Italy - Serie A", "Como 1907.png"),
    "Cremonese": ("history/2025-26/Italy - Serie A", "US Cremonese.png"),
    "Fiorentina": ("history/2025-26/Italy - Serie A", "ACF Fiorentina.png"),
    "Genoa": ("history/2025-26/Italy - Serie A", "Genoa CFC.png"),
    "Hellas Verona": ("history/2025-26/Italy - Serie A", "Hellas Verona.png"),
    "Inter": ("history/2025-26/Italy - Serie A", "Inter Milan.png"),
    "Juventus": ("history/2025-26/Italy - Serie A", "Juventus FC.png"),
    "Lazio": ("history/2025-26/Italy - Serie A", "SS Lazio.png"),
    "Lecce": ("history/2025-26/Italy - Serie A", "US Lecce.png"),
    "Milan": ("history/2025-26/Italy - Serie A", "AC Milan.png"),
    "Monza": ("history/2024-25/Italy - Serie A", "AC Monza.png"),
    "Napoli": ("history/2025-26/Italy - Serie A", "SSC Napoli.png"),
    "Parma": ("history/2025-26/Italy - Serie A", "Parma Calcio 1913.png"),
    "Pisa": ("history/2025-26/Italy - Serie A", "Pisa Sporting Club.png"),
    "Roma": ("history/2025-26/Italy - Serie A", "AS Roma.png"),
    "Sassuolo": ("history/2025-26/Italy - Serie A", "US Sassuolo.png"),
    "Torino": ("history/2025-26/Italy - Serie A", "Torino FC.png"),
    "Udinese": ("history/2025-26/Italy - Serie A", "Udinese Calcio.png"),
    "Venezia": ("history/2024-25/Italy - Serie A", "Venezia FC.png"),
    "Empoli": ("history/2024-25/Italy - Serie A", "FC Empoli.png"),
    # =========================
    # BUNDESLIGA — 2025/26
    # =========================
    "Augsburg": ("history/2025-26/Germany - Bundesliga", "FC Augsburg.png"),
    "Bayern Munich": ("history/2025-26/Germany - Bundesliga", "Bayern Munich.png"),
    "Bochum": ("history/2024-25/Germany - Bundesliga", "VfL Bochum.png"),
    "Dortmund": ("history/2025-26/Germany - Bundesliga", "Borussia Dortmund.png"),
    "Eint Frankfurt": ("history/2025-26/Germany - Bundesliga", "Eintracht Frankfurt.png"),
    "Eintracht Frankfurt": ("history/2025-26/Germany - Bundesliga", "Eintracht Frankfurt.png"),
    "Freiburg": ("history/2025-26/Germany - Bundesliga", "SC Freiburg.png"),
    "Gladbach": ("history/2025-26/Germany - Bundesliga", "Borussia Mönchengladbach.png"),
    "Hamburger SV": ("history/2025-26/Germany - Bundesliga", "Hamburger SV.png"),
    "Heidenheim": ("history/2025-26/Germany - Bundesliga", "1.FC Heidenheim 1846.png"),
    "Hoffenheim": ("history/2025-26/Germany - Bundesliga", "TSG 1899 Hoffenheim.png"),
    "Holstein Kiel": ("history/2024-25/Germany - Bundesliga", "Holstein Kiel.png"),
    "Köln": ("history/2024-25/Germany - Bundesliga", "1.FC Köln.png"),
    "Leverkusen": ("history/2025-26/Germany - Bundesliga", "Bayer 04 Leverkusen.png"),
    "Mainz 05": ("history/2025-26/Germany - Bundesliga", "1.FSV Mainz 05.png"),
    "RB Leipzig": ("history/2025-26/Germany - Bundesliga", "RB Leipzig.png"),
    "St Pauli": ("history/2025-26/Germany - Bundesliga", "FC St. Pauli.png"),
    "St. Pauli": ("history/2025-26/Germany - Bundesliga", "FC St. Pauli.png"),
    "Stuttgart": ("history/2025-26/Germany - Bundesliga", "VfB Stuttgart.png"),
    "Union Berlin": ("history/2025-26/Germany - Bundesliga", "1.FC Union Berlin.png"),
    "Werder Bremen": ("history/2025-26/Germany - Bundesliga", "SV Werder Bremen.png"),
    "Wolfsburg": ("history/2025-26/Germany - Bundesliga", "VfL Wolfsburg.png"),


    # =========================
    # LIGUE 1 — 2025/26
    # =========================
    "Angers": ("history/2025-26/France - Ligue 1", "Angers SCO.png"),
    "Auxerre": ("history/2025-26/France - Ligue 1", "AJ Auxerre.png"),
    "Brest": ("history/2025-26/France - Ligue 1", "Stade Brestois 29.png"),
    "Le Havre": ("history/2025-26/France - Ligue 1", "Le Havre AC.png"),
    "Lens": ("history/2025-26/France - Ligue 1", "RC Lens.png"),
    "Lille": ("history/2025-26/France - Ligue 1", "LOSC Lille.png"),
    "Lorient": ("history/2025-26/France - Ligue 1", "FC Lorient.png"),
    "Lyon": ("history/2025-26/France - Ligue 1", "Olympique Lyon.png"),
    "Marseille": ("history/2025-26/France - Ligue 1", "Olympique Marseille.png"),
    "Metz": ("history/2025-26/France - Ligue 1", "FC Metz.png"),
    "Monaco": ("history/2025-26/France - Ligue 1", "AS Monaco.png"),
    "Montpellier": ("history/2024-25/France - Ligue 1", "Montpellier HSC.png"),
    "Nantes": ("history/2025-26/France - Ligue 1", "FC Nantes.png"),
    "Nice": ("history/2025-26/France - Ligue 1", "OGC Nice.png"),
    "Paris FC": ("history/2025-26/France - Ligue 1", "Paris FC.png"),
    "Paris Saint-Germain": ("history/2025-26/France - Ligue 1", "Paris Saint-Germain.png"),
    "Reims": ("history/2024-25/France - Ligue 1", "Stade Reims.png"),
    "Rennes": ("history/2025-26/France - Ligue 1", "Stade Rennais FC.png"),
    "Saint-Étienne": ("history/2024-25/France - Ligue 1", "AS Saint-Étienne.png"),
    "Strasbourg": ("history/2025-26/France - Ligue 1", "RC Strasbourg Alsace.png"),
    "Toulouse": ("history/2025-26/France - Ligue 1", "FC Toulouse.png"),
}

BADGE_BASE = "https://raw.githubusercontent.com/luukhopman/football-logos/master"


def build_badge_url(team):
    if team == "Köln":
        return "https://commons.wikimedia.org/wiki/Special:Redirect/file/1.FC%20K%C3%B6ln%20escudo.png"

    if team not in BADGE_FILES:
        return None

    league_folder, filename = BADGE_FILES[team]

    if league_folder is None or filename is None:
        return None

    return (
        f"{BADGE_BASE}/"
        f"{league_folder.replace(' ', '%20')}/"
        f"{filename.replace(' ', '%20')}"
    )

def add_percentiles(profiles):
    out = profiles.copy()
    for f in FEATURES:
        out[f.replace("_per90", "") + "_pct"] = (out[f].rank(pct=True) * 100).round(1)
    return out


def top_similar(profiles, Xs, n=TOP_N):
    diff = Xs[:, None, :] - Xs[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)

    rows = []
    for i in range(len(profiles)):
        for rank, j in enumerate(np.argsort(dist[i])[:n], start=1):
            rows.append({
                "player": profiles.at[i, "player"],
                "player_badge_url": profiles.at[i, "badge_url"],
                "rank": rank,
                "similar_player": profiles.at[j, "player"],
                "similar_team": profiles.at[j, "team"],
                "similar_league": profiles.at[j, "league"],
                "similar_badge_url": profiles.at[j, "badge_url"],
                "similar_age": profiles.at[j, "age"],
                "similar_sub_position": profiles.at[j, "tm_sub_position"],
                "similar_value_m": profiles.at[j, "value_m"],
                "similar_value_ratio": profiles.at[j, "value_ratio"],
                "similar_contract_years": profiles.at[j, "contract_years"],
                "similar_role": profiles.at[j, "role_1_name"],
                
                "similar_goals_per90": profiles.at[j, "goals_per90"],
                "similar_assists_per90": profiles.at[j, "assists_per90"],
                "similar_shots_per90": profiles.at[j, "shots_per90"],
                "similar_crosses_per90": profiles.at[j, "crosses_per90"],
                "similar_tackles_won_per90": profiles.at[j, "tackles_won_per90"],
                "similar_interceptions_per90": profiles.at[j, "interceptions_per90"],

                "distance": round(dist[i, j], 3),
            })
    return pd.DataFrame(rows)


def prepare(df, mode, models):
    profiles = attach_tm(build_profiles(df, mode), df, mode)
    profiles, Xs = add_roles(profiles, models)
    profiles = add_percentiles(profiles)

    profiles["badge_url"] = profiles["team"].apply(build_badge_url)

    profiles["contract_expiration_date"] = (
        profiles["contract_expiration_date"].dt.strftime("%Y-%m-%d")
    )

    return profiles, Xs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(ENRICHED_FILE)
    models = joblib.load(MODEL_FILE)

    keep = (["player", "team", "badge_url", "league", "age", "season", "n_seasons", "position_group",
             "minutes", "role_1_name", "role_2_name", "role_gap"]
            + TM_OUT + FEATURES + PCT_COLS)

    combined, Xs = prepare(df, "combined", models)
    assert combined["player"].is_unique, "duplicate player names in combined profiles"
    combined[keep].round(3).to_csv(f"{OUT_DIR}/player_profiles.csv", index=False)

    seasonal, _ = prepare(df, "season", models)
    seasonal_keep = [
    "player",
    "team",
    "badge_url",
    "league",
    "age",
    "season",
    "n_seasons",
    "position_group",
    "minutes",
    "role_1_name",
    "role_2_name",
    "role_gap",
    "tm_sub_position",
    "height_in_cm",
    "foot",
    "value_m",
    "value_ratio",
    "contract_expiration_date",
    "contract_years",] + FEATURES + PCT_COLS

    seasonal[seasonal_keep].round(3).to_csv(
    f"{OUT_DIR}/player_season_profiles.csv", index=False)
    
    sim = top_similar(combined, Xs)
    sim.to_csv(f"{OUT_DIR}/similar_players.csv", index=False)

    print(f"player_profiles.csv:        {len(combined)} rows")
    print(f"player_season_profiles.csv: {len(seasonal)} rows")
    print(f"similar_players.csv:        {len(sim)} rows")


if __name__ == "__main__":
    main()


