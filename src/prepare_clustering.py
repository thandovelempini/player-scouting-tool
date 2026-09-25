import numpy as np
import pandas as pd
import os

import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from features import FEATURES, CLUSTER_NAMES

INPUT_FILE = "data/processed/outfield_features.csv"
OUTPUT_FILE = "data/processed/player_clusters.csv"
K = 6  

df = pd.read_csv(INPUT_FILE)
X_scaled = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(df[FEATURES])
)

print("Records:", len(df))
for k in range(3, 10):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    print(f"k={k}: silhouette={silhouette_score(X_scaled, labels):.3f}")

kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(imputer.fit_transform(df[FEATURES]))

labels = kmeans.fit_predict(X_scaled)

os.makedirs("models", exist_ok=True)
joblib.dump({"imputer": imputer, "scaler": scaler, "kmeans": kmeans}, "models/clustering.joblib")

df["cluster"] = labels
df["distance_to_centroid"] = kmeans.transform(X_scaled)[np.arange(len(df)), labels]

print("\nCluster profiles:")
print(df.groupby("cluster")[FEATURES].mean().round(2).to_string())
print("\nPosition mix (%):")
print((pd.crosstab(df["cluster"], df["position_group"], normalize="index") * 100).round(0).to_string())
season_mix = pd.crosstab(df["cluster"], df["season"], normalize="index") * 100
print("\nSeason mix (%):")
print(season_mix.round(0).to_string())

skewed = season_mix[(season_mix.max(axis=1) > 65)]
if len(skewed):
    print("\nWARNING: season-skewed clusters:", skewed.index.tolist())

df["role"] = df["cluster"].map(CLUSTER_NAMES)
df.to_csv(OUTPUT_FILE, index=False)