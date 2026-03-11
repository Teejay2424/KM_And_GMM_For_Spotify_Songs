import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.title("Spotify Song Clustering: KMeans vs GMM")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("SpotifyFeatures.csv").iloc[:50000]

features = ['danceability','energy','tempo','loudness','valence']
df = df[features].dropna()

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -----------------------------
# KMeans clustering
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

df["kmeans_cluster"] = kmeans_labels

# -----------------------------
# GMM clustering
# -----------------------------
gmm = GaussianMixture(n_components=5, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

df["gmm_cluster"] = gmm_labels

# -----------------------------
# PCA transformation
# -----------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# =============================
# KMEANS PLOT
# =============================
st.header("KMeans Clustering (PCA Visualization)")

fig1, ax1 = plt.subplots()

for c in sorted(np.unique(kmeans_labels)):
    ax1.scatter(
        pca_result[kmeans_labels==c,0],
        pca_result[kmeans_labels==c,1],
        label=f"Cluster {c}",
        s=5
    )

ax1.set_title("Spotify Clusters using KMeans")
ax1.set_xlabel("PCA 1")
ax1.set_ylabel("PCA 2")
ax1.legend()

st.pyplot(fig1)

# =============================
# GMM PLOT
# =============================
st.header("Gaussian Mixture Model Clustering (PCA Visualization)")

fig2, ax2 = plt.subplots()

for c in sorted(np.unique(gmm_labels)):
    ax2.scatter(
        pca_result[gmm_labels==c,0],
        pca_result[gmm_labels==c,1],
        label=f"Cluster {c}",
        s=5
    )

ax2.set_title("Spotify Clusters using GMM")
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.legend()

st.pyplot(fig2)

# =============================
# MODEL METRICS
# =============================

kmeans_scores = [
    silhouette_score(X_scaled, kmeans_labels),
    davies_bouldin_score(X_scaled, kmeans_labels),
    calinski_harabasz_score(X_scaled, kmeans_labels)
]

gmm_scores = [
    silhouette_score(X_scaled, gmm_labels),
    davies_bouldin_score(X_scaled, gmm_labels),
    calinski_harabasz_score(X_scaled, gmm_labels)
]

# scale CH for visualization
scale_factor = 15000

kmeans_plot = [
    kmeans_scores[0],
    kmeans_scores[1],
    kmeans_scores[2]/scale_factor
]

gmm_plot = [
    gmm_scores[0],
    gmm_scores[1],
    gmm_scores[2]/scale_factor
]

metrics = ['Silhouette','Davies-Bouldin','Calinski-Harabasz']

x = np.arange(len(metrics))
width = 0.35

# =============================
# MODEL COMPARISON PLOT
# =============================
st.header("Clustering Model Comparison")

fig3, ax3 = plt.subplots()

ax3.bar(x - width/2, kmeans_plot, width, label='KMeans')
ax3.bar(x + width/2, gmm_plot, width, label='GMM')

ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.set_ylabel("Scaled Score")
ax3.set_title("Model Evaluation Metrics")
ax3.legend()

st.pyplot(fig3)

# =============================
# CLUSTER STATS
# =============================
st.header("Cluster Feature Means (KMeans)")

st.dataframe(df.groupby("kmeans_cluster")[features].mean())
