import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

st.title("Spotify Song Clustering")

uploaded_file = st.file_uploader("Upload Spotify Dataset CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    features = ['danceability','energy','tempo','loudness','valence']
    df = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model_choice = st.selectbox("Choose Model", ["KMeans", "GMM"])

    if model_choice == "KMeans":
        model = KMeans(n_clusters=5, random_state=42)
        labels = model.fit_predict(X_scaled)

    else:
        model = GaussianMixture(n_components=5, random_state=42)
        labels = model.fit_predict(X_scaled)

    df["cluster"] = labels

    st.write("Cluster Results")
    st.dataframe(df.head())

    st.write("Cluster Means")
    st.dataframe(df.groupby("cluster").mean())
