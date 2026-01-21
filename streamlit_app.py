import json
import joblib
import numpy as np
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"   # Put artifacts folder next to this file OR change path

# ---------- Load artifacts (cached) ----------
@st.cache_resource
def load_segmentation_artifacts():
    scaler = joblib.load(ARTIFACTS_DIR / "rfm_scaler.pkl")
    kmeans = joblib.load(ARTIFACTS_DIR / "kmeans_model.pkl")
    with open(ARTIFACTS_DIR / "cluster_labels.json", "r") as f:
        cluster_labels = json.load(f)
    # JSON keys may become strings; normalize
    cluster_labels = {int(k): v for k, v in cluster_labels.items()}
    return scaler, kmeans, cluster_labels

@st.cache_resource
def load_recommender_artifacts():
    products = np.load(ARTIFACTS_DIR / "product_list.npy", allow_pickle=True).tolist()
    sim = np.load(ARTIFACTS_DIR / "similarity_matrix.npy")
    product_to_idx = {p: i for i, p in enumerate(products)}
    return products, sim, product_to_idx

def recommend_products(product_name: str, products, sim_matrix, product_to_idx, top_n=5):
    q = product_name.strip().upper()
    if q not in product_to_idx:
        return None

    idx = product_to_idx[q]
    scores = sim_matrix[idx]
    # top indices excluding itself
    top_idx = np.argsort(scores)[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_n]
    return [products[i] for i in top_idx]

# ---------- UI ----------
st.title("Shopper Spectrum: Segmentation + Recommendations")

# Load everything
try:
    scaler, kmeans, cluster_labels = load_segmentation_artifacts()
    products, sim_matrix, product_to_idx = load_recommender_artifacts()
except Exception as e:
    st.error("Artifacts not found or failed to load. Ensure 'artifacts/' is in the correct location.")
    st.exception(e)
    st.stop()

tab1, tab2 = st.tabs(["Product Recommendations", "Customer Segmentation"])

# ---------- Tab 1: Recommendations ----------
with tab1:
    st.subheader("Item-Based Product Recommendations")

    product_input = st.selectbox(
    "Select a product:",
    options=products,
    index=None,
    placeholder="Search and select a product..."
)


    colA, colB = st.columns([1, 2])
    with colA:
        st.caption("Tip: product names are stored in UPPERCASE.")
    with colB:
        st.caption("If a product is not found, try searching by copying exact name from your dataset output.")

    if st.button("Get Recommendations", type="primary"):
        if product_input is None:
            st.warning("Please enter a product name.")
        else:
            recs = recommend_products(product_input, products, sim_matrix, product_to_idx, top_n=5)
            if recs is None:
                st.error("Product not found. Please ensure the product name matches the dataset (case-insensitive).")
            else:
                st.success("Top 5 similar products:")
                for i, r in enumerate(recs, 1):
                    st.write(f"{i}. {r}")

# ---------- Tab 2: Segmentation ----------
with tab2:
    st.subheader("Customer Segmentation (RFM → Segment)")

    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
    with c2:
        frequency = st.number_input("Frequency (number of invoices)", min_value=1, value=5, step=1)
    with c3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=200.0, step=10.0)

    if st.button("Predict Segment", type="primary"):
        X = np.array([[recency, frequency, monetary]], dtype=float)
        X_scaled = scaler.transform(X)
        cluster_id = int(kmeans.predict(X_scaled)[0])
        segment = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        st.success(f"Predicted Segment: {segment}")
        st.caption(f"Cluster ID: {cluster_id}")
