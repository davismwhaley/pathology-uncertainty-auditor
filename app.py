
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURATION ---
st.set_page_config(page_title="Pathology Auditor", layout="wide")
PROJECT_ROOT = Path.cwd()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- DATA LOADING (Cached for Speed) ---
@st.cache_data
def load_data():
    # Load the Master Audit CSV
    csv_path = RESULTS_DIR / "breakhis_manifold_audited.csv"
    if not csv_path.exists():
        st.error(f"❌ File not found: {{csv_path}}")
        return None, None

    df = pd.read_csv(csv_path)

    # Load the Features (for the Search Engine)
    npy_path = PROCESSED_DIR / "resnet50_features.npy"
    if not npy_path.exists():
        st.error(f"❌ File not found: {{npy_path}}")
        return df, None

    features = np.load(npy_path)
    return df, features

# Load everything
df, features = load_data()

# Initialize Search Engine
if features is not None:
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(features)

# --- SIDEBAR ---
st.sidebar.title("🔬 Pathology Auditor")
st.sidebar.markdown("Mapping the **Ambiguous Manifold** of Breast Cancer.")

page = st.sidebar.radio("Navigation", ["Manifold Dashboard", "Cluster Inspector", "Image Search Tool"])

# --- PAGE 1: THE MANIFOLD ---
if page == "Manifold Dashboard":
    st.title("The Morphological Manifold")
    st.markdown("""
    This map represents **7,909 histology images** projected into 2D space.
    *   **Dots:** Individual cancer slides.
    *   **Colors:** Morphological Families (Clusters).
    *   **Grey Dots:** The "Ambiguous Noise" (Cluster -1) — these images defy standard categorization.
    """)

    # Plotly Chart
    fig = px.scatter(
        df, x='umap_x', y='umap_y', 
        color='cluster_id', 
        color_continuous_scale=px.colors.qualitative.G10,
        hover_data=['label', 'subtype', 'filename'],
        title="BreaKHis 400X: UMAP Projection"
    )
    fig.update_layout(height=800, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: CLUSTER INSPECTOR ---
elif page == "Cluster Inspector":
    st.title("🧐 Cluster Inspector")

    # Dropdown to pick a cluster
    cluster_ids = sorted(df['cluster_id'].unique())
    # Default to 53 if it exists, else 0
    default_idx = cluster_ids.index(53) if 53 in cluster_ids else 0
    selected_cluster = st.selectbox("Select a Cluster ID:", cluster_ids, index=default_idx)

    # Filter Data
    subset = df[df['cluster_id'] == selected_cluster]

    # Stats
    n_total = len(subset)
    n_benign = len(subset[subset['label'] == 'Benign'])
    n_malignant = len(subset[subset['label'] == 'Malignant'])
    purity = max(n_benign, n_malignant) / n_total

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", n_total)
    col2.metric("Purity Score", f"{purity:.2%}")
    col3.metric("Dominant Subtype", subset['subtype'].mode()[0])

    if purity < 0.9:
        st.warning(f"⚠️ **High Conflict Zone:** This cluster is {n_benign} Benign and {n_malignant} Malignant.")
    else:
        st.success("✅ Morphologically Pure Cluster.")

    # Gallery
    st.subheader("Cluster Gallery")

    # Sample images (Max 10)
    sample_imgs = subset.sample(min(10, len(subset)))

    cols = st.columns(5)
    for i, (idx, row) in enumerate(sample_imgs.iterrows()):
        img_path = row['path']
        try:
            image = Image.open(img_path)
            with cols[i % 5]:
                st.image(image, caption=f"{row['label']}\n{row['subtype']}", use_column_width=True)
        except:
            st.write("Image load error")

# --- PAGE 3: REVERSE IMAGE SEARCH ---
elif page == "Image Search Tool":
    st.title("🔎 Reverse Image Search")
    st.markdown("Select a 'Query Image' to find its nearest morphological neighbors.")

    # Random Query Button
    if st.button("🎲 Pick Random Malignant Case"):
        query_row = df[df['label'] == 'Malignant'].sample(1).iloc[0]
        st.session_state['query_idx'] = query_row.name

    if 'query_idx' in st.session_state:
        idx = st.session_state['query_idx']
        row = df.loc[idx]

        # Run Search
        query_feat = features[df.index.get_loc(idx)].reshape(1, -1)
        distances, indices = knn.kneighbors(query_feat)

        # Display Query
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(row['path'], caption=f"QUERY: {row['label']} ({row['cluster_id']})", width=300)

        with c2:
            st.write("### Nearest Neighbors")
            cols = st.columns(5)
            # Skip first (it's the query itself)
            neighbor_indices = indices[0][1:] 
            neighbor_dists = distances[0][1:]

            for i, n_idx in enumerate(neighbor_indices):
                # Map back to df index
                n_row = df.iloc[n_idx]

                # Color code
                color = "🟢" if n_row['label'] == 'Benign' else "🔴"
                if n_row['label'] != row['label']:
                    color = "⚠️" # Conflict!

                with cols[i]:
                    st.image(n_row['path'], caption=f"{color} {n_row['label']}\nDist: {neighbor_dists[i]:.2f}")
