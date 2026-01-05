
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURATION ---
st.set_page_config(page_title="Pathology Auditor", layout="wide")

# Fix for Cloud vs Local Paths
def get_image_path(row):
    if 'cloud_path' in row:
        path = row['cloud_path']
        # If relative path exists (cloud or local), use it
        if Path(path).exists():
            return path
        # If absolute path exists (local only), use it
        if Path(row['path']).exists():
            return row['path']
    # Fallback
    return row['path']

# --- DATA LOADING ---
@st.cache_data
def load_data():
    results_dir = Path("results")
    processed_dir = Path("data/processed")

    # Load CSV
    csv_path = results_dir / "breakhis_manifold_audited.csv"
    if not csv_path.exists():
        st.error("CSV not found.")
        return None, None
    df = pd.read_csv(csv_path)

    # Load Features
    npy_path = processed_dir / "resnet50_features.npy"
    if not npy_path.exists():
        return df, None
    features = np.load(npy_path)

    return df, features

df, features = load_data()

# Helper to safely load image
def safe_load_image(path):
    # Convert to Path object
    p = Path(path)
    if p.exists():
        return Image.open(p)
    else:
        # Return a placeholder if image is missing
        return Image.new('RGB', (224, 224), color='lightgrey')

# Initialize Search
if features is not None:
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
    knn.fit(features)

st.sidebar.title("🔬 Pathology Auditor")
page = st.sidebar.radio("Navigation", ["Manifold Dashboard", "Cluster Inspector (Demo)", "Search Tool (Demo)"])

if page == "Manifold Dashboard":
    st.title("The Morphological Manifold")
    st.info("Note: For this Cloud Demo, only 'Cluster 53' images are fully loaded to save bandwidth.")

    fig = px.scatter(
        df, x='umap_x', y='umap_y', color='cluster_id',
        color_continuous_scale=px.colors.qualitative.G10,
        hover_data=['label', 'subtype', 'filename'],
        title="BreaKHis 400X: UMAP Projection"
    )
    fig.update_layout(height=800, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Cluster Inspector (Demo)":
    st.title("🧐 Cluster Inspector")

    cluster_ids = sorted(df['cluster_id'].unique())
    idx_53 = cluster_ids.index(53) if 53 in cluster_ids else 0
    selected_cluster = st.selectbox("Select Cluster:", cluster_ids, index=idx_53)

    subset = df[df['cluster_id'] == selected_cluster]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", len(subset))
    purity = max(len(subset[subset['label']=='Benign']), len(subset[subset['label']=='Malignant'])) / len(subset)
    col2.metric("Purity", f"{purity:.2%}")
    col3.metric("Subtype", subset['subtype'].mode()[0])

    st.subheader("Gallery")
    cols = st.columns(5)

    shown_count = 0
    for _, row in subset.iterrows():
        img_path = get_image_path(row)
        if Path(img_path).exists():
            with cols[shown_count % 5]:
                st.image(safe_load_image(img_path), caption=row['label'])
            shown_count += 1
            if shown_count >= 10: break

    if shown_count == 0:
        st.warning("⚠️ No images for this cluster were uploaded to the Cloud Demo. Try Cluster 53!")

elif page == "Search Tool (Demo)":
    st.title("🔎 Reverse Image Search")

    if st.button("🎲 Random Malignant (Cluster 53)"):
        demo_candidates = df[(df['cluster_id'] == 53) & (df['label'] == 'Malignant')]
        if not demo_candidates.empty:
            st.session_state['q_idx'] = demo_candidates.sample(1).index[0]

    if 'q_idx' in st.session_state:
        idx = st.session_state['q_idx']
        row = df.loc[idx]

        q_feat = features[df.index.get_loc(idx)].reshape(1, -1)
        _, indices = knn.kneighbors(q_feat)

        path = get_image_path(row)
        st.image(safe_load_image(path), caption="QUERY", width=200)

        st.write("Neighbors:")
        cols = st.columns(5)
        for i, n_idx in enumerate(indices[0][1:]):
            n_row = df.iloc[n_idx]
            n_path = get_image_path(n_row)
            with cols[i]:
                st.image(safe_load_image(n_path), caption=n_row['label'])
