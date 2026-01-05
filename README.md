
# The Pathologist Uncertainty Auditor
### Mapping the Morphological Manifold of Breast Cancer Histopathology

## 🔬 Project Overview
Medical taxonomies treat cancer as discrete categories (Benign vs. Malignant), but cellular morphology exists on a continuous spectrum. This project audits the **BreaKHis dataset (7,909 images)** to identify the "Ambiguous Manifold"—regions where human labels conflict with mathematical morphology.

## 📊 Key Findings
*   **47.8% of the dataset** was identified as "Morphological Noise" by unsupervised clustering (HDBSCAN), suggesting discrete labels fail to capture biological reality.
*   **Validation:** A supervised classifier (Logistic Regression) achieved 94.4% accuracy, but **56.2% of all errors** occurred strictly within the identified "Noise" region.
*   **Conflict Detection:** Identified "Cluster 53" (Mucinous Carcinoma) as a high-risk zone where Malignant samples visually mimic Benign textures (Entropy: 0.998).

## 🛠 Methodology
1.  **Feature Extraction:** ResNet50 (ImageNet weights) to generate 2048D embeddings.
2.  **Manifold Learning:** UMAP for dimensionality reduction.
3.  **Auditing:** HDBSCAN for density-based clustering and Shannon Entropy for confusion scoring.
4.  **Tooling:** Built a Streamlit Dashboard for "Reverse Image Search" in the manifold.

## 💻 How to Run
1.  **Clone the repo:**
    ```bash
    git clone https://github.com/davismwhaley/pathology-uncertainty-auditor.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```
