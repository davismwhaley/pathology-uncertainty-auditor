
# The Pathologist Uncertainty Auditor
### Mapping the Morphological Manifold of Breast Cancer Histopathology

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pathology-uncertainty-auditor-n8hj3hp7m2njcgopi8vbkc.streamlit.app/)]

## 🩺 The Human Cost of Binary Labels
In a hospital waiting room, a patient waits for a diagnosis. The difference between "Benign" (safe) and "Malignant" (cancer) determines everything—their treatment, their finances, their survival.

But biology doesn't always fit into two neat boxes.

Pathology is often treated as a binary classification task, but cellular morphology exists on a continuous spectrum. There are "Transition Zones"—edge cases where a benign cell begins to mutate, adopting textures that look dangerously close to cancer, or vice versa. When AI models are forced to make a binary choice on these ambiguous cases, they often make confident mistakes.

**This project is not about building a classifier to replace doctors. It is about building an Auditor to protect them.** By mapping the "Ambiguous Manifold" of the BreaKHis dataset (7,909 images), we identify exactly where the boundary blurs, creating a safety net for diagnostic uncertainty.

---

## 📊 Key Findings: The 47% Ambiguity
We audited the BreaKHis dataset using unsupervised manifold learning to see if the data naturally separated into "Benign" and "Malignant."

*   **The "Noise" Discovery:** **47.8% of the dataset** was identified as "Morphological Noise" by the HDBSCAN algorithm. This suggests that nearly half of the slides contain textures that do not fit into standard, discrete morphological families.
*   **The "Trap" Validated:** We trained a supervised Logistic Regression classifier (94.4% Accuracy) to test this finding. **56.2% of all classification errors** occurred strictly within the "Noise" region we identified. This proves the ambiguity is objective, not random.
*   **Cluster 53 (The Benign Mimic):** We isolated a specific sub-group of Mucinous Carcinoma (Malignant) that is morphologically indistinguishable from Benign Fibroadenoma ($Entropy \approx 0.998$). A doctor looking at these cells without a molecular test could easily be misled.

**Visualizing the Conflict:**
*Blue cells are Benign, Orange/Red cells are Malignant.*

![Manifold Projection](https://github.com/davismwhaley/pathology-uncertainty-auditor/blob/main/results/Screenshot%202026-01-05%20144940.png)

---

## 🚀 Why This Matters (Implications)
### For the Patient
A "False Positive" leads to unnecessary chemotherapy and trauma. A "False Negative" leads to untreated cancer. By identifying the "Ambiguous Manifold," we can flag these specific cases for **molecular testing** rather than relying on visual inspection alone.

### For "Trustworthy AI"
Most Medical AI gives a diagnosis with 99% confidence, even when it is wrong. This project demonstrates a **Data-Centric AI** approach. Instead of forcing a decision, our system acts as a "Second Opinion," alerting the pathologist: *"Warning: This image falls into Cluster 53. It looks Malignant, but mathematically resembles Benign tissue. Please verify."*

*Below: The "Conflict Gallery." To the human eye, the Benign (top) and Malignant (bottom) samples in this cluster look nearly identical.*

![Conflict Gallery](https://github.com/davismwhaley/pathology-uncertainty-auditor/blob/main/results/Screenshot%202026-01-05%20155943.jpg)

---

## 🛠 Methodology: The Audit Pipeline
We treated the images as data points on a high-dimensional manifold rather than simple pixels.

### Phase 1: Feature Extraction (ResNet50)
We utilized a **ResNet50** architecture pre-trained on ImageNet. We removed the final classification head to use the network purely as a feature extractor.
*   **Input:** 7,909 Histology Images (40X - 400X magnification).
*   **Output:** A 2048-dimensional vector for each image, mathematically representing texture, nuclear density, and stromal patterns.

### Phase 2: Manifold Projection (UMAP)
High-dimensional data is impossible to visualize. We used **UMAP (Uniform Manifold Approximation and Projection)** to project the 2048D vectors into 2D space.
*   *Parameters:* `n_neighbors=15` (to preserve local tissue texture), `min_dist=0.1` (to allow natural clustering).
*   *Result:* A topological map where similar cells are neighbors, regardless of their human labels.

### Phase 3: The Uncertainty Audit (HDBSCAN)
We applied **HDBSCAN**, a density-based clustering algorithm, to identify natural groupings.
*   Unlike K-Means (which forces every point into a group), HDBSCAN allows for **"Noise"** (Cluster -1).
*   We hypothesized that "Noise" points represented the biological transition zones.
*   We calculated **Shannon Entropy** for each cluster to find "Impure" regions where Benign and Malignant samples mixed.

### Phase 4: Validation & Tooling
*   **Triangulation:** We validated the findings by running a supervised stress test. The errors aligned perfectly with our unsupervised "Noise" clusters.
*   **Reverse Image Search:** We built a Content-Based Image Retrieval (CBIR) tool using `NearestNeighbors` to allow pathologists to query the manifold directly.

---

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

## 📂 Repository Structure
*   `notebooks/`: The raw research code (UMAP/HDBSCAN analysis).
*   `app.py`: The interactive Pathology Dashboard (Streamlit).
*   `results/`: Validation reports, Conflict Atlas data, and generated figures.
*   `data/`: Contains the processed feature arrays (Raw images excluded for size).
