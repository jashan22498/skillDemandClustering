# 🎯 Skill Demand Structure Discovery Against Training Provision
**Full-Stack ML Platform for Curricular Alignment & Demand Structure Discovery**  
**Group:** G25 | **Student:** Jashandeep Singh Gill (Registration: 12504818)  
**Course:** INT396 - Unsupervised Learning | **Inspired by:** Smart India Hackathon 2026 (PS ID: SIH26134, Govt of Maharashtra)  
**Environment:** Python 3.15.2 | Streamlit 1.67 | Scikit-Learn 1.9 | Plotly 7.1

---

## 📌 Problem Statement & Research Context
Skill development programmes and National Skills Qualifications Framework (NSQF) courses are historically structured on rigid occupation sectors (e.g. IT-ITeS, BFSI, Healthcare, Automotive, Agriculture). In the modern economy, employers hire across multi-disciplinary skill signatures that dissolve traditional boundaries.

This full-stack machine learning dashboard discovers the genuine latent structure of industry skill demand from 50,000 real-world job postings, benchmarks it against 600 training courses, mathematically quantifies the disagreement against legacy government taxonomies, and surfaces unserved employer demand niches to modernize curriculum planning.

---

## 🚀 Quick Launch Instructions

### Option 1: Double Click (Windows)
Double-click `run_dashboard.bat` in the project folder.

### Option 2: Command Line (PowerShell / Terminal)
```bash
python run_dashboard.py
```
Or directly using Streamlit:
```bash
python -m streamlit run app.py
```
The dashboard will open automatically in your web browser at **`http://localhost:8501`**.

---

## 🔬 Syllabus Topics & Assessment Rubric Alignment (INT396)

| Syllabus Unit | Core Technique | Dashboard Implementation |
| :--- | :--- | :--- |
| **Unit II: Preprocessing & Feature Engineering** | Multi-hot binary vectorization, tokenization | 149 standardized industry skills vectorized across 50,000 postings and 600 NSQF courses. |
| **Unit III: Clustering** | Centroid, Hierarchical, Density, & Probabilistic Clustering | K-Means, MiniBatchKMeans, Agglomerative (Ward linkage), GMM (Gaussian Mixture), DBSCAN. |
| **Unit IV: Dimensionality Reduction** | Latent Structure, Intrinsic Dim, Manifold Discovery | PCA (Scree plot, cumulative variance, loadings, 2D/3D projections), t-SNE 2D manifold, Kaiser criterion, Two-Nearest-Neighbors (2NN) intrinsic estimator, **Random Projection baseline comparison**. |
| **Unit V: Cluster Validity & Evaluation** | Internal & External Evaluation Metrics | Silhouette score & per-cluster silhouette plot, Davies-Bouldin index, Calinski-Harabasz score, Inertia elbow curve, **Adjusted Rand Index (ARI)**, Normalized Mutual Information (NMI), Homogeneity, Completeness, V-Measure, Contingency matrix, Cluster Purity. |
| **Mandatory Baseline** | Official Occupation Baseline & Random Projection | 10 official occupation categories benchmarked against discovered clusters; Gaussian Random Projection distance preservation benchmark. |
| **Advanced Extension** | Unserved Demand Detection & Gap Modernization | Cosine similarity matrix between demand centroids and courses; automated detection of unserved demand groups; skill deficit vs overprovision analysis. |

---

## 📊 Comprehensive Dashboard Architecture (8 Interactive Modules)

1. **📊 Executive KPI Cockpit**:
   - High-level KPIs: Total Job Postings (50,000), Vectorized Skills (149), Discovered Demand Clusters (10), Taxonomy Disagreement Rate (68.1% / ARI = 0.3185), Cluster Cohesion (Silhouette = 0.249), Unserved Demand Clusters Count (2 clusters), Top Deficit Skill.
   - Market prevalence breakdown bar charts and legacy category pie charts.
   - Global interactive filters by Seniority, Company Type, Location, and Posting Year.

2. **🔍 Exploratory Data Analysis & Skill Graphs**:
   - Top 25 Most Demanded Industry Skills bar chart with market percentages.
   - Skill Count per Job Distribution histogram and statistics.
   - Interactive Top 20 Skill Co-occurrence Heatmap.
   - Searchable and filterable raw job postings data explorer.

3. **📉 Dimensionality Reduction & Manifold Explorer (Unit IV)**:
   - PCA Scree Plot (Eigenvalues and Explained Variance Ratio per PC).
   - Cumulative Explained Variance Curve with 80% and 90% threshold lines.
   - Intrinsic Dimensionality Box (Kaiser Criterion: 18 PCs; 2NN Estimator: 2.8; 80% Var: 22 PCs).
   - **Mandatory Syllabus Baseline**: PCA vs Gaussian Random Projection distance correlation comparison (`r = 0.805` vs `r = 0.572`).
   - Component Skill Loadings (PC1 to PC5 top positive and negative driving skills).
   - Interactive 2D PCA, 3D PCA, and 2D t-SNE Non-Linear Manifold scatter plots with customizable color coding.

4. **🧠 Demand Clustering & Model Benchmarking (Unit III & V)**:
   - Live model selector: K-Means, MiniBatchKMeans, GMM, Agglomerative, DBSCAN, Baseline.
   - Optimal K-selection curves: Elbow Inertia, Silhouette Score, Davies-Bouldin, and Calinski-Harabasz.
   - Per-cluster silhouette coefficient box distributions.
   - Ward's minimum-variance hierarchical clustering dendrogram on demand archetypes.
   - Detailed archetype profile table with defining skills, frequent job titles, and market share.

5. **⚡ Taxonomy Disagreement & Validation (Objective 2 & 3)**:
   - Disagreement metrics: ARI (0.3185), NMI (0.612), Homogeneity, Completeness, V-Measure, Cluster Purity.
   - Cross-Tabulation / Contingency Matrix Heatmap (Discovered Clusters vs Official Categories).
   - Category Fragmentation Analysis: Select any legacy sector (e.g. IT, Automotive, BFSI) and see how it shatters into modern cross-functional skill clusters.

6. **🎯 Training Provision Alignment & Unserved Demand (Extension & Objective 1 & 3)**:
   - NSQF Course Catalogue analytics across 600 training courses.
   - Demand Cluster vs Course Cosine Similarity Matrix.
   - Automated Unserved Demand Group Detection with customizable similarity threshold slider.
   - Deep-dive into unserved niches (e.g. Industrial IoT Cyber-Physical Quality Assurance, AgriTech Autonomous Sensing).
   - Skill Market Gap: Market Demand % vs Training Course Provision % (Top Deficit Skills vs Surplus Skills).

7. **🛠️ Interactive Curriculum Modernizer & What-If Simulator**:
   - Qualification Pack Proposal Generator: Select an unserved cluster or input custom job skills.
   - Computes real-time alignment with NSQF courses and flags curriculum gaps.
   - Suggests qualification pack title, NSQF Level, training duration in hours, and new syllabus modules.

8. **📋 Assessment Checklist & Data Export**:
   - Comprehensive INT396 rubric mapping table.
   - One-click CSV download of clustered job postings with discovered archetype labels.
   - One-click CSV download of the Skill Market Gap Audit Report.

---

## 📁 Repository Structure
```
├── app.py                                       # Streamlit full-stack interactive dashboard application
├── ml_engine.py                                 # Core ML & analytics engine (Preprocessing, Dim Reduction, Clustering, Validity, Alignment)
├── precompute.py                                # Precomputes and caches PCA/t-SNE/K-range embeddings for instant 1.5s load
├── precomputed_artifacts.pkl                    # Cached ML embeddings, models, and evaluation tables
├── run_dashboard.py                             # Python launcher script
├── run_dashboard.bat                            # Windows 1-click double-click launcher
├── .streamlit/
│   ├── config.toml                              # Production theme, headless configuration, UI settings
│   └── credentials.toml                         # Local credentials bypass configuration
├── job_postings_SYNTHETIC.csv                   # 50,000 job postings dataset
├── job_skills_SYNTHETIC.csv                     # 344,140 job skill mappings
├── training_courses_SYNTHETIC_NOT_OFFICIAL.csv   # 600 simulated NSQF training courses
├── course_skills_SYNTHETIC.csv                  # 3,345 course skill mappings
├── occupation_taxonomy_SYNTHETIC.csv            # 10 official legacy occupation categories
└── README.md                                    # Comprehensive project documentation
```

---

## 👥 Authorship & Acknowledgements
- **Author:** Jashandeep Singh Gill (Registration No. 12504818)
- **Project Group:** G25
- **Course:** INT396 - Unsupervised Learning
- **Hackathon Reference:** Smart India Hackathon 2026 (PS ID: SIH26134)
