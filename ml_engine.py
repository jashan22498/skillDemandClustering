"""
Skill Demand Structure Discovery Against Training Provision
Core Machine Learning and Analytics Engine
Author: Jashandeep Singh Gill (12504818) - Group G25
Compatible with Python 3.15.2
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist


class DataLoader:
    """Loads and standardizes job postings, training courses, and taxonomies."""

    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.jobs_file = os.path.join(data_dir, "job_postings_SYNTHETIC.csv")
        self.courses_file = os.path.join(data_dir, "training_courses_SYNTHETIC_NOT_OFFICIAL.csv")
        self.job_skills_file = os.path.join(data_dir, "job_skills_SYNTHETIC.csv")
        self.course_skills_file = os.path.join(data_dir, "course_skills_SYNTHETIC.csv")
        self.taxonomy_file = os.path.join(data_dir, "occupation_taxonomy_SYNTHETIC.csv")

        self.df_jobs: Optional[pd.DataFrame] = None
        self.df_courses: Optional[pd.DataFrame] = None
        self.df_taxonomy: Optional[pd.DataFrame] = None
        self.df_job_skills: Optional[pd.DataFrame] = None
        self.df_course_skills: Optional[pd.DataFrame] = None

        self.vectorizer: Optional[CountVectorizer] = None
        self.X_jobs: Optional[np.ndarray] = None
        self.X_courses: Optional[np.ndarray] = None
        self.feature_skills: List[str] = []

    def load_data(self) -> "DataLoader":
        """Loads all raw datasets into memory."""
        self.df_jobs = pd.read_csv(self.jobs_file)
        self.df_courses = pd.read_csv(self.courses_file)
        self.df_taxonomy = pd.read_csv(self.taxonomy_file)
        self.df_job_skills = pd.read_csv(self.job_skills_file)
        self.df_course_skills = pd.read_csv(self.course_skills_file)

        # Standardize skill representation
        self.vectorizer = CountVectorizer(
            tokenizer=lambda s: [item.strip() for item in s.split(";")],
            binary=True,
            token_pattern=None,
            lowercase=False
        )
        # Fit vectorizer on all job skills
        self.X_jobs = self.vectorizer.fit_transform(self.df_jobs["required_skills"]).toarray()
        self.feature_skills = list(self.vectorizer.get_feature_names_out())

        # Vectorize course taught skills in identical feature space
        self.X_courses = self.vectorizer.transform(self.df_courses["taught_skills"]).toarray()

        return self


class DimensionalityReductionEngine:
    """Performs PCA, t-SNE, Random Projection baseline, and Intrinsic Dimensionality estimation."""

    def __init__(self, X: np.ndarray, feature_names: List[str]):
        self.X = X
        self.feature_names = np.array(feature_names)
        self.pca_model: Optional[PCA] = None
        self.pca_coords_2d: Optional[np.ndarray] = None
        self.pca_coords_3d: Optional[np.ndarray] = None
        self.explained_var_ratio: Optional[np.ndarray] = None
        self.cumulative_var_ratio: Optional[np.ndarray] = None

    def fit_pca(self, max_components: int = 50) -> Dict[str, Any]:
        """Fits PCA up to max_components and computes scree statistics and loadings."""
        n_comp = min(max_components, self.X.shape[1], self.X.shape[0])
        self.pca_model = PCA(n_components=n_comp, random_state=42)
        X_pca = self.pca_model.fit_transform(self.X)

        self.pca_coords_2d = X_pca[:, :2]
        self.pca_coords_3d = X_pca[:, :3]
        self.explained_var_ratio = self.pca_model.explained_variance_ratio_
        self.cumulative_var_ratio = np.cumsum(self.explained_var_ratio)
        eigenvalues = self.pca_model.explained_variance_

        # Kaiser criterion: components with eigenvalue > 1 (or > average eigenvalue)
        avg_eigenvalue = np.mean(eigenvalues)
        kaiser_count = int(np.sum(eigenvalues > avg_eigenvalue))

        # Threshold components
        comp_80 = int(np.searchsorted(self.cumulative_var_ratio, 0.80) + 1)
        comp_90 = int(np.searchsorted(self.cumulative_var_ratio, 0.90) + 1)
        comp_95 = int(np.searchsorted(self.cumulative_var_ratio, 0.95) + 1)

        # Loadings for top components (first 5)
        loadings_dict = {}
        for c_idx in range(min(5, n_comp)):
            weights = self.pca_model.components_[c_idx]
            top_pos_idx = np.argsort(weights)[::-1][:6]
            top_neg_idx = np.argsort(weights)[:6]
            loadings_dict[f"PC{c_idx+1}"] = {
                "top_positive": [(self.feature_names[i], float(weights[i])) for i in top_pos_idx],
                "top_negative": [(self.feature_names[i], float(weights[i])) for i in top_neg_idx]
            }

        return {
            "eigenvalues": eigenvalues.tolist(),
            "explained_var_ratio": self.explained_var_ratio.tolist(),
            "cumulative_var_ratio": self.cumulative_var_ratio.tolist(),
            "kaiser_intrinsic_dim": kaiser_count,
            "dim_80_var": comp_80,
            "dim_90_var": comp_90,
            "dim_95_var": comp_95,
            "loadings": loadings_dict
        }

    def compute_intrinsic_dimension_2nn(self, sample_size: int = 3000) -> float:
        """
        Estimates intrinsic dimension using the Two-Nearest-Neighbors (2NN) estimator
        (Facco et al., Nature Scientific Reports 2017).
        """
        n_samples = self.X.shape[0]
        rng = np.random.RandomState(42)
        idx = rng.choice(n_samples, min(sample_size, n_samples), replace=False)
        X_sub = self.X[idx]

        # Pairwise euclidean distances
        dist_mat = pairwise_distances(X_sub, metric="euclidean")
        np.fill_diagonal(dist_mat, np.inf)

        # 1st and 2nd nearest neighbor distances
        sorted_dists = np.sort(dist_mat, axis=1)
        r1 = sorted_dists[:, 0]
        r2 = sorted_dists[:, 1]

        # Filter out identical duplicate records (r1 > 0)
        valid = (r1 > 1e-6) & (r2 > r1)
        if np.sum(valid) < 50:
            return float(self.fit_pca()["dim_90_var"])

        mu = r2[valid] / r1[valid]
        # Empirical cumulative distribution
        N = len(mu)
        mu_sorted = np.sort(mu)
        F_emp = np.arange(1, N + 1) / N

        # Linear regression without intercept: -log(1 - F(mu)) = d * log(mu)
        y = -np.log(1.0 - F_emp[:-1])
        x = np.log(mu_sorted[:-1])
        # Slope through origin
        d_est = float(np.sum(x * y) / np.sum(x * x))
        return round(max(1.0, d_est), 2)

    def run_random_projection_baseline(self, n_components: int = 10, sample_size: int = 2000) -> Dict[str, float]:
        """
        Mandatory syllabus baseline: Random Projection at equal dimension.
        Measures variance retained and pairwise distance preservation (Johnson-Lindenstrauss distortion).
        """
        rng = np.random.RandomState(42)
        idx = rng.choice(self.X.shape[0], min(sample_size, self.X.shape[0]), replace=False)
        X_sub = self.X[idx]

        # Original pairwise distances
        orig_dist = pdist(X_sub, metric="euclidean")

        # PCA projection at n_components
        pca = PCA(n_components=n_components, random_state=42)
        X_pca_sub = pca.fit_transform(X_sub)
        pca_dist = pdist(X_pca_sub, metric="euclidean")
        pca_corr = float(np.corrcoef(orig_dist, pca_dist)[0, 1])

        # Random Projection at n_components
        grp = GaussianRandomProjection(n_components=n_components, random_state=42)
        X_grp_sub = grp.fit_transform(X_sub)
        grp_dist = pdist(X_grp_sub, metric="euclidean")
        grp_corr = float(np.corrcoef(orig_dist, grp_dist)[0, 1])

        return {
            "n_components": n_components,
            "pca_distance_correlation": round(pca_corr, 4),
            "random_projection_correlation": round(grp_corr, 4),
            "pca_cumulative_variance": round(float(pca.explained_variance_ratio_.sum()), 4)
        }

    def compute_tsne_2d(self, sample_size: int = 4000, perplexity: float = 35.0) -> Tuple[np.ndarray, np.ndarray]:
        """Computes 2D t-SNE manifold coordinates on a representative subsample."""
        rng = np.random.RandomState(42)
        idx = rng.choice(self.X.shape[0], min(sample_size, self.X.shape[0]), replace=False)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        coords = tsne.fit_transform(self.X[idx])
        return idx, coords


class DemandClusteringEngine:
    """Manages clustering algorithms, optimal-K evaluation, and cluster archetype naming."""

    def __init__(self, X: np.ndarray, feature_names: List[str], jobs_df: pd.DataFrame):
        self.X = X
        self.feature_names = np.array(feature_names)
        self.jobs_df = jobs_df

    def evaluate_k_range(self, k_min: int = 3, k_max: int = 15, sample_size: int = 5000) -> pd.DataFrame:
        """Evaluates clustering metrics across a range of k values."""
        rng = np.random.RandomState(42)
        eval_idx = rng.choice(self.X.shape[0], min(sample_size, self.X.shape[0]), replace=False)
        X_eval = self.X[eval_idx]
        baseline_labels = self.jobs_df["official_category_baseline"].iloc[eval_idx].values

        records = []
        for k in range(k_min, k_max + 1):
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3)
            labels = km.fit_predict(self.X)
            sub_labels = labels[eval_idx]

            inertia = float(km.inertia_)
            sil = float(silhouette_score(X_eval, sub_labels))
            db = float(davies_bouldin_score(X_eval, sub_labels))
            ch = float(calinski_harabasz_score(X_eval, sub_labels))
            ari = float(adjusted_rand_score(baseline_labels, sub_labels))
            nmi = float(normalized_mutual_info_score(baseline_labels, sub_labels))

            records.append({
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "ari_vs_baseline": ari,
                "nmi_vs_baseline": nmi
            })

        return pd.DataFrame(records)

    def fit_model(self, algorithm: str = "kmeans", n_clusters: int = 10, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits chosen clustering model.
        Returns (cluster_labels, cluster_centroids).
        """
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            labels = model.fit_predict(self.X)
            centroids = model.cluster_centers_

        elif algorithm == "minibatch_kmeans":
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init=3)
            labels = model.fit_predict(self.X)
            centroids = model.cluster_centers_

        elif algorithm == "gmm":
            # GMM on PCA-reduced space for stability and speed
            pca = PCA(n_components=min(15, self.X.shape[1]), random_state=42)
            X_red = pca.fit_transform(self.X)
            gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type="diag")
            labels = gmm.fit_predict(X_red)
            # Centroids in original skill space
            centroids = np.zeros((n_clusters, self.X.shape[1]))
            for c in range(n_clusters):
                mask = (labels == c)
                if np.sum(mask) > 0:
                    centroids[c] = self.X[mask].mean(axis=0)

        elif algorithm == "agglomerative":
            # For speed on large data, cluster 1500 representative centroids or fit on sub-sample
            sample_size = min(5000, self.X.shape[0])
            rng = np.random.RandomState(42)
            idx = rng.choice(self.X.shape[0], sample_size, replace=False)
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
            sub_labels = agg.fit_predict(self.X[idx])

            # Assign remaining via nearest centroid
            sub_centroids = np.zeros((n_clusters, self.X.shape[1]))
            for c in range(n_clusters):
                mask = (sub_labels == c)
                if np.sum(mask) > 0:
                    sub_centroids[c] = self.X[idx][mask].mean(axis=0)

            dists = pairwise_distances(self.X, sub_centroids, metric="euclidean")
            labels = dists.argmin(axis=1)
            centroids = np.zeros((n_clusters, self.X.shape[1]))
            for c in range(n_clusters):
                mask = (labels == c)
                if np.sum(mask) > 0:
                    centroids[c] = self.X[mask].mean(axis=0)

        elif algorithm == "dbscan":
            # DBSCAN on subsample
            eps = kwargs.get("eps", 1.8)
            min_samples = kwargs.get("min_samples", 20)
            db = DBSCAN(eps=eps, min_samples=min_samples)
            sample_size = min(5000, self.X.shape[0])
            rng = np.random.RandomState(42)
            idx = rng.choice(self.X.shape[0], sample_size, replace=False)
            sub_labels = db.fit_predict(self.X[idx])
            n_c = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
            if n_c < 2:
                # Fallback to KMeans if DBSCAN forms degenerate 1 cluster
                return self.fit_model(algorithm="kmeans", n_clusters=n_clusters)
            # Centroids of non-noise
            centroids = []
            valid_clusters = sorted([c for c in set(sub_labels) if c != -1])
            for c in valid_clusters:
                centroids.append(self.X[idx][sub_labels == c].mean(axis=0))
            centroids = np.array(centroids)
            dists = pairwise_distances(self.X, centroids, metric="euclidean")
            labels = dists.argmin(axis=1)

        elif algorithm == "baseline":
            # Official occupation categories as baseline pseudo-clusters
            cat_mapping = {cat: i for i, cat in enumerate(self.jobs_df["official_category_baseline"].unique())}
            labels = self.jobs_df["official_category_baseline"].map(cat_mapping).values
            n_cats = len(cat_mapping)
            centroids = np.zeros((n_cats, self.X.shape[1]))
            for c in range(n_cats):
                mask = (labels == c)
                if np.sum(mask) > 0:
                    centroids[c] = self.X[mask].mean(axis=0)

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return labels, centroids

    def generate_cluster_profiles(self, labels: np.ndarray, centroids: np.ndarray) -> List[Dict[str, Any]]:
        """Generates detailed domain profile and archetype name for each discovered cluster."""
        n_clusters = centroids.shape[0]
        total_jobs = len(labels)
        profiles = []

        for c in range(n_clusters):
            mask = (labels == c)
            cluster_size = int(np.sum(mask))
            cluster_share = round((cluster_size / total_jobs) * 100, 2)

            # Top skills by centroid activation
            top_skill_indices = np.argsort(centroids[c])[::-1][:8]
            top_skills = [
                {"skill": self.feature_names[i], "prevalence": round(float(centroids[c][i]), 3)}
                for i in top_skill_indices if centroids[c][i] > 0.05
            ]

            # Top job titles in this cluster
            top_titles = self.jobs_df.loc[mask, "job_title"].value_counts().head(4).to_dict()

            # Breakdown by official occupation categories
            baseline_dist = self.jobs_df.loc[mask, "official_category_baseline"].value_counts().head(3).to_dict()

            # Generate smart domain name based on top 3 skills
            skill_tokens = [s["skill"] for s in top_skills[:3]]
            archetype_name = self._infer_archetype_name(skill_tokens, top_titles)

            profiles.append({
                "cluster_id": c,
                "archetype_name": archetype_name,
                "size": cluster_size,
                "share_pct": cluster_share,
                "top_skills": top_skills,
                "top_titles": top_titles,
                "baseline_categories": baseline_dist
            })

        return profiles

    def _infer_archetype_name(self, top_skills: List[str], top_titles: Dict[str, int]) -> str:
        """Assigns an industry-grounded title based on top skills and frequent job titles."""
        skill_str = " ".join([s.lower() for s in top_skills])

        if any(w in skill_str for w in ["kubernetes", "docker", "aws", "terraform", "cloud"]):
            return "Cloud Infrastructure & DevOps Engineering"
        elif any(w in skill_str for w in ["sql", "bi", "power", "excel", "analytics", "statistics"]):
            return "Enterprise Data & Business Intelligence"
        elif any(w in skill_str for w in ["scikit", "pandas", "learning", "machine", "visualization"]):
            return "Machine Learning & AI Development"
        elif any(w in skill_str for w in ["rest", "css", "html", "node.js", "javascript"]):
            return "Full-Stack Web & Software Services"
        elif any(w in skill_str for w in ["ev", "battery", "charging", "diagnostics"]):
            return "Clean Mobility & EV Fleet Diagnostics"
        elif any(w in skill_str for w in ["mapping", "drone", "sensors", "soil", "bim"]):
            return "AgriTech & Geospatial Autonomous Sensing"
        elif any(w in skill_str for w in ["haccp", "food", "safety", "control", "equipment"]):
            return "Agro-Food Quality & Bio-Safety Control"
        elif any(w in skill_str for w in ["network", "security", "incident", "protocols"]):
            return "Cybersecurity & Telecom Infrastructure"
        elif any(w in skill_str for w in ["quality", "inspection", "cam", "standards", "machine"]):
            return "Precision Manufacturing & CNC Inspection"
        elif any(w in skill_str for w in ["customer", "service", "communication", "aml"]):
            return "Omnichannel Client Services & Financial Compliance"
        elif any(w in skill_str for w in ["lms", "curriculum", "assessment", "design"]):
            return "Modern Instructional Design & EdTech"
        elif any(w in skill_str for w in ["risk", "fraud"]):
            return "FinTech Risk Governance & Compliance"
        else:
            # Fallback to most frequent job title
            if top_titles:
                primary_title = list(top_titles.keys())[0]
                return f"{primary_title} Specialization"
            return f"Skill Cluster ({', '.join(top_skills[:2])})"


class ValidityAndDisagreementEngine:
    """Computes internal cluster validity and external disagreement against official categories."""

    def __init__(self, X: np.ndarray, labels: np.ndarray, baseline_categories: pd.Series):
        self.X = X
        self.labels = labels
        self.baseline_categories = baseline_categories

    def compute_internal_validity(self, sample_size: int = 5000) -> Dict[str, float]:
        """Calculates Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics."""
        rng = np.random.RandomState(42)
        idx = rng.choice(self.X.shape[0], min(sample_size, self.X.shape[0]), replace=False)
        X_sub = self.X[idx]
        labels_sub = self.labels[idx]

        unique_labels = np.unique(labels_sub)
        if len(unique_labels) < 2:
            return {"silhouette": 0.0, "davies_bouldin": 99.0, "calinski_harabasz": 0.0}

        sil = float(silhouette_score(X_sub, labels_sub))
        db = float(davies_bouldin_score(X_sub, labels_sub))
        ch = float(calinski_harabasz_score(X_sub, labels_sub))

        return {
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 2)
        }

    def compute_silhouette_samples(self, sample_size: int = 2500) -> Tuple[np.ndarray, np.ndarray]:
        """Computes per-sample silhouette values for plotting the silhouette diagram."""
        rng = np.random.RandomState(42)
        idx = rng.choice(self.X.shape[0], min(sample_size, self.X.shape[0]), replace=False)
        sil_vals = silhouette_samples(self.X[idx], self.labels[idx])
        return self.labels[idx], sil_vals

    def compute_taxonomy_disagreement(self) -> Dict[str, Any]:
        """
        Quantifies disagreement against the official baseline occupation categories.
        Objectives 2 & 3:
          - Adjusted Rand Index (ARI)
          - Normalized Mutual Information (NMI)
          - Homogeneity, Completeness, V-Measure
          - Contingency Matrix
          - Cluster Purity
        """
        baseline = self.baseline_categories.values
        unique_baseline = sorted(list(set(baseline)))
        cat_to_int = {c: i for i, c in enumerate(unique_baseline)}
        y_true = np.array([cat_to_int[c] for c in baseline])
        y_pred = self.labels

        ari = float(adjusted_rand_score(y_true, y_pred))
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        homo, comp, v_meas = homogeneity_completeness_v_measure(y_true, y_pred)

        # Cross-tabulation / Contingency Matrix
        contingency = pd.crosstab(
            pd.Series(y_pred, name="Discovered_Cluster"),
            pd.Series(baseline, name="Official_Category")
        )

        # Cluster Purity: sum(max count per cluster) / total
        cluster_max_counts = contingency.max(axis=1).sum()
        purity = float(cluster_max_counts / len(y_pred))

        # Disagreement rate (1 - ARI)
        disagreement_rate = round(1.0 - ari, 4)

        return {
            "adjusted_rand_index": round(ari, 4),
            "normalized_mutual_info": round(nmi, 4),
            "homogeneity": round(float(homo), 4),
            "completeness": round(float(comp), 4),
            "v_measure": round(float(v_meas), 4),
            "cluster_purity": round(purity, 4),
            "disagreement_rate": disagreement_rate,
            "contingency_matrix": contingency
        }


class TrainingProvisionAlignmentEngine:
    """
    Evaluates discovered demand clusters against training courses, detects unserved demand,
    and analyzes skill deficit vs overprovision in the curriculum.
    """

    def __init__(
        self,
        cluster_centroids: np.ndarray,
        cluster_profiles: List[Dict[str, Any]],
        courses_df: pd.DataFrame,
        X_courses: np.ndarray,
        feature_names: List[str],
        jobs_df: pd.DataFrame,
        X_jobs: np.ndarray
    ):
        self.centroids = cluster_centroids
        self.profiles = cluster_profiles
        self.courses_df = courses_df
        self.X_courses = X_courses
        self.feature_names = np.array(feature_names)
        self.jobs_df = jobs_df
        self.X_jobs = X_jobs

    def evaluate_course_alignment(self, unserved_threshold: float = 0.70) -> Dict[str, Any]:
        """
        Computes cosine similarity between each demand cluster centroid and all courses.
        Identifies unserved demand groups where similarity is below unserved_threshold.
        """
        n_clusters = self.centroids.shape[0]
        # Cosine similarity matrix: (n_clusters, n_courses)
        sim_matrix = cosine_similarity(self.centroids, self.X_courses)

        cluster_alignments = []
        unserved_count = 0

        for c in range(n_clusters):
            sims = sim_matrix[c]
            max_sim = float(sims.max())
            best_course_idx = int(sims.argmax())
            best_course = self.courses_df.iloc[best_course_idx]

            # Top 3 matching courses
            top3_idx = np.argsort(sims)[::-1][:3]
            top3_courses = [
                {
                    "course_id": self.courses_df.iloc[i]["course_id"],
                    "course_title": self.courses_df.iloc[i]["course_title"],
                    "course_category": self.courses_df.iloc[i]["course_category"],
                    "nsqf_level": float(self.courses_df.iloc[i]["nsqf_level_simulated"]),
                    "similarity": round(float(sims[i]), 3)
                }
                for i in top3_idx
            ]

            is_unserved = bool(max_sim < unserved_threshold)
            if is_unserved:
                unserved_count += 1

            # Missing skills: skills prominent in cluster centroid (>0.35) but absent in best matching course
            best_course_skills = set(
                [s.strip() for s in str(best_course["taught_skills"]).split(";")]
            )
            top_cluster_skills = [
                item["skill"] for item in self.profiles[c]["top_skills"] if item["prevalence"] >= 0.30
            ]
            missing_skills = [s for s in top_cluster_skills if s not in best_course_skills]

            status = "Unserved" if is_unserved else ("Moderately Served" if max_sim < 0.85 else "Well Served")

            cluster_alignments.append({
                "cluster_id": c,
                "archetype_name": self.profiles[c]["archetype_name"],
                "cluster_size": self.profiles[c]["size"],
                "max_similarity": round(max_sim, 3),
                "mean_top5_similarity": round(float(np.sort(sims)[-5:].mean()), 3),
                "status": status,
                "best_matching_course": best_course["course_title"],
                "best_course_category": best_course["course_category"],
                "best_course_nsqf": float(best_course["nsqf_level_simulated"]),
                "missing_skills": missing_skills,
                "top3_courses": top3_courses
            })

        return {
            "unserved_count": unserved_count,
            "total_clusters": n_clusters,
            "unserved_percentage": round((unserved_count / n_clusters) * 100, 1),
            "similarity_threshold": unserved_threshold,
            "cluster_alignments": cluster_alignments,
            "sim_matrix": sim_matrix
        }

    def compute_skill_market_gap(self) -> pd.DataFrame:
        """
        Compares market demand frequency (%) against course provision frequency (%).
        Identifies deficit skills and surplus skills.
        """
        # Job market frequency
        job_counts = self.X_jobs.sum(axis=0)
        total_jobs = self.X_jobs.shape[0]
        job_pct = (job_counts / total_jobs) * 100

        # Course catalog frequency
        course_counts = self.X_courses.sum(axis=0)
        total_courses = self.X_courses.shape[0]
        course_pct = (course_counts / total_courses) * 100

        # Gap = Job Demand % - Course Provision %
        gap = job_pct - course_pct

        gap_df = pd.DataFrame({
            "skill": self.feature_names,
            "job_demand_pct": np.round(job_pct, 2),
            "course_provision_pct": np.round(course_pct, 2),
            "gap_deficit_pct": np.round(gap, 2)
        })

        gap_df["status"] = np.where(
            gap_df["gap_deficit_pct"] > 3.0, "High Deficit (Under-Provided)",
            np.where(gap_df["gap_deficit_pct"] < -3.0, "Surplus (Over-Provided)", "Balanced")
        )

        return gap_df.sort_values(by="gap_deficit_pct", ascending=False).reset_index(drop=True)

    def generate_curriculum_recommendation(
        self,
        cluster_id: int,
        alignment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates modern NSQF curriculum modernization advice for a specific demand cluster.
        """
        cluster_info = alignment_data["cluster_alignments"][cluster_id]
        profile = self.profiles[cluster_id]

        recommended_title = f"Advanced Qualification Pack: {profile['archetype_name']}"
        dominant_skills = [item["skill"] for item in profile["top_skills"][:6]]
        missing_skills = cluster_info["missing_skills"]

        # Recommend NSQF Level: 5 for technician/hands-on, 6 for specialist, 7 for engineer/architect
        if any(w in profile["archetype_name"].lower() for w in ["engineering", "ai", "cloud", "full-stack"]):
            nsqf_level = 6.0
            duration = 320
        elif any(w in profile["archetype_name"].lower() for w in ["intelligence", "risk", "specialist"]):
            nsqf_level = 5.5
            duration = 240
        else:
            nsqf_level = 5.0
            duration = 200

        return {
            "target_cluster": profile["archetype_name"],
            "cluster_size": profile["size"],
            "current_status": cluster_info["status"],
            "max_existing_course_match": f"{cluster_info['best_matching_course']} ({cluster_info['max_similarity'] * 100:.1f}%)",
            "proposed_course_title": recommended_title,
            "suggested_nsqf_level": nsqf_level,
            "recommended_duration_hours": duration,
            "core_skills_to_include": dominant_skills,
            "curriculum_gap_modules": [
                f"Module: Applied {skill} Integration & Hands-on Lab" for skill in missing_skills
            ] if missing_skills else ["Module: Advanced Industry Capstone & Production Readiness"],
            "cross_cutting_domains": list(profile["baseline_categories"].keys())
        }
