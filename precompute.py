"""
Precomputation Pipeline for Skill Demand Structure Discovery Dashboard
Author: Jashandeep Singh Gill (12504818) - Group G25
Generates high-performance cached embeddings, metrics, and models.
Compatible with Python 3.15.2
"""

import time
import os
import pickle
import numpy as np
import pandas as pd
from ml_engine import (
    DataLoader,
    DimensionalityReductionEngine,
    DemandClusteringEngine,
    ValidityAndDisagreementEngine,
    TrainingProvisionAlignmentEngine
)

def run_precomputation():
    print("=====================================================================")
    print("Starting Precomputation Pipeline for Skill Demand Discovery Dashboard")
    print("=====================================================================")
    start_total = time.time()

    # 1. Load Data
    t0 = time.time()
    print("1. Loading datasets and vectorizing skill matrices...")
    loader = DataLoader().load_data()
    print(f"   Done in {time.time()-t0:.2f}s | Jobs: {loader.df_jobs.shape[0]}, Courses: {loader.df_courses.shape[0]}, Skills: {len(loader.feature_skills)}")

    # 2. Dimensionality Reduction
    t0 = time.time()
    print("2. Computing PCA (2D, 3D, scree, loadings, intrinsic dimensionality)...")
    dr_engine = DimensionalityReductionEngine(loader.X_jobs, loader.feature_skills)
    pca_stats = dr_engine.fit_pca(max_components=40)
    print(f"   PCA completed in {time.time()-t0:.2f}s | Kaiser Dim: {pca_stats['kaiser_intrinsic_dim']}, 80% Var Dim: {pca_stats['dim_80_var']}")

    t0 = time.time()
    print("   Computing 2NN Intrinsic Dimensionality...")
    intrinsic_dim_2nn = dr_engine.compute_intrinsic_dimension_2nn(sample_size=3000)
    print(f"   2NN Intrinsic Dim Estimate: {intrinsic_dim_2nn} (in {time.time()-t0:.2f}s)")

    t0 = time.time()
    print("   Computing Random Projection baseline comparison (Mandatory Syllabus Baseline)...")
    rp_baseline = dr_engine.run_random_projection_baseline(n_components=10, sample_size=2000)
    print(f"   RP Baseline: PCA Dist Corr={rp_baseline['pca_distance_correlation']}, RP Dist Corr={rp_baseline['random_projection_correlation']}")

    t0 = time.time()
    print("   Computing 2D t-SNE manifold embedding for 4,000 points...")
    tsne_sample_idx, tsne_coords_2d = dr_engine.compute_tsne_2d(sample_size=4000, perplexity=35.0)
    print(f"   t-SNE done in {time.time()-t0:.2f}s")

    # 3. K-range Evaluation (Elbow, Silhouette, Davies-Bouldin, ARI, NMI)
    t0 = time.time()
    print("3. Evaluating K range (k=3 to 15) for optimal cluster selection...")
    clustering_engine = DemandClusteringEngine(loader.X_jobs, loader.feature_skills, loader.df_jobs)
    k_eval_df = clustering_engine.evaluate_k_range(k_min=3, k_max=15, sample_size=5000)
    print(f"   K-range evaluation finished in {time.time()-t0:.2f}s")

    # 4. Fit Baseline vs Default KMeans (K=10)
    t0 = time.time()
    print("4. Fitting default Demand Clustering (K=10 KMeans)...")
    kmeans_labels, kmeans_centroids = clustering_engine.fit_model("kmeans", n_clusters=10)
    cluster_profiles = clustering_engine.generate_cluster_profiles(kmeans_labels, kmeans_centroids)
    print(f"   Clustering and profile generation done in {time.time()-t0:.2f}s")

    # 5. Validity and Taxonomy Disagreement Metrics
    t0 = time.time()
    print("5. Computing cluster validity and taxonomy disagreement metrics...")
    val_engine = ValidityAndDisagreementEngine(
        loader.X_jobs,
        kmeans_labels,
        loader.df_jobs["official_category_baseline"]
    )
    internal_validity = val_engine.compute_internal_validity(sample_size=5000)
    disagreement_metrics = val_engine.compute_taxonomy_disagreement()
    sil_sample_labels, sil_sample_vals = val_engine.compute_silhouette_samples(sample_size=2500)
    print(f"   Done in {time.time()-t0:.2f}s | Silhouette: {internal_validity['silhouette']}, ARI: {disagreement_metrics['adjusted_rand_index']}")

    # 6. Training Provision Alignment and Unserved Demand Groups
    t0 = time.time()
    print("6. Analyzing course alignment and unserved demand groups...")
    align_engine = TrainingProvisionAlignmentEngine(
        kmeans_centroids,
        cluster_profiles,
        loader.df_courses,
        loader.X_courses,
        loader.feature_skills,
        loader.df_jobs,
        loader.X_jobs
    )
    alignment_results = align_engine.evaluate_course_alignment(unserved_threshold=0.70)
    skill_gap_df = align_engine.compute_skill_market_gap()
    print(f"   Alignment complete in {time.time()-t0:.2f}s | Unserved Clusters: {alignment_results['unserved_count']}/{alignment_results['total_clusters']}")

    # 7. Package and Save Artifacts
    print("7. Packaging cache artifact...")
    artifacts = {
        "feature_skills": loader.feature_skills,
        "pca_stats": pca_stats,
        "pca_coords_2d": dr_engine.pca_coords_2d,
        "pca_coords_3d": dr_engine.pca_coords_3d,
        "intrinsic_dim_2nn": intrinsic_dim_2nn,
        "rp_baseline": rp_baseline,
        "tsne_sample_idx": tsne_sample_idx,
        "tsne_coords_2d": tsne_coords_2d,
        "k_eval_df": k_eval_df,
        "kmeans_labels": kmeans_labels,
        "kmeans_centroids": kmeans_centroids,
        "cluster_profiles": cluster_profiles,
        "internal_validity": internal_validity,
        "disagreement_metrics": disagreement_metrics,
        "sil_sample_labels": sil_sample_labels,
        "sil_sample_vals": sil_sample_vals,
        "alignment_results": alignment_results,
        "skill_gap_df": skill_gap_df
    }

    cache_path = os.path.join(loader.data_dir, "precomputed_artifacts.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved artifacts to {cache_path} ({os.path.getsize(cache_path) / (1024*1024):.2f} MB)")
    print(f"Total Precomputation completed in {time.time() - start_total:.2f}s successfully!")
    print("=====================================================================")

if __name__ == "__main__":
    run_precomputation()
