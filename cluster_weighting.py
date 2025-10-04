import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm
import pickle
import os
from Functions_v7 import *

def compute_rna_edit_distance_matrix(sequences, cache_path='edit_distance_cache.pkl'):
    """
    Compute edit distance matrix for RNA sequences with caching.

    Args:
        sequences: List of RNA sequence strings
        cache_path: Path to cache computed distances

    Returns:
        Square distance matrix (n_sequences, n_sequences)
    """
    n_sequences = len(sequences)

    # Try to load cached distances
    if os.path.exists(cache_path):
        print(f"Loading cached edit distances from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            if cached_data['n_sequences'] == n_sequences:
                return cached_data['distance_matrix']

    print(f"Computing edit distances for {n_sequences} sequences...")
    distance_matrix = np.zeros((n_sequences, n_sequences))

    # Compute pairwise edit distances
    for i in tqdm(range(n_sequences), desc="Computing edit distances"):
        for j in range(i + 1, n_sequences):
            # Use Levenshtein distance for sequence similarity
            dist = levenshtein_distance(sequences[i], sequences[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric matrix

    # Cache computed distances
    cache_data = {
        'n_sequences': n_sequences,
        'distance_matrix': distance_matrix
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"Edit distance computation complete. Cached to {cache_path}")
    return distance_matrix


def perform_edit_distance_clustering(sequences, n_clusters=None, distance_threshold=10):
    """
    Perform hierarchical clustering based on edit distances.

    Args:
        sequences: List of RNA sequence strings
        n_clusters: Fixed number of clusters (if None, use distance_threshold)
        distance_threshold: Maximum distance for cluster formation

    Returns:
        cluster_labels: Array of cluster assignments
        cluster_stats: Dictionary with clustering statistics
    """
    distance_matrix = compute_rna_edit_distance_matrix(sequences)

    # Configure clustering parameters
    if n_clusters is not None:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
    else:
        clustering = AgglomerativeClustering(
            distance_threshold=distance_threshold,
            n_clusters=None,
            metric='precomputed',
            linkage='average'
        )

    # Perform clustering
    cluster_labels = clustering.fit_predict(distance_matrix)

    # Generate clustering statistics
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = {
        'n_clusters': len(unique_clusters),
        'cluster_sizes': dict(zip(unique_clusters, cluster_counts)),
        'min_cluster_size': cluster_counts.min(),
        'max_cluster_size': cluster_counts.max(),
        'singleton_clusters': (cluster_counts == 1).sum(),
        'avg_cluster_size': cluster_counts.mean()
    }

    print(f"Clustering Results:")
    print(f"  Total clusters: {cluster_stats['n_clusters']}")
    print(f"  Cluster size range: {cluster_stats['min_cluster_size']}-{cluster_stats['max_cluster_size']}")
    print(f"  Singleton clusters: {cluster_stats['singleton_clusters']}")
    print(f"  Average cluster size: {cluster_stats['avg_cluster_size']:.1f}")

    return cluster_labels, cluster_stats


def compute_cluster_sample_weights(cluster_labels, alpha=0.5):
    """
    Compute sample weights based on cluster membership.
    Winner's strategy: weight ∝ 1/sqrt(count_in_cluster)

    Args:
        cluster_labels: Array of cluster assignments
        alpha: Exponent for weight computation (0.5 = sqrt, per winner)

    Returns:
        sample_weights: Array of weights for each sample
    """
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    cluster_count_map = dict(zip(unique_clusters, cluster_counts))

    # Compute weights: 1/sqrt(cluster_size)
    sample_weights = np.array([
        1.0 / (cluster_count_map[label] ** alpha)
        for label in cluster_labels
    ])

    # Normalize weights to have unit mean
    sample_weights = sample_weights / sample_weights.mean()

    print(f"Sample weight statistics:")
    print(f"  Weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")
    print(f"  Weight std: {sample_weights.std():.3f}")

    return sample_weights


def integrate_cluster_weights_into_training(json_data, distance_threshold=10):
    """
    Integrate edit distance clustering into training pipeline.

    Args:
        json_data: Training dataframe with sequences
        distance_threshold: Clustering distance threshold

    Returns:
        json_data: Enhanced dataframe with cluster weights
    """
    sequences = json_data['sequence'].tolist()

    # Perform clustering
    cluster_labels, cluster_stats = perform_edit_distance_clustering(
        sequences, distance_threshold=distance_threshold
    )

    # Compute sample weights
    cluster_weights = compute_cluster_sample_weights(cluster_labels)

    # Add to dataframe
    json_data['cluster_id'] = cluster_labels
    json_data['cluster_weight'] = cluster_weights

    # Enhanced error weight computation
    if 'error_weight' in json_data.columns:
        # Combine with existing error weights
        json_data['combined_weight'] = json_data['error_weight'] * json_data['cluster_weight']
    else:
        json_data['combined_weight'] = json_data['cluster_weight']

    return json_data


# INTEGRATION: Modify train_v7.py to include cluster weighting
def enhanced_error_weight_computation(json_data, opts):
    """
    Enhanced error weight computation with cluster weighting.
    Replace the existing error weight computation in train_v7.py
    """
    # Original error weights
    error_weights = get_errors(json_data)
    error_weights = opts.error_alpha + np.exp(-error_weights * opts.error_beta)

    # Add cluster weighting
    json_data_enhanced = integrate_cluster_weights_into_training(json_data)
    cluster_weights = json_data_enhanced['cluster_weight'].values

    # Combine weights: element-wise multiplication
    # error_weights shape: (n_samples, seq_len, 5)
    # cluster_weights shape: (n_samples,)
    enhanced_weights = error_weights * cluster_weights[:, np.newaxis, np.newaxis]

    print(f"Enhanced weighting applied:")
    print(f"  Original weight range: {error_weights.min():.3f} - {error_weights.max():.3f}")
    print(f"  Cluster weight range: {cluster_weights.min():.3f} - {cluster_weights.max():.3f}")
    print(f"  Combined weight range: {enhanced_weights.min():.3f} - {enhanced_weights.max():.3f}")

    return enhanced_weights, json_data_enhanced