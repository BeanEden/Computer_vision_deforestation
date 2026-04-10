import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from typing import Tuple, List

def sample_pixels(features: np.ndarray, n_samples: int = 2000) -> np.ndarray:
    """Subsamples pixels to speed up expensive algorithms like CAH.

    Args:
        features: Full feature vector array.
        n_samples: Number of samples to draw.

    Returns:
        A subset of the feature vector array.
    """
    indices = np.random.choice(len(features), n_samples, replace=False)
    return features[indices]

def run_cah(sample: np.ndarray) -> np.ndarray:
    """Executes Hierarchical Agglomerative Clustering (CAH).

    Args:
        sample: Sampled feature vector array.

    Returns:
        The linkage matrix Z.
    """
    return linkage(sample, method='ward')

def run_kmeans(features: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Executes K-Means clustering.

    Args:
        features: Full feature vector array.
        n_clusters: Number of clusters (k).

    Returns:
        A tuple (labels, centers).
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(features)
    return labels, km.cluster_centers_

def identify_vegetation_cluster(features: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> int:
    """Automatically identifies which cluster index represents vegetation.
    
    Rule: Vegetation typically has a high green ratio (feature at index 6).

    Args:
        features: Full feature vector array.
        labels: Cluster labels.
        centers: Cluster coordinates (centroids).

    Returns:
        The index of the vegetation cluster.
    """
    # L'index 6 dans notre pipeline est le ratio vert
    green_ratios = centers[:, 6]
    return int(np.argmax(green_ratios))

def get_vegetation_mask(labels: np.ndarray, veg_idx: int, shape: Tuple[int, int]) -> np.ndarray:
    """Converts labels to a binary mask for the identified vegetation.

    Args:
        labels: Flat cluster labels.
        veg_idx: Index of the vegetation cluster.
        shape: Original image (height, width).

    Returns:
        A binary mask array (0 or 1).
    """
    mask = (labels == veg_idx).astype(np.uint8)
    return mask.reshape(shape)
