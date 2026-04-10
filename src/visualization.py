import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from typing import Dict, Optional, List

# Configuration Headless pour serveurs
import matplotlib
matplotlib.use('Agg')

def plot_side_by_side(img_a: np.ndarray, img_b: np.ndarray, 
                      title_a: str = "t0", title_b: str = "t1", 
                      main_title: str = "Comparison", 
                      save_path: Optional[str] = None) -> None:
    """Displays two images side by side for visual comparison.

    Args:
        img_a: First image array.
        img_b: Second image array.
        title_a: Label for the first image.
        title_b: Label for the second image.
        main_title: Title of the whole figure.
        save_path: Path to save the plot if provided.
    """
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_a)
    plt.title(title_a)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_b)
    plt.title(title_b)
    plt.axis('off')
    
    plt.suptitle(main_title, fontsize=16)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_psnr_comparison(results: Dict[str, float], title: str = "PSNR Comparison") -> None:
    """Bar plot to justify the choice of denoising filter based on PSNR.

    Args:
        results: Dictionary mapping method names to PSNR values.
        title: Title of the chart.
    """
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title(title)
    plt.ylabel("PSNR (dB)")
    plt.show()

def plot_dendrogram(Z: np.ndarray, title: str = "Dendrogram (CAH)") -> None:
    """Visualizes the hierarchical clustering to justify the number of clusters (k).

    Args:
        Z: Linkage matrix from CAH.
        title: Title of the dendrogram.
    """
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode='lastp', p=12)
    plt.title(title)
    plt.axhline(y=10, color='r', linestyle='--') # Exemple de seuil
    plt.show()

def plot_feature_distributions(features: np.ndarray, labels: np.ndarray, 
                               feat_names: List[str]) -> None:
    """Analyzes cluster distributions for each feature to verify separability.

    Args:
        features: Flattened feature vector array.
        labels: Assigned cluster labels.
        feat_names: Names of the features.
    """
    n_feat = features.shape[1]
    plt.figure(figsize=(15, 10))
    for i in range(n_feat):
        plt.subplot(3, 3, i+1)
        sns.boxplot(x=labels[::100], y=features[::100, i])
        plt.title(feat_names[i])
    plt.tight_layout()
    plt.show()

def plot_change_map(change_img: np.ndarray, save_path: Optional[str] = None) -> None:
    """Displays the final expert change map.

    Args:
        change_img: 3-channel change image array.
        save_path: Path to save the map if provided.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(change_img)
    plt.title("Carte de la Résilience Forestière (Expert View)")
    plt.axis('off')
    if save_path:
        # On sauvegarde le PNG sans les axes pour la dashboard
        plt.imsave(save_path + ".png", change_img)
    plt.show()
