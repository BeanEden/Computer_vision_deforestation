import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import silhouette_score
from typing import Tuple

def get_feature_vector(img_rgb: np.ndarray) -> np.ndarray:
    """Extracts a multi-spectral feature vector for every pixel.
    
    Features: RGB, HSV, Green Ratio, and Local Statistics.

    Args:
        img_rgb: Source RGB image array.

    Returns:
        A 2D array of shape (N_pixels, N_features).
    """
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # 1. RGB
    r, g, b = cv2.split(img_float)
    
    # 2. HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(img_hsv)
    h /= 180.0 # Normalization
    s /= 255.0
    v /= 255.0
    
    # 3. Ratio Vert
    green_ratio = g / (r + g + b + 1e-6)
    
    # 4. Statistiques Locales (Texture simple)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean_kernel = np.ones((5, 5), np.float32) / 25.0
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, mean_kernel) / 255.0
    
    features = [r, g, b, h, s, v, green_ratio, local_mean]
    stacked = np.stack(features, axis=-1)
    
    return stacked.reshape(-1, len(features))

def compute_glcm_features(img_gray: np.ndarray) -> Tuple[float, float]:
    """Computes second-order texture features using Gray-Level Co-occurrence Matrix.

    Args:
        img_gray: Grayscale image array.

    Returns:
        A tuple of (contrast, correlation).
    """
    glcm = graycomatrix(img_gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return float(contrast), float(correlation)

def get_silhouette(features: np.ndarray, labels: np.ndarray, n_samples: int = 1000) -> float:
    """Calculates the silhouette score to evaluate cluster cohesion.

    Args:
        features: Feature vector 2D array.
        labels: Assigned labels array.
        n_samples: Number of samples to use for calculation (for speed).

    Returns:
        The silhouette score between -1 and 1.
    """
    indices = np.random.choice(len(features), n_samples, replace=False)
    return float(silhouette_score(features[indices], labels[indices]))
