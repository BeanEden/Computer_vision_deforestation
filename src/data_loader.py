import os
import cv2
import glob
import numpy as np
from typing import List, Tuple

def get_image_pairs(data_dir: str, folder_a: str = 'Lo1_A', folder_b: str = 'Lot1_B') -> List[Tuple[str, str]]:
    """Identifies common image pairs between two specified folders.

    Args:
        data_dir: Base directory containing the image folders.
        folder_a: Name of the first folder (e.g., 'Lo1_A').
        folder_b: Name of the second folder (e.g., 'Lot1_B').

    Returns:
        A list of tuples containing absolute paths (path_a, path_b) for matching filenames.
    """
    path_a = os.path.join(data_dir, folder_a)
    path_b = os.path.join(data_dir, folder_b)
    
    files_a = glob.glob(os.path.join(path_a, '*.png'))
    files_b = glob.glob(os.path.join(path_b, '*.png'))
    
    names_a = {os.path.basename(f): f for f in files_a}
    names_b = {os.path.basename(f): f for f in files_b}
    
    common_names = sorted(set(names_a.keys()) & set(names_b.keys()))
    pairs = [(names_a[name], names_b[name]) for name in common_names]
    
    return pairs

def load_pair(pair: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a pair of images from paths and converts them to RGB.

    Args:
        pair: Tuple of paths (image_t0, image_t1).

    Returns:
        A tuple of numpy arrays (img_t0, img_t1) in RGB format.
    """
    img_a = cv2.imread(pair[0])
    img_b = cv2.imread(pair[1])
    
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
    
    return img_a, img_b
