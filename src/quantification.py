import numpy as np
from typing import Dict, Any

def compute_deforestation_stats(mask0: np.ndarray, mask1: np.ndarray) -> Dict[str, Any]:
    """Quantifies the temporal change in canopy surface.

    Args:
        mask0: Binary mask at t0.
        mask1: Binary mask at t1.

    Returns:
        A dictionary containing pixels counts and the loss percentage.
    """
    s0 = int(np.sum(mask0))
    s1 = int(np.sum(mask1))
    
    # Évolution relative
    loss_pct = 0.0
    if s0 > 0:
        loss_pct = ((s0 - s1) / s0) * 100
        
    return {
        'surface_t0': s0,
        'surface_t1': s1,
        'loss_percentage': loss_pct
    }

def get_change_map(mask0: np.ndarray, mask1: np.ndarray) -> np.ndarray:
    """Generates a visualization of changes between t0 and t1.
    
    Logic:
        - Red: Loss (Deforestation)
        - Blue: Gain (Revegetation)
        - Gray/Original: Stability

    Args:
        mask0: Binary mask at t0.
        mask1: Binary mask at t1.

    Returns:
        A 3-channel image representing geographical change.
    """
    h, w = mask0.shape
    change_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Déforestation (Présent en t0, absent en t1) -> Rouge
    loss = (mask0 == 1) & (mask1 == 0)
    change_img[loss] = [255, 0, 0]
    
    # Revégétalisation (Absent en t0, présent en t1) -> Bleu
    gain = (mask0 == 0) & (mask1 == 1)
    change_img[gain] = [0, 0, 255]
    
    # Stabilité (Présent aux deux dates) -> Vert discret
    stable = (mask0 == 1) & (mask1 == 1)
    change_img[stable] = [0, 255, 0]
    
    return change_img
