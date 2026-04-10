import cv2
import numpy as np

def apply_morphology(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Refines the segmentation mask using mathematical morphology.
    
    Operations: Opening (remove noise) followed by Closing (fill holes).

    Args:
        mask: Input binary mask (0 or 1).
        kernel_size: Size of the square structuring element.

    Returns:
        The cleaned binary mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Élimination du bruit (Sel et Poivre)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Comblement des lacunes dans la canopée
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    
    return mask_close
