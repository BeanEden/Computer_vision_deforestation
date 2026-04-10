import cv2
import numpy as np

def denoise_image(img_rgb: np.ndarray, method: str = 'gaussian', kernel_size: int = 3) -> np.ndarray:
    """Applies denoising to the image to improve signal quality.

    Args:
        img_rgb: Source RGB image array.
        method: Denoising method ('gaussian', 'median', 'bilateral').
        kernel_size: Size of the filter kernel (must be odd).

    Returns:
        The denoised image array.
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(img_rgb, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img_rgb, kernel_size, 75, 75)
    return img_rgb

def improve_contrast(img_rgb: np.ndarray) -> np.ndarray:
    """Applies CLAHE on the L channel of Lab space to improve contrast.

    Args:
        img_rgb: Source RGB image array.

    Returns:
        The contrast-enhanced image array.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def preprocess(img_rgb: np.ndarray, denoise_method: str = 'gaussian', k_size: int = 3, contrast: bool = True) -> np.ndarray:
    """Master expert preprocessing pipeline.

    Args:
        img_rgb: Source RGB image array.
        denoise_method: Denoising method to apply.
        k_size: Kernel size for denoising.
        contrast: Whether to apply contrast enhancement.

    Returns:
        The fully preprocessed image array.
    """
    img = img_rgb.copy()
    if denoise_method:
        img = denoise_image(img, method=denoise_method, kernel_size=k_size)
    if contrast:
        img = improve_contrast(img)
    return img
