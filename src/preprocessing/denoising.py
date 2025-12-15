"""
Denoising algorithms for microscopy images.

Implements various noise reduction techniques while preserving sperm morphology.
"""

import numpy as np
import cv2
from skimage import restoration
from typing import Optional


def gaussian_denoise(
    image: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian blur for denoising.
    
    Args:
        image: Input image (grayscale).
        sigma: Standard deviation of Gaussian kernel.
    
    Returns:
        Denoised image.
    """
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def bilateral_denoise(
    image: np.ndarray,
    sigma_color: float = 75,
    sigma_space: float = 75,
    diameter: int = 9
) -> np.ndarray:
    """
    Apply bilateral filtering for edge-preserving denoising.
    
   Bilateral filtering preserves edges while smoothing uniform regions,
    ideal for maintaining sperm boundaries.
    
    Args:
        image: Input image (grayscale).
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.
        diameter: Diameter of pixel neighborhood.
    
    Returns:
        Denoised image.
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def nlm_denoise(
    image: np.ndarray,
    h: float = 10,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Apply Non-Local Means denoising.
    
    NLM exploits self-similarity in images for superior noise reduction.
    More computationally expensive but effective for microscopy.
    
    Args:
        image: Input image (grayscale).
        h: Filter strength. Higher h removes more noise but also removes detail.
        template_window_size: Size of template patch.
        search_window_size: Size of search area.
    
    Returns:
        Denoised image.
    """
    return cv2.fastNlMeansDenoising(
        image,
        h=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )


def normalize_frame(
    image: np.ndarray,
    method: str = 'clahe',
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Normalize image contrast.
    
    Args:
        image: Input image (grayscale).
        method: Normalization method ('clahe', 'minmax', 'zscore').
        clip_limit: Contrast limiting parameter for CLAHE.
        tile_grid_size: Grid size for CLAHE tiles.
    
    Returns:
        Normalized image.
    """
    if method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    elif method == 'minmax':
        # Min-max normalization to [0, 255]
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            normalized = 255 * (image - min_val) / (max_val - min_val)
            return normalized.astype(np.uint8)
        return image
    
    elif method == 'zscore':
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            normalized = (image - mean) / std
            # Clip to ±3 sigma and scale to [0, 255]
            normalized = np.clip(normalized, -3, 3)
            normalized = 255 * (normalized + 3) / 6
            return normalized.astype(np.uint8)
        return image
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_frame(
    image: np.ndarray,
    denoise_method: str = 'bilateral',
    normalize: bool = True,
    **kwargs
) -> np.ndarray:
    """
    High-level preprocessing function combining multiple steps.
    
    Args:
        image: Input image (grayscale).
        denoise_method: Denoising method ('gaussian', 'bilateral', 'nlm', 'none').
        normalize: Whether to apply CLAHE normalization.
        **kwargs: Additional arguments for denoising functions.
    
    Returns:
        Preprocessed image.
    """
    processed = image.copy()
    
    # Denoising
    if denoise_method == 'gaussian':
        sigma = kwargs.get('gaussian_sigma', 1.0)
        processed = gaussian_denoise(processed, sigma=sigma)
    
    elif denoise_method == 'bilateral':
        sigma_color = kwargs.get('bilateral_sigma_color', 75)
        sigma_space = kwargs.get('bilateral_sigma_space', 75)
        processed = bilateral_denoise(
            processed,
            sigma_color=sigma_color,
            sigma_space=sigma_space
        )
    
    elif denoise_method == 'nlm':
        h = kwargs.get('nlm_h', 10)
        processed = nlm_denoise(processed, h=h)
    
    # Normalization
    if normalize:
        processed = normalize_frame(processed, method='clahe')
    
    return processed
