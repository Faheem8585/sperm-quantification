"""
Background subtraction for removing static microscopy artifacts.

Separates moving sperm from stationary background.
"""

import numpy as np
import cv2
from typing import Optional, List
from scipy.ndimage import morphology


class BackgroundSubtractor:
    """
    Base class for background subtraction algorithms.
    """
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction to frame.
        
        Args:
            frame: Input frame.
        
        Returns:
            Foreground mask or subtracted image.
        """
        raise NotImplementedError


class MedianBackgroundSubtractor(BackgroundSubtractor):
    """
    Median-based background estimation.
    
    Computes background as median over a window of frames,
    then subtracts from each frame.
    """
    
    def __init__(self, history: int = 100):
        """
        Initialize median background subtractor.
        
        Args:
            history: Number of frames to use for background estimation.
        """
        self.history = history
        self.frame_buffer: List[np.ndarray] = []
        self.background: Optional[np.ndarray] = None
    
    def update_background(self, frame: np.ndarray):
        """
        Update background model with new frame.
        
        Args:
            frame: Input frame.
        """
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) > self.history:
            self.frame_buffer.pop(0)
        
        # Compute median background
        if len(self.frame_buffer) >= min(10, self.history):
            self.background = np.median(self.frame_buffer, axis=0).astype(np.uint8)
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction.
        
        Args:
            frame: Input frame.
        
        Returns:
            Background-subtracted image.
        """
        self.update_background(frame)
        
        if self.background is None:
            return frame
        
        # Subtract background
        subtracted = cv2.absdiff(frame, self.background)
        
        return subtracted


class MOG2BackgroundSubtractor(BackgroundSubtractor):
    """
    Gaussian Mixture Model (MOG2) background subtraction.
    
    Adaptive background model that handles gradual lighting changes.
    Implemented using OpenCV's BackgroundSubtractorMOG2.
    """
    
    def __init__(
        self,
        history: int = 500,
        var_threshold: float = 16,
        detect_shadows: bool = False,
        learning_rate: float = 0.01
    ):
        """
        Initialize MOG2 background subtractor.
        
        Args:
            history: Length of history for background model.
            var_threshold: Threshold on squared Mahalanobis distance.
            detect_shadows: Whether to detect shadows (slower).
            learning_rate: Learning rate for background update (0-1).
                          Higher = faster adaptation to changes.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.learning_rate = learning_rate
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply MOG2 background subtraction.
        
        Args:
            frame: Input frame.
        
        Returns:
            Foreground mask (binary image).
        """
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        return fg_mask


def morphological_cleanup(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Clean up binary mask using morphological operations.
    
    Removes small noise and fills small holes.
    
    Args:
        mask: Binary mask.
        kernel_size: Size of morphological kernel.
        iterations: Number of iterations for opening/closing.
    
    Returns:
        Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Opening: remove small objects (erosion + dilation)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # Closing: fill small holes (dilation + erosion)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return closed


class BackgroundSubtractionPipeline:
    """
    Complete background subtraction pipeline with post-processing.
    """
    
    def __init__(
        self,
        method: str = 'mog2',
        cleanup: bool = True,
        **kwargs
    ):
        """
        Initialize background subtraction pipeline.
        
        Args:
            method: Background subtraction method ('median' or 'mog2').
            cleanup: Whether to apply morphological cleanup.
            **kwargs: Additional arguments for background subtractor.
        """
        self.method = method
        self.cleanup = cleanup
        
        if method == 'median':
            self.subtractor = MedianBackgroundSubtractor(**kwargs)
        elif method == 'mog2':
            self.subtractor = MOG2BackgroundSubtractor(**kwargs)
        else:
            raise ValueError(f"Unknown background subtraction method: {method}")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame through background subtraction pipeline.
        
        Args:
            frame: Input frame (grayscale).
        
        Returns:
            Processed foreground image or mask.
        """
        # Apply background subtraction
        result = self.subtractor.apply(frame)
        
        # Morphological cleanup for binary masks
        if self.cleanup and self.method == 'mog2':
            result = morphological_cleanup(result)
        
        return result
