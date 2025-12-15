"""
Advanced segmentation methods for crowded sperm fields.

Implements watershed and morphological segmentation techniques.
"""

import numpy as np
import cv2
from skimage import morphology, segmentation, measure
from scipy import ndimage
from typing import List, Tuple, Optional

from .blob_detector import Detection


def watershed_segmentation(
    image: np.ndarray,
    min_distance: int = 5,
    threshold_rel: float = 0.5
) -> Tuple[np.ndarray, List[Detection]]:
    """
    Segment touching sperm using watershed algorithm.
    
    The watershed algorithm treats the image as a topographic surface
    and finds watershed lines to separate adjacent objects.
    
    Args:
        image: Input image (grayscale).
        min_distance: Minimum distance between local maxima (seeds).
        threshold_rel: Relative threshold for peak detection (0-1).
    
    Returns:
        Tuple of (labeled_image, detections).
        labeled_image: Label map where each object has a unique integer.
        detections: List of Detection objects.
    """
    # Normalize image
    if image.dtype == np.uint8:
        image_norm = image.astype(float) / 255.0
    else:
        image_norm = image
    
    # Threshold to get binary mask
    threshold = threshold_rel * image_norm.max()
    binary = image_norm > threshold
    
    # Distance transform
    distance = ndimage.distance_transform_edt(binary)
    
    # Find local maxima as markers
    from skimage.feature import peak_local_max
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary.astype(int)
    )
    
    # Create markers
    markers = np.zeros_like(distance, dtype=int)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    
    # Apply watershed
    labels = segmentation.watershed(-distance, markers, mask=binary)
    
    # Extract detections from labeled regions
    detections = []
    for region in measure.regionprops(labels, intensity_image=image_norm):
        # Skip background
        if region.label == 0:
            continue
        
        y, x = region.centroid
        area = region.area
        radius = np.sqrt(area / np.pi)
        confidence = region.mean_intensity
        
        detections.append(Detection(
            x=float(x),
            y=float(y),
            radius=float(radius),
            confidence=float(confidence)
        ))
    
    return labels, detections


def adaptive_threshold_segmentation(
    image: np.ndarray,
    block_size: int = 11,
    C: float = 2,
    morphology_cleanup: bool = True
) -> Tuple[np.ndarray, List[Detection]]:
    """
    Segment using adaptive thresholding.
    
    Useful for images with varying illumination.
    
    Args:
        image: Input image (grayscale).
        block_size: Size of neighborhood for adaptive threshold (odd number).
        C: Constant subtracted from weighted mean.
        morphology_cleanup: Whether to apply morphological operations.
    
    Returns:
        Tuple of (binary_mask, detections).
    """
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        -C  # Negative because we want bright objects
    )
    
    # Morphological cleanup
    if morphology_cleanup:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(binary)
    
    # Extract detections
    detections = []
    for region in measure.regionprops(labels):
        if region.label == 0:  # Skip background
            continue
        
        y, x = region.centroid
        area = region.area
        radius = np.sqrt(area / np.pi)
        
        # Simple confidence based on circularity
        perimeter = region.perimeter
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        confidence = float(circularity)
        
        detections.append(Detection(
            x=float(x),
            y=float(y),
            radius=float(radius),
            confidence=confidence
        ))
    
    return binary, detections


def contour_based_detection(
    image: np.ndarray,
    threshold: int = 127,
    min_area: int = 10,
    max_area: int = 200,
    min_circularity: float = 0.5
) -> List[Detection]:
    """
    Detect sperm using contour analysis.
    
    Args:
        image: Input image (grayscale).
        threshold: Binary threshold value.
        min_area: Minimum contour area.
        max_area: Maximum contour area.
        min_circularity: Minimum circularity (4π*area/perimeter²).
    
    Returns:
        List of Detection objects.
    """
    # Threshold
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if not (min_area <= area <= max_area):
            continue
        
        # Compute circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        
        # Filter by circularity
        if circularity < min_circularity:
            continue
        
        # Get centroid and equivalent radius
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        radius = np.sqrt(area / np.pi)
        
        detections.append(Detection(
            x=float(cx),
            y=float(cy),
            radius=float(radius),
            confidence=float(circularity)
        ))
    
    return detections


class MultiMethodDetector:
    """
    Combines multiple detection methods with voting or ensemble.
    """
    
    def __init__(
        self,
        methods: List[str] = ['blob', 'watershed'],
        consensus_threshold: int = 1,
        nms_threshold: float = 0.5
    ):
        """
        Initialize multi-method detector.
        
        Args:
            methods: List of methods to use ('blob', 'watershed', 'adaptive', 'contour').
            consensus_threshold: Minimum number of methods that must detect an object.
            nms_threshold: NMS IoU threshold.
        """
        self.methods = methods
        self.consensus_threshold = consensus_threshold
        self.nms_threshold = nms_threshold
    
    def detect(self, image: np.ndarray, **kwargs) -> List[Detection]:
        """
        Detect using multiple methods and combine results.
        
        Args:
            image: Input image.
            **kwargs: Parameters for individual methods.
        
        Returns:
            Combined list of detections.
        """
        all_detections = []
        
        for method in self.methods:
            if method == 'blob':
                from .blob_detector import BlobDetector
                detector = BlobDetector(**kwargs.get('blob_params', {}))
                dets = detector.detect(image)
            
            elif method == 'watershed':
                _, dets = watershed_segmentation(image, **kwargs.get('watershed_params', {}))
            
            elif method == 'adaptive':
                _, dets = adaptive_threshold_segmentation(image, **kwargs.get('adaptive_params', {}))
            
            elif method == 'contour':
                dets = contour_based_detection(image, **kwargs.get('contour_params', {}))
            
            else:
                continue
            
            all_detections.extend(dets)
        
        # Apply NMS to combined detections
        from .blob_detector import non_maximum_suppression
        final_detections = non_maximum_suppression(all_detections, self.nms_threshold)
        
        return final_detections
