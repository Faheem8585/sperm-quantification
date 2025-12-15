"""
Blob detection for sperm head identification.

Uses Laplacian of Gaussian and Difference of Gaussian methods
optimized for sperm morphology.
"""

import numpy as np
import cv2
from skimage.feature import blob_dog, blob_log, blob_doh
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection result."""
    x: float  # X coordinate (pixels)
    y: float  # Y coordinate (pixels)
    radius: float  # Blob radius (pixels)
    confidence: float  # Detection confidence score (0-1)
    
    def to_bbox(self) -> Tuple[int, int, int, int]:
        """
        Convert to bounding box (x1, y1, x2, y2).
        
        Returns:
            Tuple of (x1, y1, x2, y2).
        """
        x1 = int(self.x - self.radius)
        y1 = int(self.y - self.radius)
        x2 = int(self.x + self.radius)
        y2 = int(self.y + self.radius)
        return (x1, y1, x2, y2)


class BlobDetector:
    """
    Blob detection for sperm heads.
    
    Uses scale-space blob detection to identify sperm heads,
    which appear as bright circular objects in microscopy.
    """
    
    def __init__(
        self,
        method: str = 'dog',
        min_sigma: float = 1.5,
        max_sigma: float = 3.0,
        num_sigma: int = 10,
        threshold: float = 0.1,
        overlap: float = 0.5,
        min_area: int = 10,
        max_area: int = 200
    ):
        """
        Initialize blob detector.
        
        Args:
            method: Detection method ('dog', 'log', or 'doh').
                   'dog' = Difference of Gaussians (fastest)
                   'log' = Laplacian of Gaussian (most accurate)
                   'doh' = Determinant of Hessian (blob enhancement)
            min_sigma: Minimum standard deviation for Gaussian kernel.
            max_sigma: Maximum standard deviation for Gaussian kernel.
            num_sigma: Number of scales between min and max sigma.
            threshold: Detection threshold (higher = fewer detections).
            overlap: Maximum allowed overlap between blobs (0-1).
            min_area: Minimum blob area in pixels.
            max_area: Maximum blob area in pixels.
        """
        self.method = method
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.min_area = min_area
        self.max_area = max_area
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect blobs in image.
        
        Args:
            image: Input image (grayscale, uint8 or float).
        
        Returns:
            List of Detection objects.
        """
        # Normalize image to [0, 1] for blob detection
        if image.dtype == np.uint8:
            image_norm = image.astype(float) / 255.0
        else:
            image_norm = image
        
        # Apply blob detection
        if self.method == 'dog':
            blobs = blob_dog(
                image_norm,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                threshold=self.threshold,
                overlap=self.overlap
            )
        elif self.method == 'log':
            blobs = blob_log(
                image_norm,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_sigma=self.num_sigma,
                threshold=self.threshold,
                overlap=self.overlap
            )
        elif self.method == 'doh':
            blobs = blob_doh(
                image_norm,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_sigma=self.num_sigma,
                threshold=self.threshold,
                overlap=self.overlap
            )
        else:
            raise ValueError(f"Unknown blob detection method: {self.method}")
        
        # Convert to Detection objects and filter by area
        detections = []
        for blob in blobs:
            y, x, sigma = blob
            radius = sigma * np.sqrt(2)  # Approximate radius
            area = np.pi * radius**2
            
            # Filter by area
            if self.min_area <= area <= self.max_area:
                # Compute confidence (normalized intensity at blob center)
                confidence = self._compute_confidence(image_norm, x, y, radius)
                
                detections.append(Detection(
                    x=float(x),
                    y=float(y),
                    radius=float(radius),
                    confidence=confidence
                ))
        
        return detections
    
    def _compute_confidence(
        self,
        image: np.ndarray,
        x: float,
        y: float,
        radius: float
    ) -> float:
        """
        Compute detection confidence based on local intensity.
        
        Args:
            image: Normalized image.
            x, y: Blob center.
            radius: Blob radius.
        
        Returns:
            Confidence score (0-1).
        """
        # Sample intensity in circular region
        y_int, x_int = int(y), int(x)
        h, w = image.shape
        
        # Bounds check
        if not (0 <= x_int < w and 0 <= y_int < h):
            return 0.0
        
        # Extract patch
        r = int(radius) + 1
        y1 = max(0, y_int - r)
        y2 = min(h, y_int + r + 1)
        x1 = max(0, x_int - r)
        x2 = min(w, x_int + r + 1)
        
        patch = image[y1:y2, x1:x2]
        
        # Confidence = mean intensity in patch
        confidence = float(np.mean(patch))
        
        return confidence
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detections on image for visualization.
        
        Args:
            image: Input image (grayscale or color).
            detections: List of detections.
            color: Circle color (BGR).
            thickness: Line thickness.
        
        Returns:
            Image with drawn detections.
        """
        # Convert grayscale to BGR if needed
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        for det in detections:
            center = (int(det.x), int(det.y))
            radius = int(det.radius)
            cv2.circle(vis_image, center, radius, color, thickness)
            
            # Draw confidence score
            text = f"{det.confidence:.2f}"
            cv2.putText(
                vis_image, text, (int(det.x) + radius, int(det.y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        return vis_image


def non_maximum_suppression(
    detections: List[Detection],
    iou_threshold: float = 0.5
) -> List[Detection]:
    """
    Apply non-maximum suppression to remove overlapping detections.
    
    Args:
        detections: List of detections.
        iou_threshold: IoU threshold for suppression.
    
    Returns:
        Filtered list of detections.
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    # NMS
    kept = []
    while len(detections) > 0:
        current = detections.pop(0)
        kept.append(current)
        
        # Remove detections with high overlap
        detections = [
            det for det in detections
            if _compute_iou_circles(current, det) < iou_threshold
        ]
    
    return kept


def _compute_iou_circles(det1: Detection, det2: Detection) -> float:
    """
    Compute IoU (Intersection over Union) between two circular detections.
    
    Args:
        det1, det2: Detection objects.
    
    Returns:
        IoU value (0-1).
    """
    # Distance between centers
    dx = det1.x - det2.x
    dy = det1.y - det2.y
    dist = np.sqrt(dx**2 + dy**2)
    
    r1 = det1.radius
    r2 = det2.radius
    
    # No overlap
    if dist >= r1 + r2:
        return 0.0
    
    # One circle inside the other
    if dist <= abs(r1 - r2):
        smaller_area = np.pi * min(r1, r2)**2
        larger_area = np.pi * max(r1, r2)**2
        return smaller_area / larger_area
    
    # Partial overlap - use circle intersection formula
    # Approximate with bounding box IoU for simplicity
    x1_min = det1.x - r1
    y1_min = det1.y - r1
    x1_max = det1.x + r1
    y1_max = det1.y + r1
    
    x2_min = det2.x - r2
    y2_min = det2.y - r2
    x2_max = det2.x + r2
    y2_max = det2.y + r2
    
    # Intersection box
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)
    
    area1 = 4 * r1**2
    area2 = 4 * r2**2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0
