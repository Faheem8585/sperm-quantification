"""Detection module for sperm identification and segmentation."""

from .blob_detector import (
    Detection,
    BlobDetector,
    non_maximum_suppression
)
from .segmentation import (
    watershed_segmentation,
    adaptive_threshold_segmentation,
    contour_based_detection,
    MultiMethodDetector
)

__all__ = [
    'Detection',
    'BlobDetector',
    'non_maximum_suppression',
    'watershed_segmentation',
    'adaptive_threshold_segmentation',
    'contour_based_detection',
    'MultiMethodDetector'
]
