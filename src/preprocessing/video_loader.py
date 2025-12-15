"""
Video loading and I/O utilities for microscopy data.

Supports common video formats and TIFF stacks.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Iterator, Union
import imageio


class VideoReader:
    """
    Memory-efficient video reader with frame-by-frame iteration.
    
    Supports:
    - Video files (AVI, MP4, MOV via OpenCV)
    - TIFF stacks
    - Image sequences
    """
    
    def __init__(
        self,
        video_path: Union[str, Path],
        grayscale: bool = True,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file or image sequence.
            grayscale: Whether to convert frames to grayscale.
            start_frame: First frame to read (0-indexed).
            end_frame: Last frame to read (exclusive). None = read all.
        """
        self.video_path = Path(video_path)
        self.grayscale = grayscale
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        # Determine file type
        if self.video_path.suffix.lower() in ['.tif', '.tiff']:
            self._reader_type = 'tiff'
            self._open_tiff()
        else:
            self._reader_type = 'opencv'
            self._open_opencv()
        
        # Navigate to start frame
        if start_frame > 0:
            self.seek(start_frame)
    
    def _open_opencv(self):
        """Open video file using OpenCV."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Extract metadata
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.end_frame is None:
            self.end_frame = self.total_frames
        
        self.current_frame = 0
    
    def _open_tiff(self):
        """Open TIFF stack using imageio."""
        self.tiff_reader = imageio.get_reader(str(self.video_path))
        
        # Get metadata
        self.total_frames = len(self.tiff_reader)
        
        # Read first frame to get dimensions
        first_frame = self.tiff_reader.get_data(0)
        if len(first_frame.shape) == 3 and self.grayscale:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
        
        self.height, self.width = first_frame.shape[:2]
        
        # FPS not typically stored in TIFF, use default
        self.fps = 30.0
        
        if self.end_frame is None:
            self.end_frame = self.total_frames
        
        self.current_frame = 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.
        
        Returns:
            Tuple of (success, frame).
            success is True if frame was read successfully.
        """
        if self.current_frame >= self.end_frame:
            return False, None
        
        if self._reader_type == 'opencv':
            ret, frame = self.cap.read()
            if not ret:
                return False, None
            
            if self.grayscale and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        elif self._reader_type == 'tiff':
            try:
                frame = self.tiff_reader.get_data(self.current_frame)
                if self.grayscale and len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            except IndexError:
                return False, None
        
        self.current_frame += 1
        return True, frame
    
    def seek(self, frame_number: int):
        """
        Seek to specific frame.
        
        Args:
            frame_number: Frame index to seek to.
        """
        if self._reader_type == 'opencv':
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        self.current_frame = frame_number
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames."""
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame
    
    def __len__(self) -> int:
        """Get number of frames to be read."""
        return self.end_frame - self.start_frame
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def release(self):
        """Release video resources."""
        if self._reader_type == 'opencv':
            self.cap.release()
        elif self._reader_type == 'tiff':
            self.tiff_reader.close()
    
    def get_metadata(self) -> dict:
        """
        Get video metadata.
        
        Returns:
            Dictionary with video properties.
        """
        return {
            'path': str(self.video_path),
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'duration_s': self.total_frames / self.fps if self.fps > 0 else 0
        }


def load_video_array(
    video_path: Union[str, Path],
    grayscale: bool = True,
    start_frame: int = 0,
    end_frame: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Load entire video into memory as numpy array.
    
    Warning: May use large amounts of memory for long videos.
    
    Args:
        video_path: Path to video file.
        grayscale: Whether to convert to grayscale.
        start_frame: First frame to load.
        end_frame: Last frame to load (exclusive).
    
    Returns:
        Tuple of (video_array, metadata).
        video_array has shape (n_frames, height, width) for grayscale
        or (n_frames, height, width, 3) for color.
    """
    with VideoReader(video_path, grayscale, start_frame, end_frame) as reader:
        frames = []
        for frame in reader:
            frames.append(frame)
        
        video_array = np.stack(frames, axis=0)
        metadata = reader.get_metadata()
    
    return video_array, metadata
