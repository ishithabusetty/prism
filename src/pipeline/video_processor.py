"""
Video Processing Module
Extracts frames from video for analysis.
"""

import os
from typing import List, Generator, Optional, Tuple
import numpy as np
from pathlib import Path


class VideoProcessor:
    """
    Video frame extraction for processing.
    Uses OpenCV for frame extraction.
    """
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def __init__(
        self,
        frame_interval: int = 30,
        max_frames: int = 100
    ):
        """
        Initialize video processor.
        
        Args:
            frame_interval: Extract every Nth frame
            max_frames: Maximum frames to extract per video
        """
        self.frame_interval = frame_interval
        self.max_frames = max_frames
    
    def is_supported(self, video_path: str) -> bool:
        """Check if video format is supported."""
        ext = Path(video_path).suffix.lower()
        return ext in self.SUPPORTED_FORMATS
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video info
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'path': video_path,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration_seconds': 0
        }
        
        if info['fps'] > 0:
            info['duration_seconds'] = info['total_frames'] / info['fps']
        
        cap.release()
        return info
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        save_frames: bool = False
    ) -> List[np.ndarray]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames (if save_frames=True)
            save_frames: Whether to save frames as images
            
        Returns:
            List of frames as numpy arrays (BGR format)
        """
        import cv2
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if not self.is_supported(video_path):
            raise ValueError(f"Unsupported video format: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        extracted_count = 0
        
        if save_frames and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        while cap.isOpened() and extracted_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_interval == 0:
                frames.append(frame)
                
                if save_frames and output_dir:
                    frame_path = os.path.join(
                        output_dir,
                        f"frame_{extracted_count:04d}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    def extract_frames_generator(
        self,
        video_path: str
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames one at a time.
        Memory efficient for large videos.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Tuple of (frame_index, frame_array)
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_interval == 0:
                yield (extracted_count, frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
    
    def extract_keyframes(
        self,
        video_path: str,
        threshold: float = 30.0
    ) -> List[np.ndarray]:
        """
        Extract keyframes based on scene changes.
        
        Args:
            video_path: Path to video file
            threshold: Scene change threshold (higher = less sensitive)
            
        Returns:
            List of keyframes
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        keyframes = []
        prev_frame = None
        frame_count = 0
        
        while cap.isOpened() and len(keyframes) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is None:
                keyframes.append(frame)
                prev_frame = gray
                frame_count += 1
                continue
            
            # Calculate difference
            diff = cv2.absdiff(prev_frame, gray)
            mean_diff = np.mean(diff)
            
            if mean_diff > threshold:
                keyframes.append(frame)
                prev_frame = gray
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(keyframes)} keyframes from video")
        return keyframes


def create_processor(
    frame_interval: int = 30,
    max_frames: int = 100
) -> VideoProcessor:
    """
    Factory function to create video processor.
    
    Args:
        frame_interval: Extract every Nth frame
        max_frames: Maximum frames to extract
        
    Returns:
        Configured VideoProcessor instance
    """
    return VideoProcessor(
        frame_interval=frame_interval,
        max_frames=max_frames
    )
