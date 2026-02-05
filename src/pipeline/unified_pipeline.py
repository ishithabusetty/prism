"""
Unified Detection Pipeline
Combines all detectors for comprehensive harmful content detection.
"""

import os
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
from pathlib import Path

from ..detectors.weapon_detector import WeaponDetector
from ..detectors.text_detector import TextDetector
from ..detectors.logo_detector import LogoDetector
from .decision_engine import DecisionEngine, DecisionResult, SafetyStatus
from .video_processor import VideoProcessor


class UnifiedPipeline:
    """
    Main inference pipeline that runs all detectors and produces final verdict.
    
    Pipeline Flow:
    1. Input image/video
    2. Run weapon detection
    3. Run OCR + NLP text classification
    4. Run logo detection
    5. Aggregate results via Decision Engine
    6. Output SAFE/UNSAFE with explanation
    """
    
    def __init__(
        self,
        weapon_model_path: Optional[str] = None,
        logo_model_path: Optional[str] = None,
        nlp_model_path: Optional[str] = None,
        device: str = "cuda",
        config_dir: str = "config"
    ):
        """
        Initialize unified pipeline with all detectors.
        
        Args:
            weapon_model_path: Path to trained weapon YOLOv8 model
            logo_model_path: Path to trained logo YOLOv8 model
            nlp_model_path: Path to trained NLP model
            device: 'cuda' or 'cpu'
            config_dir: Directory containing config files
        """
        self.device = device
        self.config_dir = config_dir
        
        # Initialize detectors
        print("Initializing Unified Detection Pipeline...")
        
        self.weapon_detector = WeaponDetector(
            model_path=weapon_model_path,
            config_path=os.path.join(config_dir, "weapon_classes.yaml"),
            device=device
        )
        
        self.text_detector = TextDetector(
            nlp_model_path=nlp_model_path,
            config_path=os.path.join(config_dir, "text_classification.yaml"),
            device=device
        )
        
        self.logo_detector = LogoDetector(
            model_path=logo_model_path,
            config_path=os.path.join(config_dir, "logo_classes.yaml"),
            device=device
        )
        
        # Initialize decision engine
        self.decision_engine = DecisionEngine()
        
        # Initialize video processor
        self.video_processor = VideoProcessor()
        
        print("✓ Pipeline initialized successfully")
    
    def process_image(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_visualizations: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single image through all detection stages.
        
        Args:
            image: Image path, numpy array, or PIL Image
            return_visualizations: Include annotated images in output
            
        Returns:
            Complete detection results with SAFE/UNSAFE verdict
        """
        # Load image if path
        if isinstance(image, str):
            image_path = image
            image = np.array(Image.open(image))
        else:
            image_path = None
        
        # Stage 1: Weapon Detection
        weapon_results = self.weapon_detector.detect(
            image,
            return_visualization=return_visualizations
        )
        
        # Stage 2: Text Detection + Classification
        text_results = self.text_detector.detect(image)
        
        # Stage 3: Logo Detection
        logo_results = self.logo_detector.detect(
            image,
            return_visualization=return_visualizations
        )
        
        # Stage 4: Decision Engine
        decision = self.decision_engine.evaluate(
            weapon_results=weapon_results,
            text_results=text_results,
            logo_results=logo_results
        )
        
        # Build final output
        output = {
            'status': decision.status.value,
            'is_safe': decision.status == SafetyStatus.SAFE,
            'summary': decision.summary,
            'input': {
                'type': 'image',
                'path': image_path
            },
            'detections': {
                'weapons': {
                    'found': weapon_results.get('detected', False),
                    'count': weapon_results.get('detection_count', 0),
                    'items': weapon_results.get('detections', [])
                },
                'text': {
                    'found': text_results.get('text_found', False),
                    'content': text_results.get('combined_text', ''),
                    'classification': text_results.get('classification', {})
                },
                'logos': {
                    'found': logo_results.get('detected', False),
                    'competitor_found': logo_results.get('competitor_detected', False),
                    'brands': [d['brand'] for d in logo_results.get('detections', [])]
                }
            },
            'flags': [
                {
                    'reason': f.reason.value,
                    'confidence': f.confidence,
                    'priority': f.priority
                }
                for f in decision.flags
            ],
            'explanation': self.decision_engine.get_detailed_explanation(decision)
        }
        
        if return_visualizations:
            output['visualizations'] = {
                'weapons': weapon_results.get('visualization'),
                'logos': logo_results.get('visualization')
            }
        
        return output
    
    def process_video(
        self,
        video_path: str,
        frame_interval: int = 30,
        max_frames: int = 50,
        aggregate_mode: str = "any"
    ) -> Dict[str, Any]:
        """
        Process video by analyzing frames.
        
        Args:
            video_path: Path to video file
            frame_interval: Process every Nth frame
            max_frames: Maximum frames to analyze
            aggregate_mode: 'any' (unsafe if any frame unsafe) or 
                          'majority' (unsafe if >50% frames unsafe)
            
        Returns:
            Aggregated detection results for video
        """
        # Configure video processor
        self.video_processor.frame_interval = frame_interval
        self.video_processor.max_frames = max_frames
        
        # Get video info
        video_info = self.video_processor.get_video_info(video_path)
        
        # Extract and process frames
        frame_results = []
        unsafe_frames = []
        
        for frame_idx, frame in self.video_processor.extract_frames_generator(video_path):
            result = self.process_image(frame)
            result['frame_index'] = frame_idx
            frame_results.append(result)
            
            if not result['is_safe']:
                unsafe_frames.append({
                    'frame_index': frame_idx,
                    'flags': result['flags']
                })
        
        # Aggregate results
        total_frames = len(frame_results)
        unsafe_count = len(unsafe_frames)
        
        if aggregate_mode == "any":
            is_safe = unsafe_count == 0
        else:  # majority
            is_safe = unsafe_count < total_frames / 2
        
        # Collect all unique flags
        all_flags = {}
        for uf in unsafe_frames:
            for flag in uf['flags']:
                reason = flag['reason']
                if reason not in all_flags or flag['confidence'] > all_flags[reason]['confidence']:
                    all_flags[reason] = flag
        
        status = "SAFE" if is_safe else "UNSAFE"
        
        if is_safe:
            summary = f"Video analyzed ({total_frames} frames). No harmful content detected."
        else:
            summary = f"Video UNSAFE: {unsafe_count}/{total_frames} frames contain harmful content."
        
        return {
            'status': status,
            'is_safe': is_safe,
            'summary': summary,
            'input': {
                'type': 'video',
                'path': video_path,
                'info': video_info
            },
            'analysis': {
                'total_frames_analyzed': total_frames,
                'unsafe_frames_count': unsafe_count,
                'unsafe_frame_indices': [uf['frame_index'] for uf in unsafe_frames],
                'aggregate_mode': aggregate_mode
            },
            'flags': list(all_flags.values()),
            'frame_results': frame_results  # Detailed per-frame results
        }
    
    def process(
        self,
        input_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Auto-detect input type and process accordingly.
        
        Args:
            input_path: Path to image or video file
            **kwargs: Additional arguments for processing
            
        Returns:
            Detection results
        """
        ext = Path(input_path).suffix.lower()
        
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if ext in image_exts:
            return self.process_image(input_path, **kwargs)
        elif ext in video_exts:
            return self.process_video(input_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def get_status_emoji(self, result: Dict[str, Any]) -> str:
        """Get emoji for status display."""
        return "✅" if result['is_safe'] else "🚨"
    
    def print_result(self, result: Dict[str, Any]):
        """Pretty print detection result."""
        emoji = self.get_status_emoji(result)
        print(f"\n{emoji} STATUS: {result['status']}")
        print(f"{'─' * 50}")
        print(result['summary'])
        
        if not result['is_safe']:
            print(f"\n⚠️  Issues detected:")
            for flag in result['flags']:
                print(f"   • {flag['reason']} (confidence: {flag['confidence']:.2%})")


def create_pipeline(
    weapon_model_path: str = None,
    logo_model_path: str = None,
    nlp_model_path: str = None,
    device: str = "cuda"
) -> UnifiedPipeline:
    """
    Factory function to create unified pipeline.
    
    Args:
        weapon_model_path: Path to weapon model
        logo_model_path: Path to logo model
        nlp_model_path: Path to NLP model
        device: 'cuda' or 'cpu'
        
    Returns:
        Configured UnifiedPipeline instance
    """
    return UnifiedPipeline(
        weapon_model_path=weapon_model_path,
        logo_model_path=logo_model_path,
        nlp_model_path=nlp_model_path,
        device=device
    )
