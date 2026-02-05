"""
Logo Detection Module
Uses YOLOv8 for detecting brand logos (Samsung competitors).
"""

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import yaml


class LogoDetector:
    """
    YOLOv8-based logo detector for competitor brands.
    
    Detects: Apple, Google, Huawei, Xiaomi, OnePlus, Oppo, Vivo, Sony, LG
    """
    
    # Competitor brands (flag as UNSAFE if detected)
    COMPETITOR_BRANDS = [
        'apple', 'google', 'huawei', 'xiaomi',
        'oneplus', 'oppo', 'vivo', 'sony', 'lg'
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: str = "config/logo_classes.yaml",
        confidence_threshold: float = 0.6,
        device: str = "cuda"
    ):
        """
        Initialize the logo detector.
        
        Args:
            model_path: Path to trained YOLOv8 weights
            config_path: Path to logo classes configuration
            confidence_threshold: Minimum confidence for detections
            device: 'cuda' or 'cpu'
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.classes = {}
        
        # Load configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.classes = config.get('classes', {})
                self.confidence_threshold = config.get('thresholds', {}).get(
                    'confidence_min', confidence_threshold
                )
                # Update competitor list from config if available
                flag_brands = config.get('flag_as_competitor', [])
                if flag_brands:
                    self.COMPETITOR_BRANDS = flag_brands
        
        # Load model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load YOLOv8 model from weights file."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✓ Logo detector model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model = None
    
    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Detect logos in an image.
        
        Args:
            image: Image path, numpy array, or PIL Image
            return_visualization: Whether to return annotated image
            
        Returns:
            Dictionary with detection results including competitor flags
        """
        if self.model is None:
            return {
                'detected': False,
                'competitor_detected': False,
                'detections': [],
                'error': 'Model not loaded'
            }
        
        # Run inference
        results = self.model(image, verbose=False)[0]
        
        detections = []
        competitor_detections = []
        samsung_detected = False
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= self.confidence_threshold:
                cls_id = int(box.cls[0])
                cls_name = results.names.get(cls_id, f"class_{cls_id}").lower()
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                detection = {
                    'brand': cls_name,
                    'class_id': cls_id,
                    'confidence': round(conf, 4),
                    'is_competitor': cls_name in self.COMPETITOR_BRANDS,
                    'bbox': {
                        'x1': int(bbox[0]),
                        'y1': int(bbox[1]),
                        'x2': int(bbox[2]),
                        'y2': int(bbox[3])
                    }
                }
                
                detections.append(detection)
                
                if cls_name in self.COMPETITOR_BRANDS:
                    competitor_detections.append(detection)
                elif cls_name == 'samsung':
                    samsung_detected = True
        
        result = {
            'detected': len(detections) > 0,
            'competitor_detected': len(competitor_detections) > 0,
            'samsung_detected': samsung_detected,
            'detections': detections,
            'competitor_detections': competitor_detections,
            'detection_count': len(detections),
            'competitor_count': len(competitor_detections)
        }
        
        if return_visualization:
            result['visualization'] = results.plot()
        
        return result
    
    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Detect logos in multiple images.
        
        Args:
            images: List of image paths or arrays
            batch_size: Batch size for inference
            
        Returns:
            List of detection results
        """
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                all_results.append(self.detect(img))
        
        return all_results
    
    def get_summary(self, detection_result: Dict[str, Any]) -> str:
        """
        Get human-readable summary of detection result.
        
        Args:
            detection_result: Result from detect() method
            
        Returns:
            Summary string
        """
        if not detection_result['detected']:
            return "No logos detected"
        
        brands_found = [d['brand'] for d in detection_result['detections']]
        competitors = [d['brand'] for d in detection_result.get('competitor_detections', [])]
        
        summary = f"Detected logos: {', '.join(brands_found)}"
        if competitors:
            summary += f" | COMPETITORS: {', '.join(competitors)}"
        
        return summary
    
    def is_competitor_content(self, detection_result: Dict[str, Any]) -> Tuple:
        """
        Check if image contains competitor content.
        
        Args:
            detection_result: Result from detect() method
            
        Returns:
            Tuple of (is_competitor: bool, brands: List[str])
        """
        if detection_result.get('competitor_detected', False):
            brands = [d['brand'] for d in detection_result.get('competitor_detections', [])]
            return True, brands
        return False, []


# Import for type hints
from typing import Tuple


def create_detector(
    model_path: str = None,
    device: str = "cuda"
) -> LogoDetector:
    """
    Factory function to create a logo detector.
    
    Args:
        model_path: Path to trained model weights
        device: 'cuda' or 'cpu'
        
    Returns:
        Configured LogoDetector instance
    """
    return LogoDetector(
        model_path=model_path,
        device=device
    )
