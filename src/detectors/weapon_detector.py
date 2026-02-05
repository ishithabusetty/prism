"""
Weapon Detection Module
Uses YOLOv8 for detecting weapons and dangerous objects in images.
"""

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import yaml


class WeaponDetector:
    """
    YOLOv8-based weapon and dangerous object detector.
    
    Detects: guns, knives, machetes, bats, alcohol bottles, swords, etc.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: str = "config/weapon_classes.yaml",
        confidence_threshold: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize the weapon detector.
        
        Args:
            model_path: Path to trained YOLOv8 weights (.pt file)
            config_path: Path to weapon classes configuration
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
        
        # Load model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load YOLOv8 model from weights file."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✓ Weapon detector model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.model = None
    
    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Detect weapons in an image.
        
        Args:
            image: Image path, numpy array, or PIL Image
            return_visualization: Whether to return annotated image
            
        Returns:
            Dictionary with detection results:
            {
                'detected': bool,
                'detections': List[{class, confidence, bbox}],
                'visualization': Optional[np.ndarray]
            }
        """
        if self.model is None:
            return {
                'detected': False,
                'detections': [],
                'error': 'Model not loaded'
            }
        
        # Run inference
        results = self.model(image, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= self.confidence_threshold:
                cls_id = int(box.cls[0])
                cls_name = results.names.get(cls_id, f"class_{cls_id}")
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                detections.append({
                    'class': cls_name,
                    'class_id': cls_id,
                    'confidence': round(conf, 4),
                    'bbox': {
                        'x1': int(bbox[0]),
                        'y1': int(bbox[1]),
                        'x2': int(bbox[2]),
                        'y2': int(bbox[3])
                    }
                })
        
        result = {
            'detected': len(detections) > 0,
            'detections': detections,
            'detection_count': len(detections)
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
        Detect weapons in multiple images.
        
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
            return "No weapons detected"
        
        classes_found = [d['class'] for d in detection_result['detections']]
        class_counts = {}
        for cls in classes_found:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        summary_parts = [f"{count} {cls}" for cls, count in class_counts.items()]
        return f"Detected: {', '.join(summary_parts)}"


# For Colab/standalone usage
def create_detector(model_path: str = None, device: str = "cuda") -> WeaponDetector:
    """
    Factory function to create a weapon detector.
    
    Args:
        model_path: Path to trained model weights
        device: 'cuda' or 'cpu'
        
    Returns:
        Configured WeaponDetector instance
    """
    return WeaponDetector(
        model_path=model_path,
        device=device
    )
