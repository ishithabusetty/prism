"""
Visualization Utilities
Helpers for visualizing detection results.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


class Visualizer:
    """
    Utility class for visualizing detection results.
    """
    
    # Color palette for different detection types
    COLORS = {
        'weapon': (255, 0, 0),      # Red
        'text': (0, 255, 0),         # Green
        'logo': (0, 0, 255),         # Blue
        'competitor': (255, 165, 0), # Orange
        'default': (128, 128, 128)   # Gray
    }
    
    def __init__(self, font_scale: float = 0.6, thickness: int = 2):
        """
        Initialize visualizer.
        
        Args:
            font_scale: Font scale for labels
            thickness: Bounding box thickness
        """
        self.font_scale = font_scale
        self.thickness = thickness
    
    def draw_detection(
        self,
        image: np.ndarray,
        bbox: Dict[str, int],
        label: str,
        confidence: float,
        color: Tuple[int, int, int] = None,
        detection_type: str = 'default'
    ) -> np.ndarray:
        """
        Draw a single detection on image.
        
        Args:
            image: Input image (BGR)
            bbox: Bounding box with x1, y1, x2, y2
            label: Label text
            confidence: Detection confidence
            color: Custom color (B, G, R)
            detection_type: Type for color selection
            
        Returns:
            Image with drawn detection
        """
        if cv2 is None:
            raise ImportError("OpenCV required for visualization")
        
        img = image.copy()
        
        # Get color
        if color is None:
            color = self.COLORS.get(detection_type, self.COLORS['default'])
        
        # Draw bounding box
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.thickness)
        
        # Draw label background
        text = f"{label} ({confidence:.0%})"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(
            img, text, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1
        )
        
        return img
    
    def visualize_results(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        show_weapons: bool = True,
        show_text: bool = True,
        show_logos: bool = True
    ) -> np.ndarray:
        """
        Visualize all detection results on image.
        
        Args:
            image: Input image (BGR)
            results: Detection results from unified pipeline
            show_weapons: Show weapon detections
            show_text: Show text regions
            show_logos: Show logo detections
            
        Returns:
            Image with all detections drawn
        """
        img = image.copy()
        
        detailed = results.get('detailed_results', {})
        
        # Draw weapon detections
        if show_weapons:
            weapon_results = detailed.get('weapon_detection', {})
            for det in weapon_results.get('detections', []):
                img = self.draw_detection(
                    img, det['bbox'], det['class'],
                    det['confidence'], detection_type='weapon'
                )
        
        # Draw logo detections
        if show_logos:
            logo_results = detailed.get('logo_detection', {})
            for det in logo_results.get('detections', []):
                det_type = 'competitor' if det.get('is_competitor') else 'logo'
                img = self.draw_detection(
                    img, det['bbox'], det['brand'],
                    det['confidence'], detection_type=det_type
                )
        
        # Add status overlay
        status = results.get('status', 'UNKNOWN')
        color = (0, 255, 0) if status == 'SAFE' else (0, 0, 255)
        
        # Draw status bar
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 40), color, -1)
        cv2.putText(
            img, f"STATUS: {status}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        
        return img
    
    def create_summary_image(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        output_path: str = None
    ) -> np.ndarray:
        """
        Create a summary image with detection results.
        
        Args:
            image: Input image
            results: Detection results
            output_path: Optional path to save image
            
        Returns:
            Summary image
        """
        # Visualize detections
        vis_img = self.visualize_results(image, results)
        
        # Add summary panel
        h, w = vis_img.shape[:2]
        panel_height = 100
        summary = np.ones((panel_height, w, 3), dtype=np.uint8) * 255
        
        # Add summary text
        y_offset = 25
        flags = results.get('flags', [])
        
        cv2.putText(
            summary, f"Summary: {results.get('summary', '')[:60]}...",
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
        
        y_offset += 25
        if flags:
            for i, flag in enumerate(flags[:3]):
                text = f"Flag {i+1}: {flag['type']} (conf: {flag['confidence']:.2%})"
                cv2.putText(
                    summary, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1
                )
                y_offset += 20
        
        # Combine
        combined = np.vstack([vis_img, summary])
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, combined)
            print(f"✓ Saved to {output_path}")
        
        return combined
    
    def plot_metrics(
        self,
        metrics: Dict[str, float],
        title: str = "Model Metrics",
        output_path: str = None
    ):
        """
        Plot evaluation metrics as bar chart.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Chart title
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib required for plotting")
            return
        
        names = list(metrics.keys())
        values = list(metrics.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, values, color='steelblue')
        
        # Add value labels
        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom'
            )
        
        plt.title(title)
        plt.ylabel('Value')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {output_path}")
        
        plt.close()


def create_visualizer(font_scale: float = 0.6, thickness: int = 2) -> Visualizer:
    """Factory function to create Visualizer."""
    return Visualizer(font_scale=font_scale, thickness=thickness)
