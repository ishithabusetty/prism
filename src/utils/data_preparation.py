"""
Data Preparation Utilities
Helpers for dataset curation, annotation conversion, and augmentation.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml


class DatasetPreparer:
    """
    Utility class for preparing datasets for YOLOv8 training.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize dataset preparer.
        
        Args:
            data_dir: Root directory for dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_yolo_structure(self):
        """Create standard YOLOv8 folder structure."""
        for split in ['train', 'valid', 'test']:
            (self.data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created YOLO structure in {self.data_dir}")
    
    def create_data_yaml(
        self,
        classes: Dict[int, str],
        output_path: str = None
    ) -> str:
        """
        Create data.yaml configuration file for YOLOv8.
        
        Args:
            classes: Dictionary mapping class_id to class_name
            output_path: Path to save data.yaml
            
        Returns:
            Path to created file
        """
        yaml_content = {
            'path': str(self.data_dir),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': classes,
            'nc': len(classes)
        }
        
        if output_path is None:
            output_path = self.data_dir / 'data.yaml'
        
        with open(output_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✓ Created data.yaml with {len(classes)} classes")
        return str(output_path)
    
    def split_dataset(
        self,
        images_dir: str,
        labels_dir: str,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.2,
        test_ratio: float = 0.1
    ):
        """
        Split dataset into train/valid/test sets.
        
        Args:
            images_dir: Source directory with images
            labels_dir: Source directory with labels
            train_ratio: Ratio for training set
            valid_ratio: Ratio for validation set
            test_ratio: Ratio for test set
        """
        import random
        
        images = list(Path(images_dir).glob('*.*'))
        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)
        
        splits = {
            'train': images[:train_end],
            'valid': images[train_end:valid_end],
            'test': images[valid_end:]
        }
        
        for split_name, split_images in splits.items():
            for img_path in split_images:
                # Copy image
                dest_img = self.data_dir / split_name / 'images' / img_path.name
                shutil.copy(img_path, dest_img)
                
                # Copy label
                label_name = img_path.stem + '.txt'
                label_path = Path(labels_dir) / label_name
                if label_path.exists():
                    dest_label = self.data_dir / split_name / 'labels' / label_name
                    shutil.copy(label_path, dest_label)
            
            print(f"  {split_name}: {len(split_images)} images")
        
        print(f"✓ Split complete")
    
    def convert_coco_to_yolo(
        self,
        coco_json_path: str,
        images_dir: str,
        output_labels_dir: str
    ):
        """
        Convert COCO format annotations to YOLO format.
        
        Args:
            coco_json_path: Path to COCO JSON annotation file
            images_dir: Directory containing images
            output_labels_dir: Output directory for YOLO labels
        """
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build image lookup
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Convert annotations
        os.makedirs(output_labels_dir, exist_ok=True)
        
        for img_id, anns in annotations_by_image.items():
            img_info = images[img_id]
            img_w = img_info['width']
            img_h = img_info['height']
            
            label_lines = []
            for ann in anns:
                # COCO bbox: [x, y, width, height]
                bbox = ann['bbox']
                x, y, w, h = bbox
                
                # Convert to YOLO: [class, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                class_id = ann['category_id']
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # Save label file
            label_filename = Path(img_info['file_name']).stem + '.txt'
            label_path = Path(output_labels_dir) / label_filename
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
        
        print(f"✓ Converted {len(annotations_by_image)} annotations to YOLO format")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            if images_dir.exists():
                img_count = len(list(images_dir.glob('*.*')))
                lbl_count = len(list(labels_dir.glob('*.txt')))
                stats[split] = {'images': img_count, 'labels': lbl_count}
        
        return stats


def create_preparer(data_dir: str) -> DatasetPreparer:
    """Factory function to create DatasetPreparer."""
    return DatasetPreparer(data_dir)
