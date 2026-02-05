"""
Text Detection and Classification Module
Uses EasyOCR for text extraction and DistilBERT for context-aware classification.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image
import yaml


class TextDetector:
    """
    OCR + NLP pipeline for detecting and classifying text in images.
    
    - EasyOCR for text extraction
    - DistilBERT for context-aware classification
    
    Classifications: SAFE, PROMOTIONAL, ABUSIVE
    """
    
    # Label mappings
    LABEL_MAP = {
        0: 'SAFE',
        1: 'PROMOTIONAL',
        2: 'ABUSIVE'
    }
    
    def __init__(
        self,
        nlp_model_path: Optional[str] = None,
        config_path: str = "config/text_classification.yaml",
        languages: List[str] = ['en'],
        device: str = "cuda"
    ):
        """
        Initialize the text detector.
        
        Args:
            nlp_model_path: Path to trained DistilBERT model
            config_path: Path to text classification config
            languages: List of languages for OCR
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.languages = languages
        self.ocr_reader = None
        self.nlp_model = None
        self.tokenizer = None
        
        # Load configuration
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Initialize OCR
        self._init_ocr()
        
        # Initialize NLP model
        if nlp_model_path:
            self._load_nlp_model(nlp_model_path)
    
    def _init_ocr(self):
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            gpu = self.device == 'cuda'
            self.ocr_reader = easyocr.Reader(self.languages, gpu=gpu)
            print(f"✓ EasyOCR initialized with languages: {self.languages}")
        except Exception as e:
            print(f"✗ Failed to initialize EasyOCR: {e}")
    
    def _load_nlp_model(self, model_path: str):
        """Load DistilBERT classification model."""
        try:
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizer
            )
            import torch
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.nlp_model = DistilBertForSequenceClassification.from_pretrained(model_path)
            
            if self.device == 'cuda' and torch.cuda.is_available():
                self.nlp_model = self.nlp_model.cuda()
            
            self.nlp_model.eval()
            print(f"✓ NLP model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Failed to load NLP model: {e}")
    
    def extract_text(
        self,
        image: Union[str, np.ndarray, Image.Image],
        min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Extract text from image using OCR.
        
        Args:
            image: Image path, numpy array, or PIL Image
            min_confidence: Minimum OCR confidence threshold
            
        Returns:
            List of detected text regions with bbox and confidence
        """
        if self.ocr_reader is None:
            return []
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run OCR
        results = self.ocr_reader.readtext(image)
        
        extracted = []
        for (bbox, text, confidence) in results:
            if confidence >= min_confidence:
                # Convert bbox to standard format
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                extracted.append({
                    'text': text,
                    'confidence': round(float(confidence), 4),
                    'bbox': {
                        'x1': int(min(x_coords)),
                        'y1': int(min(y_coords)),
                        'x2': int(max(x_coords)),
                        'y2': int(max(y_coords))
                    }
                })
        
        return extracted
    
    def classify_text(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Classify text using NLP model.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification result with label and confidence
        """
        if self.nlp_model is None or self.tokenizer is None:
            # Fallback to rule-based classification
            return self._rule_based_classify(text)
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.nlp_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()
        
        return {
            'label': self.LABEL_MAP[pred_class],
            'label_id': pred_class,
            'confidence': round(confidence, 4),
            'all_scores': {
                self.LABEL_MAP[i]: round(probs[0][i].item(), 4)
                for i in range(len(self.LABEL_MAP))
            }
        }
    
    def _rule_based_classify(self, text: str) -> Dict[str, Any]:
        """
        Fallback rule-based classification when NLP model not available.
        Uses keyword patterns from config.
        """
        text_lower = text.lower()
        
        # Get patterns from config
        promo_patterns = self.config.get('promotional_patterns', [
            'buy now', 'sale', 'discount', '% off', 'free shipping',
            'limited offer', 'order today', 'special offer'
        ])
        
        # Check for promotional content
        promo_score = 0
        matched_promo = []
        for pattern in promo_patterns:
            if pattern.lower() in text_lower:
                promo_score += 1
                matched_promo.append(pattern)
        
        # Simple abusive word check (this is basic - real model would be better)
        abusive_indicators = ['hate', 'kill', 'die', 'stupid', 'idiot']
        abusive_score = sum(1 for word in abusive_indicators if word in text_lower)
        
        # Determine label
        if abusive_score >= 1:
            label = 'ABUSIVE'
            confidence = min(0.5 + abusive_score * 0.1, 0.9)
        elif promo_score >= 1:
            label = 'PROMOTIONAL'
            confidence = min(0.5 + promo_score * 0.1, 0.9)
        else:
            label = 'SAFE'
            confidence = 0.8
        
        return {
            'label': label,
            'label_id': list(self.LABEL_MAP.values()).index(label),
            'confidence': round(confidence, 4),
            'matched_patterns': matched_promo if label == 'PROMOTIONAL' else [],
            'note': 'Rule-based classification (NLP model not loaded)'
        }
    
    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Full OCR + NLP pipeline: extract text and classify.
        
        Args:
            image: Image to process
            
        Returns:
            Complete detection and classification result
        """
        # Step 1: Extract text
        extracted_texts = self.extract_text(image)
        
        if not extracted_texts:
            return {
                'text_found': False,
                'extracted_texts': [],
                'combined_text': '',
                'classification': {
                    'label': 'SAFE',
                    'confidence': 1.0,
                    'reason': 'No text detected in image'
                }
            }
        
        # Step 2: Combine all text
        combined_text = ' '.join([t['text'] for t in extracted_texts])
        
        # Step 3: Classify combined text
        classification = self.classify_text(combined_text)
        
        # Step 4: Also classify individual texts for detailed analysis
        individual_classifications = []
        for text_region in extracted_texts:
            text_class = self.classify_text(text_region['text'])
            individual_classifications.append({
                'text': text_region['text'],
                'bbox': text_region['bbox'],
                'classification': text_class
            })
        
        # Determine overall classification (worst case)
        worst_label = 'SAFE'
        for cls in individual_classifications:
            if cls['classification']['label'] == 'ABUSIVE':
                worst_label = 'ABUSIVE'
                break
            elif cls['classification']['label'] == 'PROMOTIONAL':
                worst_label = 'PROMOTIONAL'
        
        return {
            'text_found': True,
            'extracted_texts': extracted_texts,
            'combined_text': combined_text,
            'classification': {
                'label': worst_label,
                'confidence': classification['confidence'],
                'combined_analysis': classification,
                'individual_analysis': individual_classifications
            }
        }
    
    def get_explanation(self, result: Dict[str, Any]) -> str:
        """
        Get human-readable explanation of classification.
        
        Args:
            result: Detection result from detect() method
            
        Returns:
            Explanation string
        """
        if not result['text_found']:
            return "No text was detected in the image."
        
        label = result['classification']['label']
        text = result['combined_text']
        
        if label == 'SAFE':
            return f"Text detected: '{text[:50]}...' - Content is safe."
        elif label == 'PROMOTIONAL':
            return f"Promotional content detected: '{text[:50]}...' - Contains sales/marketing language."
        else:
            return f"Potentially abusive content detected: '{text[:50]}...' - Contains harmful language."


def create_detector(
    nlp_model_path: str = None,
    device: str = "cuda"
) -> TextDetector:
    """
    Factory function to create a text detector.
    
    Args:
        nlp_model_path: Path to trained NLP model
        device: 'cuda' or 'cpu'
        
    Returns:
        Configured TextDetector instance
    """
    return TextDetector(
        nlp_model_path=nlp_model_path,
        device=device
    )
