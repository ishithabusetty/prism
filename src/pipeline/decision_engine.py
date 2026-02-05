"""
Decision Engine Module
Aggregates results from all detectors and makes final SAFE/UNSAFE decision.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class SafetyStatus(Enum):
    """Safety classification for content."""
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"


class UnsafeReason(Enum):
    """Reasons for UNSAFE classification."""
    WEAPON_DETECTED = "Weapon or dangerous object detected"
    ABUSIVE_TEXT = "Abusive or harmful text detected"
    PROMOTIONAL_TEXT = "Promotional/advertising content detected"
    COMPETITOR_LOGO = "Competitor brand logo detected"


@dataclass
class DetectionFlag:
    """Represents a single detection flag."""
    reason: UnsafeReason
    confidence: float
    details: Dict[str, Any]
    priority: int  # Lower = higher priority


@dataclass
class DecisionResult:
    """Final decision result with all details."""
    status: SafetyStatus
    flags: List[DetectionFlag]
    summary: str
    weapon_results: Optional[Dict[str, Any]] = None
    text_results: Optional[Dict[str, Any]] = None
    logo_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': self.status.value,
            'is_safe': self.status == SafetyStatus.SAFE,
            'flags': [
                {
                    'reason': f.reason.value,
                    'confidence': f.confidence,
                    'details': f.details,
                    'priority': f.priority
                }
                for f in self.flags
            ],
            'summary': self.summary,
            'detailed_results': {
                'weapon_detection': self.weapon_results,
                'text_classification': self.text_results,
                'logo_detection': self.logo_results
            }
        }


class DecisionEngine:
    """
    Aggregates detection results and makes final SAFE/UNSAFE decision.
    
    Decision Logic:
    - UNSAFE if ANY weapon detected (confidence >= threshold)
    - UNSAFE if abusive text detected
    - UNSAFE if promotional text detected (optional, configurable)
    - UNSAFE if competitor logo detected
    - SAFE otherwise
    """
    
    def __init__(
        self,
        weapon_threshold: float = 0.5,
        logo_threshold: float = 0.6,
        text_promotional_threshold: float = 0.7,
        text_abusive_threshold: float = 0.8,
        flag_promotional_as_unsafe: bool = True
    ):
        """
        Initialize decision engine with thresholds.
        
        Args:
            weapon_threshold: Minimum confidence for weapon detection
            logo_threshold: Minimum confidence for logo detection
            text_promotional_threshold: Threshold for promotional text
            text_abusive_threshold: Threshold for abusive text
            flag_promotional_as_unsafe: Whether to flag promotional content
        """
        self.weapon_threshold = weapon_threshold
        self.logo_threshold = logo_threshold
        self.text_promotional_threshold = text_promotional_threshold
        self.text_abusive_threshold = text_abusive_threshold
        self.flag_promotional_as_unsafe = flag_promotional_as_unsafe
    
    def evaluate(
        self,
        weapon_results: Optional[Dict[str, Any]] = None,
        text_results: Optional[Dict[str, Any]] = None,
        logo_results: Optional[Dict[str, Any]] = None
    ) -> DecisionResult:
        """
        Evaluate all detection results and make final decision.
        
        Args:
            weapon_results: Results from WeaponDetector
            text_results: Results from TextDetector
            logo_results: Results from LogoDetector
            
        Returns:
            DecisionResult with final status and all flags
        """
        flags: List[DetectionFlag] = []
        
        # Check weapon detection
        if weapon_results and weapon_results.get('detected', False):
            detections = weapon_results.get('detections', [])
            high_conf_detections = [
                d for d in detections
                if d.get('confidence', 0) >= self.weapon_threshold
            ]
            
            if high_conf_detections:
                flags.append(DetectionFlag(
                    reason=UnsafeReason.WEAPON_DETECTED,
                    confidence=max(d['confidence'] for d in high_conf_detections),
                    details={
                        'detected_objects': [d['class'] for d in high_conf_detections],
                        'count': len(high_conf_detections),
                        'detections': high_conf_detections
                    },
                    priority=1
                ))
        
        # Check text classification
        if text_results and text_results.get('text_found', False):
            classification = text_results.get('classification', {})
            label = classification.get('label', 'SAFE')
            confidence = classification.get('confidence', 0)
            
            if label == 'ABUSIVE' and confidence >= self.text_abusive_threshold:
                flags.append(DetectionFlag(
                    reason=UnsafeReason.ABUSIVE_TEXT,
                    confidence=confidence,
                    details={
                        'detected_text': text_results.get('combined_text', ''),
                        'classification': classification
                    },
                    priority=2
                ))
            elif label == 'PROMOTIONAL' and self.flag_promotional_as_unsafe:
                if confidence >= self.text_promotional_threshold:
                    flags.append(DetectionFlag(
                        reason=UnsafeReason.PROMOTIONAL_TEXT,
                        confidence=confidence,
                        details={
                            'detected_text': text_results.get('combined_text', ''),
                            'classification': classification
                        },
                        priority=4
                    ))
        
        # Check logo detection
        if logo_results and logo_results.get('competitor_detected', False):
            competitor_detections = logo_results.get('competitor_detections', [])
            high_conf_logos = [
                d for d in competitor_detections
                if d.get('confidence', 0) >= self.logo_threshold
            ]
            
            if high_conf_logos:
                flags.append(DetectionFlag(
                    reason=UnsafeReason.COMPETITOR_LOGO,
                    confidence=max(d['confidence'] for d in high_conf_logos),
                    details={
                        'detected_brands': [d['brand'] for d in high_conf_logos],
                        'count': len(high_conf_logos),
                        'detections': high_conf_logos
                    },
                    priority=3
                ))
        
        # Sort flags by priority
        flags.sort(key=lambda f: f.priority)
        
        # Determine final status
        if flags:
            status = SafetyStatus.UNSAFE
            # Build summary from highest priority flag
            primary_flag = flags[0]
            summary = self._build_summary(primary_flag, len(flags))
        else:
            status = SafetyStatus.SAFE
            summary = "No harmful content detected. Content is safe."
        
        return DecisionResult(
            status=status,
            flags=flags,
            summary=summary,
            weapon_results=weapon_results,
            text_results=text_results,
            logo_results=logo_results
        )
    
    def _build_summary(self, primary_flag: DetectionFlag, total_flags: int) -> str:
        """Build human-readable summary from primary flag."""
        reason = primary_flag.reason
        
        if reason == UnsafeReason.WEAPON_DETECTED:
            objects = primary_flag.details.get('detected_objects', [])
            return f"UNSAFE: Weapon detected ({', '.join(objects)}). Confidence: {primary_flag.confidence:.2%}"
        
        elif reason == UnsafeReason.ABUSIVE_TEXT:
            text = primary_flag.details.get('detected_text', '')[:50]
            return f"UNSAFE: Abusive text detected. Content: '{text}...'"
        
        elif reason == UnsafeReason.PROMOTIONAL_TEXT:
            text = primary_flag.details.get('detected_text', '')[:50]
            return f"UNSAFE: Promotional content detected. Content: '{text}...'"
        
        elif reason == UnsafeReason.COMPETITOR_LOGO:
            brands = primary_flag.details.get('detected_brands', [])
            return f"UNSAFE: Competitor logo detected ({', '.join(brands)})"
        
        return f"UNSAFE: {reason.value}"
    
    def get_detailed_explanation(self, result: DecisionResult) -> str:
        """
        Generate detailed explanation of the decision.
        
        Args:
            result: DecisionResult from evaluate()
            
        Returns:
            Multi-line explanation string
        """
        lines = [
            "=" * 50,
            f"FINAL DECISION: {result.status.value}",
            "=" * 50,
            ""
        ]
        
        if result.status == SafetyStatus.SAFE:
            lines.append("✓ No harmful content detected")
            lines.append("✓ No weapons found")
            lines.append("✓ No abusive or promotional text")
            lines.append("✓ No competitor logos detected")
        else:
            lines.append(f"⚠ {len(result.flags)} issue(s) detected:")
            lines.append("")
            
            for i, flag in enumerate(result.flags, 1):
                lines.append(f"{i}. {flag.reason.value}")
                lines.append(f"   Confidence: {flag.confidence:.2%}")
                lines.append(f"   Priority: {flag.priority}")
                
                if flag.reason == UnsafeReason.WEAPON_DETECTED:
                    objects = flag.details.get('detected_objects', [])
                    lines.append(f"   Objects: {', '.join(objects)}")
                
                elif flag.reason in [UnsafeReason.ABUSIVE_TEXT, UnsafeReason.PROMOTIONAL_TEXT]:
                    text = flag.details.get('detected_text', '')
                    lines.append(f"   Text: '{text[:100]}...'")
                
                elif flag.reason == UnsafeReason.COMPETITOR_LOGO:
                    brands = flag.details.get('detected_brands', [])
                    lines.append(f"   Brands: {', '.join(brands)}")
                
                lines.append("")
        
        lines.append("=" * 50)
        return "\n".join(lines)
