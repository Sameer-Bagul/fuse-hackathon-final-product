"""
Pydantic models for hallucination detection and analysis.

This module defines the data structures used for hallucination detection,
analysis results, configuration, and metrics tracking.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class HallucinationType(str, Enum):
    """Types of hallucinations that can be detected."""
    OVERCONFIDENT = "overconfident"
    CONTRADICTION = "contradiction"
    FACTUAL_INACCURACY = "factual_inaccuracy"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    INCONSISTENT = "inconsistent"


class HallucinationSeverity(str, Enum):
    """Severity levels for detected hallucinations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HallucinationIndicator(BaseModel):
    """Individual hallucination indicator found in text."""
    type: HallucinationType
    severity: HallucinationSeverity
    confidence: float = Field(..., ge=0.0, le=1.0)
    text_snippet: str
    explanation: str
    position: Optional[Dict[str, int]] = None  # start/end character positions


class HallucinationAnalysis(BaseModel):
    """Complete analysis result for a text response."""
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score (1.0 = likely hallucinated)")
    is_hallucinated: bool
    indicators: List[HallucinationIndicator]
    uncertainty_score: float = Field(..., ge=0.0, le=1.0, description="Uncertainty metric")
    factuality_score: float = Field(..., ge=0.0, le=1.0, description="Factuality assessment")
    risk_level: HallucinationSeverity
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)


class HallucinationConfig(BaseModel):
    """Configuration for hallucination detection thresholds and parameters."""
    enabled: bool = True
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Threshold for flagging hallucinations")
    uncertainty_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Threshold for high uncertainty")
    factuality_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight for factuality in overall score")
    pattern_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for pattern-based detection")
    contradiction_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for contradiction detection")

    # Pattern detection parameters
    overconfident_phrases: List[str] = Field(
        default_factory=lambda: [
            "absolutely certain", "definitely", "without a doubt", "clearly", "obviously",
            "undoubtedly", "certainly", "unquestionably", "indisputably", "incontrovertibly"
        ]
    )
    hedging_phrases: List[str] = Field(
        default_factory=lambda: [
            "might", "may", "could", "possibly", "perhaps", "maybe", "I think", "seems",
            "appears", "likely", "probably", "potentially"
        ]
    )

    # Fact-checking parameters
    enable_fact_checking: bool = True
    knowledge_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "ml_algorithms": ["neural network", "decision tree", "random forest", "svm", "k-means"],
            "programming_languages": ["python", "java", "javascript", "c++", "r"],
            "ml_libraries": ["tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"]
        }
    )


class HallucinationMetrics(BaseModel):
    """Metrics and statistics for hallucination detection performance."""
    total_checks: int = 0
    hallucinated_responses: int = 0
    average_confidence: float = 0.0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    average_uncertainty: float = 0.0
    common_hallucination_types: Dict[str, int] = Field(default_factory=dict)
    severity_distribution: Dict[str, int] = Field(default_factory=dict)


class HallucinationCheckRequest(BaseModel):
    """Request model for hallucination checking API."""
    prompt_text: str
    response_text: str
    context: Optional[Dict[str, Any]] = None
    config_override: Optional[HallucinationConfig] = None


class HallucinationCheckResponse(BaseModel):
    """Response model for hallucination checking API."""
    analysis: HallucinationAnalysis
    processing_time: float
    config_used: HallucinationConfig


class HallucinationConfigUpdate(BaseModel):
    """Request model for updating hallucination configuration."""
    config: HallucinationConfig


class HallucinationMetricsResponse(BaseModel):
    """Response model for hallucination metrics API."""
    metrics: HallucinationMetrics
    recent_analyses: List[HallucinationAnalysis] = Field(default_factory=list)
    timestamp: float