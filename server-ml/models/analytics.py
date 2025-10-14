"""
Pydantic models for analytics and monitoring in the LLM learning system.

This module defines the data structures used for advanced analytics including
skill gap analysis, bottleneck detection, performance prediction, and learning insights.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime


class SkillGapSeverity(str, Enum):
    """Severity levels for skill gaps."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BottleneckType(str, Enum):
    """Types of learning bottlenecks."""
    CONCEPTUAL = "conceptual"
    PRACTICAL = "practical"
    MOTIVATIONAL = "motivational"
    RESOURCE = "resource"
    TECHNICAL = "technical"


class PredictionConfidence(str, Enum):
    """Confidence levels for performance predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InsightPriority(str, Enum):
    """Priority levels for learning insights."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class SkillGap(BaseModel):
    """Individual skill gap identified in the learning system."""
    skill_id: str
    skill_name: str
    current_level: float = Field(..., ge=0.0, le=1.0, description="Current mastery level (0.0-1.0)")
    required_level: float = Field(..., ge=0.0, le=1.0, description="Required mastery level (0.0-1.0)")
    gap_size: float = Field(..., ge=0.0, le=1.0, description="Size of the skill gap")
    severity: SkillGapSeverity
    affected_tasks: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    estimated_time_to_close: Optional[int] = None  # Hours


class SkillGapAnalysis(BaseModel):
    """Complete analysis of skill gaps in the learning system."""
    learner_id: str
    timestamp: float
    skill_gaps: List[SkillGap]
    overall_gap_score: float = Field(..., ge=0.0, le=1.0, description="Overall skill gap severity score")
    critical_gaps_count: int
    recommendations: List[str] = Field(default_factory=list)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)


class LearningBottleneck(BaseModel):
    """Individual learning bottleneck identified in the system."""
    bottleneck_id: str
    type: BottleneckType
    description: str
    severity: float = Field(..., ge=0.0, le=1.0, description="Bottleneck severity (0.0-1.0)")
    affected_area: str
    root_causes: List[str] = Field(default_factory=list)
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Impact on learning progress")
    resolution_suggestions: List[str] = Field(default_factory=list)
    estimated_resolution_time: Optional[int] = None  # Hours


class BottleneckDetection(BaseModel):
    """Complete bottleneck detection analysis."""
    timestamp: float
    bottlenecks: List[LearningBottleneck]
    overall_bottleneck_score: float = Field(..., ge=0.0, le=1.0, description="Overall bottleneck severity")
    critical_bottlenecks_count: int
    system_health_score: float = Field(..., ge=0.0, le=1.0, description="Overall system health")
    recommendations: List[str] = Field(default_factory=list)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)


class PerformancePrediction(BaseModel):
    """Performance prediction for future learning outcomes."""
    learner_id: str
    prediction_horizon: int  # Days
    timestamp: float
    predicted_performance: float = Field(..., ge=0.0, le=1.0, description="Predicted performance level")
    confidence_level: PredictionConfidence
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Numerical confidence score")
    factors_influencing: Dict[str, float] = Field(default_factory=dict)  # factor -> impact
    risk_factors: List[str] = Field(default_factory=list)
    improvement_trajectory: List[Dict[str, Any]] = Field(default_factory=list)  # Time series predictions
    recommendations: List[str] = Field(default_factory=list)
    prediction_metadata: Dict[str, Any] = Field(default_factory=dict)


class LearningInsight(BaseModel):
    """Individual actionable learning insight."""
    insight_id: str
    title: str
    description: str
    priority: InsightPriority
    category: str
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Expected impact of implementing this insight")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this insight")
    actionable_steps: List[str] = Field(default_factory=list)
    expected_benefits: List[str] = Field(default_factory=list)
    implementation_complexity: str  # "low", "medium", "high"
    timeframe: str  # "immediate", "short_term", "long_term"


class LearningInsights(BaseModel):
    """Collection of learning insights and recommendations."""
    timestamp: float
    insights: List[LearningInsight]
    overall_insight_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality/usefulness of insights")
    high_priority_count: int
    categories_covered: List[str] = Field(default_factory=list)
    implementation_roadmap: List[Dict[str, Any]] = Field(default_factory=list)
    insights_metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemHealthMetrics(BaseModel):
    """System health monitoring metrics."""
    timestamp: float
    overall_health_score: float = Field(..., ge=0.0, le=1.0, description="Overall system health")
    component_health: Dict[str, float] = Field(default_factory=dict)  # component -> health_score
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    error_rates: Dict[str, float] = Field(default_factory=dict)
    resource_utilization: Dict[str, float] = Field(default_factory=dict)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)


class TrendAnalysis(BaseModel):
    """Trend analysis for learning patterns."""
    metric_name: str
    timeframe: str  # "hour", "day", "week", "month"
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of the trend")
    data_points: List[Dict[str, Any]] = Field(default_factory=list)
    statistical_significance: float = Field(..., ge=0.0, le=1.0, description="Statistical significance of trend")
    insights: List[str] = Field(default_factory=list)


class AnomalyDetection(BaseModel):
    """Anomaly detection in learning patterns."""
    timestamp: float
    anomaly_type: str
    severity: float = Field(..., ge=0.0, le=1.0, description="Anomaly severity")
    affected_metric: str
    description: str
    detected_value: Union[float, int, str]
    expected_range: Dict[str, Any] = Field(default_factory=dict)
    potential_causes: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class AnalyticsDashboard(BaseModel):
    """Comprehensive analytics dashboard data."""
    timestamp: float
    learner_id: Optional[str] = None

    # Core analytics
    skill_gap_analysis: Optional[SkillGapAnalysis] = None
    bottleneck_detection: Optional[BottleneckDetection] = None
    performance_prediction: Optional[PerformancePrediction] = None
    learning_insights: Optional[LearningInsights] = None

    # Monitoring data
    system_health: Optional[SystemHealthMetrics] = None
    trend_analysis: List[TrendAnalysis] = Field(default_factory=list)
    anomalies: List[AnomalyDetection] = Field(default_factory=list)

    # Summary metrics
    overall_learning_score: float = Field(..., ge=0.0, le=1.0, description="Overall learning system score")
    key_indicators: Dict[str, Any] = Field(default_factory=dict)
    alerts: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    data_freshness: Dict[str, float] = Field(default_factory=dict)  # component -> last_update_timestamp
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)


# Request/Response models for API endpoints

class AnalyticsDashboardRequest(BaseModel):
    """Request model for analytics dashboard."""
    learner_id: Optional[str] = None
    include_historical: bool = False
    time_range: Optional[str] = None  # "hour", "day", "week", "month"


class AnalyticsDashboardResponse(BaseModel):
    """Response model for analytics dashboard."""
    dashboard: AnalyticsDashboard
    processing_time: float
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score of the analytics data")


class SkillGapAnalysisRequest(BaseModel):
    """Request model for skill gap analysis."""
    learner_id: str
    include_recommendations: bool = True


class SkillGapAnalysisResponse(BaseModel):
    """Response model for skill gap analysis."""
    analysis: SkillGapAnalysis
    processing_time: float


class BottleneckDetectionRequest(BaseModel):
    """Request model for bottleneck detection."""
    include_system_wide: bool = True
    learner_id: Optional[str] = None


class BottleneckDetectionResponse(BaseModel):
    """Response model for bottleneck detection."""
    analysis: BottleneckDetection
    processing_time: float


class PerformancePredictionRequest(BaseModel):
    """Request model for performance prediction."""
    learner_id: str
    prediction_horizon_days: int = 30
    include_factors: bool = True


class PerformancePredictionResponse(BaseModel):
    """Response model for performance prediction."""
    prediction: PerformancePrediction
    processing_time: float


class LearningInsightsRequest(BaseModel):
    """Request model for learning insights."""
    learner_id: Optional[str] = None
    categories: Optional[List[str]] = None
    priority_filter: Optional[InsightPriority] = None


class LearningInsightsResponse(BaseModel):
    """Response model for learning insights."""
    insights: LearningInsights
    processing_time: float


class SystemHealthRequest(BaseModel):
    """Request model for system health monitoring."""
    include_detailed_metrics: bool = False


class SystemHealthResponse(BaseModel):
    """Response model for system health monitoring."""
    health: SystemHealthMetrics
    processing_time: float


class AnalyticsConfig(BaseModel):
    """Configuration for analytics and monitoring."""
    enabled: bool = True
    update_interval_seconds: int = 300  # 5 minutes
    retention_days: int = 90
    anomaly_detection_threshold: float = 0.8
    prediction_confidence_threshold: float = 0.7
    alert_thresholds: Dict[str, float] = Field(default_factory=dict)
    monitoring_components: List[str] = Field(default_factory=lambda: [
        "reward_service", "hallucination_service", "meta_learning_service",
        "curriculum_service", "learning_loop", "external_llm_service"
    ])