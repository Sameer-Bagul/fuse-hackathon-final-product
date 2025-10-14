"""
Pydantic models for user feedback integration and analysis.

This module defines the data structures used for collecting, processing, and analyzing
user feedback on AI responses, including corrections, preferences, and collaborative learning.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class FeedbackType(str, Enum):
    """Types of user feedback that can be provided."""
    RATING = "rating"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    COMMENT = "comment"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    ERROR_REPORT = "error_report"


class FeedbackCategory(str, Enum):
    """Categories for classifying feedback."""
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    FACTUALITY = "factuality"
    CREATIVITY = "creativity"
    USEFULNESS = "usefulness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"


class CorrectionType(str, Enum):
    """Types of corrections users can provide."""
    FACTUAL_ERROR = "factual_error"
    LOGICAL_ERROR = "logical_error"
    INCOMPLETE_INFO = "incomplete_info"
    BETTER_ALTERNATIVE = "better_alternative"
    STYLE_IMPROVEMENT = "style_improvement"
    FORMAT_ISSUE = "format_issue"


class UserFeedback(BaseModel):
    """Individual feedback instance from a user."""
    user_id: str
    session_id: Optional[str] = None
    prompt_id: str
    response_id: str
    feedback_type: FeedbackType
    category: FeedbackCategory
    rating: Optional[float] = Field(None, ge=1.0, le=5.0, description="Rating from 1-5 stars")
    comment: Optional[str] = None
    correction_text: Optional[str] = None
    correction_type: Optional[CorrectionType] = None
    preference_data: Optional[Dict[str, Any]] = None  # For preference-based feedback
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    is_processed: bool = False


class FeedbackAnalysis(BaseModel):
    """Aggregated feedback insights and analysis."""
    prompt_id: str
    response_id: str
    total_feedbacks: int
    average_rating: Optional[float] = None
    rating_distribution: Dict[int, int] = Field(default_factory=dict)  # rating -> count
    category_breakdown: Dict[FeedbackCategory, int] = Field(default_factory=dict)
    common_issues: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment (-1 to 1)")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    feedback_trends: Dict[str, Any] = Field(default_factory=dict)


class CorrectionData(BaseModel):
    """User corrections and improvements for wrong answers."""
    correction_id: str
    original_prompt: str
    original_response: str
    corrected_response: str
    correction_type: CorrectionType
    user_id: str
    explanation: Optional[str] = None
    improvement_tags: List[str] = Field(default_factory=list)
    validation_status: str = "pending"  # pending, validated, rejected
    validated_by: Optional[str] = None
    validation_timestamp: Optional[datetime] = None
    usage_count: int = 0  # How many times this correction has been used
    effectiveness_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class PreferenceProfile(BaseModel):
    """Learned user preferences for personalization."""
    user_id: str
    preference_weights: Dict[str, float] = Field(default_factory=dict)  # category -> weight
    preferred_styles: List[str] = Field(default_factory=list)
    disliked_patterns: List[str] = Field(default_factory=list)
    response_length_preference: Optional[str] = None  # short, medium, long
    complexity_preference: Optional[str] = None  # simple, moderate, complex
    domain_expertise: Dict[str, float] = Field(default_factory=dict)  # domain -> expertise level
    feedback_history: List[str] = Field(default_factory=list)  # feedback_ids
    last_updated: datetime = Field(default_factory=datetime.now)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)  # preference -> confidence
    adaptation_count: int = 0


class CollaborativeLearningData(BaseModel):
    """Data for collaborative learning from user feedback."""
    pattern_id: str
    pattern_type: str  # error_pattern, improvement_pattern, preference_pattern
    description: str
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    correction_templates: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    usage_frequency: int = 0
    last_used: Optional[datetime] = None
    created_from_feedbacks: List[str] = Field(default_factory=list)  # feedback_ids


class FeedbackMetrics(BaseModel):
    """Metrics and statistics for feedback system performance."""
    total_feedbacks: int = 0
    processed_feedbacks: int = 0
    average_rating: Optional[float] = None
    feedback_categories: Dict[FeedbackCategory, int] = Field(default_factory=dict)
    correction_types: Dict[CorrectionType, int] = Field(default_factory=dict)
    user_engagement_rate: float = 0.0
    correction_adoption_rate: float = 0.0
    preference_learning_accuracy: float = 0.0
    collaborative_patterns_learned: int = 0
    feedback_quality_score: float = 0.0


# API Request/Response Models

class SubmitFeedbackRequest(BaseModel):
    """Request model for submitting user feedback."""
    user_id: str
    session_id: Optional[str] = None
    prompt_id: str
    response_id: str
    feedback_type: FeedbackType
    category: FeedbackCategory
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    comment: Optional[str] = None
    correction_text: Optional[str] = None
    correction_type: Optional[CorrectionType] = None
    preference_data: Optional[Dict[str, Any]] = None


class SubmitFeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    feedback_id: str
    success: bool
    message: str
    processed_immediately: bool = False


class GetFeedbackHistoryRequest(BaseModel):
    """Request model for retrieving user's feedback history."""
    user_id: str
    limit: int = 50
    offset: int = 0
    category_filter: Optional[FeedbackCategory] = None
    feedback_type_filter: Optional[FeedbackType] = None


class GetFeedbackHistoryResponse(BaseModel):
    """Response model for feedback history."""
    user_id: str
    feedbacks: List[UserFeedback]
    total_count: int
    has_more: bool


class SubmitCorrectionRequest(BaseModel):
    """Request model for submitting corrections."""
    user_id: str
    prompt_id: str
    response_id: str
    corrected_response: str
    correction_type: CorrectionType
    explanation: Optional[str] = None
    improvement_tags: List[str] = Field(default_factory=list)


class SubmitCorrectionResponse(BaseModel):
    """Response model for correction submission."""
    correction_id: str
    success: bool
    message: str
    validation_status: str


class GetPreferencesRequest(BaseModel):
    """Request model for retrieving user preferences."""
    user_id: str
    include_history: bool = False


class GetPreferencesResponse(BaseModel):
    """Response model for user preferences."""
    user_id: str
    preferences: PreferenceProfile
    last_updated: datetime
    confidence_level: float


class RateResponseRequest(BaseModel):
    """Request model for rating a response."""
    user_id: str
    response_id: str
    rating: float = Field(..., ge=1.0, le=5.0)
    category: FeedbackCategory = FeedbackCategory.USEFULNESS
    comment: Optional[str] = None


class RateResponseResponse(BaseModel):
    """Response model for response rating."""
    success: bool
    message: str
    updated_preferences: bool = False


class SubmitErrorReportRequest(BaseModel):
    """Request model for submitting error reports."""
    user_id: str
    session_id: Optional[str] = None
    error_message: str
    stack_trace: Optional[str] = None
    context: Optional[str] = None
    component: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical
    user_agent: Optional[str] = None
    url: Optional[str] = None


class SubmitErrorReportResponse(BaseModel):
    """Response model for error report submission."""
    error_report_id: str
    success: bool
    message: str
    processed_immediately: bool = False