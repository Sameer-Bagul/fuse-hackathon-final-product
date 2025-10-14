"""
User Feedback Integration Service

This service provides comprehensive user feedback processing capabilities including:
- Storage and analysis of user feedback on AI responses
- Preference learning from user choices and ratings
- Correction mechanisms for wrong answers
- Collaborative learning from aggregated feedback patterns
- Integration with reward systems and learning loops
"""

import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json

from models.feedback import (
    UserFeedback,
    FeedbackAnalysis,
    CorrectionData,
    PreferenceProfile,
    CollaborativeLearningData,
    FeedbackMetrics,
    FeedbackType,
    FeedbackCategory,
    CorrectionType,
    SubmitFeedbackRequest,
    SubmitFeedbackResponse,
    GetFeedbackHistoryRequest,
    GetFeedbackHistoryResponse,
    SubmitCorrectionRequest,
    SubmitCorrectionResponse,
    GetPreferencesRequest,
    GetPreferencesResponse,
    RateResponseRequest,
    RateResponseResponse
)
from utils.logging_config import get_logger, log_feedback_processing

logger = get_logger(__name__)


class FeedbackService:
    """
    Service for processing and learning from user feedback.

    Provides feedback storage, preference learning, correction handling,
    and collaborative learning capabilities to enhance AI personalization.
    """

    def __init__(self):
        # Storage for feedback data
        self.feedback_store: Dict[str, UserFeedback] = {}
        self.correction_store: Dict[str, CorrectionData] = {}
        self.preference_store: Dict[str, PreferenceProfile] = {}
        self.feedback_analysis_store: Dict[str, FeedbackAnalysis] = {}

        # Collaborative learning data
        self.collaborative_patterns: Dict[str, CollaborativeLearningData] = {}

        # Metrics tracking
        self.metrics = FeedbackMetrics()

        # Learning parameters
        self.preference_learning_rate = 0.1
        self.collaborative_threshold = 5  # Minimum feedbacks for pattern recognition

        logger.info("FeedbackService initialized")

    def submit_feedback(self, request: SubmitFeedbackRequest) -> SubmitFeedbackResponse:
        """
        Submit user feedback on an AI response.

        Args:
            request: Feedback submission request

        Returns:
            SubmitFeedbackResponse: Response with feedback ID and status
        """
        try:
            feedback_id = str(uuid.uuid4())

            logger.info(f"💬 Processing feedback submission from user {request.user_id}")

            # Log feedback processing start
            log_feedback_processing(request.user_id, "feedback_submission", {
                'response_id': request.response_id,
                'feedback_type': request.feedback_type.value,
                'category': request.category.value if request.category else None,
                'rating': request.rating,
                'has_comment': bool(request.comment),
                'has_correction': bool(request.correction_text)
            })

            feedback = UserFeedback(
                user_id=request.user_id,
                session_id=request.session_id,
                prompt_id=request.prompt_id,
                response_id=request.response_id,
                feedback_type=request.feedback_type,
                category=request.category,
                rating=request.rating,
                comment=request.comment,
                correction_text=request.correction_text,
                correction_type=request.correction_type,
                preference_data=request.preference_data,
                metadata={"feedback_id": feedback_id}
            )

            # Store feedback
            self.feedback_store[feedback_id] = feedback

            # Process feedback immediately for preference learning
            self._process_feedback_for_preferences(feedback)

            # Update collaborative learning patterns
            self._update_collaborative_patterns(feedback)

            # Mark as processed
            feedback.is_processed = True

            # Update metrics
            self._update_metrics(feedback)

            # Log successful feedback processing
            log_feedback_processing(request.user_id, "feedback_processed", {
                'feedback_id': feedback_id,
                'response_id': request.response_id,
                'feedback_type': request.feedback_type.value,
                'rating': request.rating,
                'processed': True,
                'updated_preferences': True,
                'updated_collaborative': True
            })

            logger.info(f"✅ Feedback processed successfully | User: {request.user_id} | "
                       f"Type: {request.feedback_type.value} | Rating: {request.rating}")

            return SubmitFeedbackResponse(
                feedback_id=feedback_id,
                success=True,
                message="Feedback submitted successfully",
                processed_immediately=True
            )

        except Exception as e:
            logger.error(f"❌ Failed to submit feedback: {str(e)}")

            # Log failed feedback processing
            log_feedback_processing(request.user_id, "feedback_failed", {
                'response_id': request.response_id,
                'error': str(e)
            })

            return SubmitFeedbackResponse(
                feedback_id="",
                success=False,
                message=f"Failed to submit feedback: {str(e)}",
                processed_immediately=False
            )

    def submit_correction(self, request: SubmitCorrectionRequest) -> SubmitCorrectionResponse:
        """
        Submit a correction for a wrong AI response.

        Args:
            request: Correction submission request

        Returns:
            SubmitCorrectionResponse: Response with correction ID and status
        """
        try:
            correction_id = str(uuid.uuid4())

            correction = CorrectionData(
                correction_id=correction_id,
                original_prompt="",  # Would be fetched from prompt store
                original_response="",  # Would be fetched from response store
                corrected_response=request.corrected_response,
                correction_type=request.correction_type,
                user_id=request.user_id,
                explanation=request.explanation,
                improvement_tags=request.improvement_tags
            )

            # Store correction
            self.correction_store[correction_id] = correction

            # Update collaborative patterns with correction
            self._learn_from_correction(correction)

            # Update metrics
            self.metrics.correction_types[request.correction_type] += 1

            logger.info(f"Correction submitted: user={request.user_id}, type={request.correction_type.value}")

            return SubmitCorrectionResponse(
                correction_id=correction_id,
                success=True,
                message="Correction submitted successfully",
                validation_status="pending"
            )

        except Exception as e:
            logger.error(f"Failed to submit correction: {str(e)}")
            return SubmitCorrectionResponse(
                correction_id="",
                success=False,
                message=f"Failed to submit correction: {str(e)}",
                validation_status="failed"
            )

    def get_feedback_history(self, request: GetFeedbackHistoryRequest) -> GetFeedbackHistoryResponse:
        """
        Get user's feedback history.

        Args:
            request: History request with filters

        Returns:
            GetFeedbackHistoryResponse: Filtered feedback history
        """
        try:
            # Filter feedbacks by user
            user_feedbacks = [
                fb for fb in self.feedback_store.values()
                if fb.user_id == request.user_id
            ]

            # Apply category filter
            if request.category_filter:
                user_feedbacks = [
                    fb for fb in user_feedbacks
                    if fb.category == request.category_filter
                ]

            # Apply type filter
            if request.feedback_type_filter:
                user_feedbacks = [
                    fb for fb in user_feedbacks
                    if fb.feedback_type == request.feedback_type_filter
                ]

            # Sort by timestamp (newest first)
            user_feedbacks.sort(key=lambda x: x.timestamp, reverse=True)

            # Apply pagination
            total_count = len(user_feedbacks)
            start_idx = request.offset
            end_idx = start_idx + request.limit
            paginated_feedbacks = user_feedbacks[start_idx:end_idx]

            return GetFeedbackHistoryResponse(
                user_id=request.user_id,
                feedbacks=paginated_feedbacks,
                total_count=total_count,
                has_more=end_idx < total_count
            )

        except Exception as e:
            logger.error(f"Failed to get feedback history: {str(e)}")
            return GetFeedbackHistoryResponse(
                user_id=request.user_id,
                feedbacks=[],
                total_count=0,
                has_more=False
            )

    def get_user_preferences(self, request: GetPreferencesRequest) -> GetPreferencesResponse:
        """
        Get learned user preferences.

        Args:
            request: Preferences request

        Returns:
            GetPreferencesResponse: User preference profile
        """
        try:
            preferences = self.preference_store.get(request.user_id)

            if not preferences:
                # Create default preferences if none exist
                preferences = PreferenceProfile(user_id=request.user_id)
                self.preference_store[request.user_id] = preferences

            return GetPreferencesResponse(
                user_id=request.user_id,
                preferences=preferences,
                last_updated=preferences.last_updated,
                confidence_level=self._calculate_preference_confidence(preferences)
            )

        except Exception as e:
            logger.error(f"Failed to get user preferences: {str(e)}")
            # Return default preferences on error
            default_prefs = PreferenceProfile(user_id=request.user_id)
            return GetPreferencesResponse(
                user_id=request.user_id,
                preferences=default_prefs,
                last_updated=datetime.now(),
                confidence_level=0.0
            )

    def rate_response(self, request: RateResponseRequest) -> RateResponseResponse:
        """
        Rate a response and update preferences.

        Args:
            request: Rating request

        Returns:
            RateResponseResponse: Rating response
        """
        try:
            # Create feedback from rating
            feedback_request = SubmitFeedbackRequest(
                user_id=request.user_id,
                response_id=request.response_id,
                feedback_type=FeedbackType.RATING,
                category=request.category,
                rating=request.rating,
                comment=request.comment
            )

            feedback_response = self.submit_feedback(feedback_request)

            return RateResponseResponse(
                success=feedback_response.success,
                message=feedback_response.message,
                updated_preferences=feedback_response.success
            )

        except Exception as e:
            logger.error(f"Failed to rate response: {str(e)}")
            return RateResponseResponse(
                success=False,
                message=f"Failed to rate response: {str(e)}",
                updated_preferences=False
            )

    def analyze_feedback_patterns(self, response_id: str) -> FeedbackAnalysis:
        """
        Analyze aggregated feedback for a specific response.

        Args:
            response_id: ID of the response to analyze

        Returns:
            FeedbackAnalysis: Aggregated feedback insights
        """
        try:
            # Get all feedback for this response
            response_feedbacks = [
                fb for fb in self.feedback_store.values()
                if fb.response_id == response_id
            ]

            if not response_feedbacks:
                return FeedbackAnalysis(
                    prompt_id="",  # Would be fetched from response data
                    response_id=response_id,
                    total_feedbacks=0,
                    sentiment_score=0.0,
                    confidence_score=0.0
                )

            # Calculate metrics
            ratings = [fb.rating for fb in response_feedbacks if fb.rating]
            average_rating = sum(ratings) / len(ratings) if ratings else None

            rating_distribution = Counter(ratings) if ratings else {}

            category_breakdown = Counter(fb.category.value for fb in response_feedbacks)

            # Simple sentiment analysis from ratings and comments
            sentiment_score = self._calculate_sentiment_score(response_feedbacks)

            # Extract common issues and suggestions
            common_issues = self._extract_common_issues(response_feedbacks)
            improvement_suggestions = self._generate_improvement_suggestions(response_feedbacks)

            analysis = FeedbackAnalysis(
                prompt_id=response_feedbacks[0].prompt_id,
                response_id=response_id,
                total_feedbacks=len(response_feedbacks),
                average_rating=average_rating,
                rating_distribution=dict(rating_distribution),
                category_breakdown=dict(category_breakdown),
                common_issues=common_issues,
                improvement_suggestions=improvement_suggestions,
                sentiment_score=sentiment_score,
                confidence_score=min(len(response_feedbacks) / 10.0, 1.0)  # Confidence increases with more feedback
            )

            # Store analysis
            analysis_key = f"{response_id}_analysis"
            self.feedback_analysis_store[analysis_key] = analysis

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze feedback patterns: {str(e)}")
            return FeedbackAnalysis(
                prompt_id="",
                response_id=response_id,
                total_feedbacks=0,
                sentiment_score=0.0,
                confidence_score=0.0
            )

    def get_collaborative_insights(self, category: Optional[FeedbackCategory] = None) -> List[CollaborativeLearningData]:
        """
        Get collaborative learning insights from aggregated feedback.

        Args:
            category: Optional category filter

        Returns:
            List[CollaborativeLearningData]: Collaborative learning patterns
        """
        try:
            patterns = list(self.collaborative_patterns.values())

            if category:
                patterns = [p for p in patterns if category.value in p.pattern_type]

            # Sort by confidence and usage
            patterns.sort(key=lambda x: (x.confidence_score, x.usage_frequency), reverse=True)

            return patterns[:20]  # Return top 20 patterns

        except Exception as e:
            logger.error(f"Failed to get collaborative insights: {str(e)}")
            return []

    def get_feedback_for_reward_adjustment(self, user_id: str, response_id: str) -> Dict[str, Any]:
        """
        Get feedback data for reward system adjustment.

        Args:
            user_id: User ID
            response_id: Response ID

        Returns:
            Dict with feedback-based reward adjustments
        """
        try:
            # Get user's preferences
            preferences = self.preference_store.get(user_id)
            if not preferences:
                return {}

            # Get feedback for this specific response
            response_feedbacks = [
                fb for fb in self.feedback_store.values()
                if fb.user_id == user_id and fb.response_id == response_id
            ]

            adjustments = {}

            # Adjust weights based on user preferences
            if preferences.preference_weights:
                adjustments['weight_adjustments'] = preferences.preference_weights.copy()

            # Apply feedback-based penalties/boosts
            for feedback in response_feedbacks:
                if feedback.feedback_type == FeedbackType.THUMBS_DOWN:
                    adjustments['penalty'] = adjustments.get('penalty', 0.0) + 0.2
                elif feedback.feedback_type == FeedbackType.THUMBS_UP:
                    adjustments['boost'] = adjustments.get('boost', 0.0) + 0.1

                if feedback.rating and feedback.rating < 3:
                    adjustments['rating_penalty'] = adjustments.get('rating_penalty', 0.0) + (5 - feedback.rating) * 0.1

            return adjustments

        except Exception as e:
            logger.error(f"Failed to get feedback for reward adjustment: {str(e)}")
            return {}

    def get_metrics(self) -> FeedbackMetrics:
        """Get current feedback system metrics."""
        return self.metrics

    def _process_feedback_for_preferences(self, feedback: UserFeedback):
        """Process feedback to update user preferences."""
        user_id = feedback.user_id

        # Get or create preference profile
        if user_id not in self.preference_store:
            self.preference_store[user_id] = PreferenceProfile(user_id=user_id)

        preferences = self.preference_store[user_id]

        # Update preference weights based on feedback
        if feedback.category and feedback.rating:
            category_key = feedback.category.value
            current_weight = preferences.preference_weights.get(category_key, 0.5)

            # Adjust weight based on rating (higher rating = higher preference)
            rating_factor = (feedback.rating - 3.0) / 2.0  # -1 to 1 scale
            new_weight = current_weight + rating_factor * self.preference_learning_rate
            new_weight = max(0.0, min(1.0, new_weight))  # Clamp to [0,1]

            preferences.preference_weights[category_key] = new_weight

        # Update preferred styles from comments
        if feedback.comment:
            self._extract_style_preferences(feedback.comment, preferences)

        # Update disliked patterns
        if feedback.feedback_type in [FeedbackType.THUMBS_DOWN, FeedbackType.CORRECTION]:
            if feedback.comment:
                preferences.disliked_patterns.append(feedback.comment[:100])
                # Keep only recent patterns
                if len(preferences.disliked_patterns) > 10:
                    preferences.disliked_patterns = preferences.disliked_patterns[-10:]

        # Update metadata
        preferences.feedback_history.append(str(uuid.uuid4())[:8])
        preferences.last_updated = datetime.now()
        preferences.adaptation_count += 1

    def _update_collaborative_patterns(self, feedback: UserFeedback):
        """Update collaborative learning patterns from feedback."""
        # Create pattern key based on feedback characteristics
        pattern_key = f"{feedback.category.value}_{feedback.feedback_type.value}"

        if pattern_key not in self.collaborative_patterns:
            self.collaborative_patterns[pattern_key] = CollaborativeLearningData(
                pattern_id=pattern_key,
                pattern_type=f"{feedback.category.value}_feedback",
                description=f"Pattern for {feedback.category.value} {feedback.feedback_type.value}",
                confidence_score=0.5
            )

        pattern = self.collaborative_patterns[pattern_key]

        # Add example if we have enough similar feedbacks
        if len(pattern.examples) < 5:
            example = {
                "feedback_type": feedback.feedback_type.value,
                "rating": feedback.rating,
                "comment": feedback.comment[:100] if feedback.comment else None,
                "timestamp": feedback.timestamp.isoformat()
            }
            pattern.examples.append(example)

        # Update usage frequency
        pattern.usage_frequency += 1

        # Increase confidence with more examples
        pattern.confidence_score = min(pattern.usage_frequency / self.collaborative_threshold, 1.0)

        pattern.last_used = datetime.now()

    def _learn_from_correction(self, correction: CorrectionData):
        """Learn from user corrections to improve future responses."""
        # Create collaborative pattern for corrections
        correction_key = f"correction_{correction.correction_type.value}"

        if correction_key not in self.collaborative_patterns:
            self.collaborative_patterns[correction_key] = CollaborativeLearningData(
                pattern_id=correction_key,
                pattern_type="correction_pattern",
                description=f"Correction pattern for {correction.correction_type.value}",
                confidence_score=0.6
            )

        pattern = self.collaborative_patterns[correction_key]

        # Add correction template
        if correction.explanation and len(pattern.correction_templates) < 10:
            pattern.correction_templates.append(correction.explanation[:200])

        pattern.usage_frequency += 1
        pattern.confidence_score = min(pattern.usage_frequency / (self.collaborative_threshold * 2), 1.0)
        pattern.last_used = datetime.now()

    def _calculate_sentiment_score(self, feedbacks: List[UserFeedback]) -> float:
        """Calculate sentiment score from feedbacks (-1 to 1)."""
        if not feedbacks:
            return 0.0

        sentiment_sum = 0.0
        count = 0

        for feedback in feedbacks:
            if feedback.rating:
                # Convert 1-5 rating to -1 to 1 sentiment
                sentiment = (feedback.rating - 3.0) / 2.0
                sentiment_sum += sentiment
                count += 1

            # Analyze comment sentiment (simplified)
            if feedback.comment:
                comment_sentiment = self._analyze_comment_sentiment(feedback.comment)
                sentiment_sum += comment_sentiment
                count += 1

        return sentiment_sum / count if count > 0 else 0.0

    def _analyze_comment_sentiment(self, comment: str) -> float:
        """Simple sentiment analysis for comments."""
        positive_words = ["good", "great", "excellent", "helpful", "useful", "clear", "accurate"]
        negative_words = ["bad", "poor", "wrong", "confusing", "inaccurate", "unhelpful", "unclear"]

        comment_lower = comment.lower()
        positive_count = sum(1 for word in positive_words if word in comment_lower)
        negative_count = sum(1 for word in negative_words if word in comment_lower)

        if positive_count + negative_count == 0:
            return 0.0

        return (positive_count - negative_count) / (positive_count + negative_count)

    def _extract_common_issues(self, feedbacks: List[UserFeedback]) -> List[str]:
        """Extract common issues from feedback comments."""
        issues = []
        comments = [fb.comment for fb in feedbacks if fb.comment]

        # Simple keyword-based issue extraction
        issue_keywords = {
            "inaccurate": ["wrong", "incorrect", "inaccurate", "false"],
            "incomplete": ["missing", "incomplete", "partial", "unfinished"],
            "unclear": ["confusing", "unclear", "vague", "ambiguous"],
            "too long": ["long", "verbose", "too much", "overly detailed"],
            "too short": ["short", "brief", "insufficient", "lacking"]
        }

        for issue_type, keywords in issue_keywords.items():
            mentions = sum(1 for comment in comments
                          if any(keyword in comment.lower() for keyword in keywords))
            if mentions >= len(comments) * 0.3:  # 30% of comments mention this
                issues.append(issue_type)

        return issues

    def _generate_improvement_suggestions(self, feedbacks: List[UserFeedback]) -> List[str]:
        """Generate improvement suggestions based on feedback patterns."""
        suggestions = []

        # Analyze rating patterns
        ratings = [fb.rating for fb in feedbacks if fb.rating]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            if avg_rating < 3.0:
                suggestions.append("Focus on improving response accuracy and helpfulness")
            elif avg_rating < 4.0:
                suggestions.append("Consider adding more detailed explanations")

        # Analyze correction patterns
        corrections = [fb for fb in feedbacks if fb.correction_text]
        if len(corrections) > len(feedbacks) * 0.2:  # 20% corrections
            suggestions.append("Review and improve fact-checking processes")

        return suggestions

    def _extract_style_preferences(self, comment: str, preferences: PreferenceProfile):
        """Extract style preferences from feedback comments."""
        style_indicators = {
            "concise": ["concise", "brief", "short", "to the point"],
            "detailed": ["detailed", "comprehensive", "thorough", "in-depth"],
            "formal": ["formal", "professional", "academic"],
            "casual": ["casual", "conversational", "friendly"],
            "technical": ["technical", "specific", "precise"],
            "simple": ["simple", "easy", "straightforward"]
        }

        comment_lower = comment.lower()
        for style, indicators in style_indicators.items():
            if any(indicator in comment_lower for indicator in indicators):
                if style not in preferences.preferred_styles:
                    preferences.preferred_styles.append(style)

    def _calculate_preference_confidence(self, preferences: PreferenceProfile) -> float:
        """Calculate confidence level for user preferences."""
        factors = []

        # Factor 1: Number of feedback interactions
        feedback_count = len(preferences.feedback_history)
        factors.append(min(feedback_count / 20.0, 1.0))  # Max at 20 feedbacks

        # Factor 2: Adaptation count
        factors.append(min(preferences.adaptation_count / 10.0, 1.0))  # Max at 10 adaptations

        # Factor 3: Preference weight diversity
        weight_count = len(preferences.preference_weights)
        factors.append(min(weight_count / 5.0, 1.0))  # Max at 5 different weights

        return sum(factors) / len(factors) if factors else 0.0

    def _update_metrics(self, feedback: UserFeedback):
        """Update feedback system metrics."""
        self.metrics.total_feedbacks += 1

        # Update category counts
        category = feedback.category
        self.metrics.feedback_categories[category] = self.metrics.feedback_categories.get(category, 0) + 1

        # Update average rating
        if feedback.rating:
            current_total = (self.metrics.average_rating or 0) * (self.metrics.total_feedbacks - 1)
            self.metrics.average_rating = (current_total + feedback.rating) / self.metrics.total_feedbacks

        # Calculate engagement rate (simplified)
        unique_users = len(set(fb.user_id for fb in self.feedback_store.values()))
        self.metrics.user_engagement_rate = unique_users / max(self.metrics.total_feedbacks, 1)

    def get_correction_based_adjustments(self, response: str) -> Dict[str, Any]:
        """
        Get collaborative correction-based adjustments for a response.

        Args:
            response: The response text to analyze

        Returns:
            Dict with correction-based penalties for different aspects
        """
        try:
            adjustments = {}

            # Check for common correction patterns from stored corrections
            for correction in self.correction_store.values():
                if correction.corrected_response and correction.correction_type:
                    # Simple text similarity check (could be improved with better NLP)
                    response_lower = response.lower()
                    corrected_lower = correction.corrected_response.lower()

                    # Check if this correction pattern applies
                    if self._has_correction_pattern(response_lower, corrected_lower, correction.correction_type):
                        aspect = self._get_aspect_from_correction_type(correction.correction_type)
                        if aspect:
                            adjustments[aspect] = adjustments.get(aspect, 0.0) + 0.1

            return adjustments

        except Exception as e:
            logger.error(f"Failed to get correction-based adjustments: {str(e)}")
            return {}

    def _has_correction_pattern(self, response: str, corrected: str, correction_type: CorrectionType) -> bool:
        """Check if a correction pattern applies to the response."""
        # Simple pattern matching - could be enhanced with ML
        if correction_type == CorrectionType.FACTUAL_ERROR:
            # Look for factual inconsistencies
            return len(response.split()) > len(corrected.split()) * 1.5  # Overly verbose responses
        elif correction_type == CorrectionType.INCOMPLETE_INFO:
            return len(response.split()) < 5  # Too short responses
        elif correction_type == CorrectionType.LOGICAL_ERROR:
            # Look for logical connectors that might indicate errors
            return "but" in response.lower() and "however" in response.lower()
        return False

    def _get_aspect_from_correction_type(self, correction_type: CorrectionType) -> str:
        """Map correction type to response aspect."""
        mapping = {
            CorrectionType.FACTUAL_ERROR: 'factuality',
            CorrectionType.LOGICAL_ERROR: 'coherence',
            CorrectionType.STYLE_IMPROVEMENT: 'creativity',
            CorrectionType.INCOMPLETE_INFO: 'coherence',
            CorrectionType.BETTER_ALTERNATIVE: 'usefulness',
            CorrectionType.FORMAT_ISSUE: 'coherence'
        }
        return mapping.get(correction_type)