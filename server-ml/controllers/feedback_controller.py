"""
Feedback Controller for User Feedback Integration

This controller provides API endpoints for collecting, processing, and analyzing
user feedback on AI responses, including ratings, corrections, and preference learning.
"""

import sys
import os
import logging
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.feedback_service import FeedbackService
from models.feedback import (
    SubmitFeedbackRequest,
    SubmitFeedbackResponse,
    GetFeedbackHistoryRequest,
    GetFeedbackHistoryResponse,
    SubmitCorrectionRequest,
    SubmitCorrectionResponse,
    GetPreferencesRequest,
    GetPreferencesResponse,
    RateResponseRequest,
    RateResponseResponse,
    SubmitErrorReportRequest,
    SubmitErrorReportResponse,
    FeedbackType
)

logger = logging.getLogger(__name__)


class FeedbackController:
    """
    Controller for handling user feedback operations.

    Provides endpoints for feedback submission, correction handling,
    preference management, and collaborative learning.
    """

    def __init__(self, feedback_service: Optional[FeedbackService] = None):
        """
        Initialize the feedback controller.

        Args:
            feedback_service: Feedback service instance (created if None)
        """
        try:
            self.feedback_service = feedback_service or FeedbackService()
            logger.info("FeedbackController initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FeedbackController: {str(e)}")
            raise

    def submit_feedback(self, request: SubmitFeedbackRequest) -> SubmitFeedbackResponse:
        """
        Submit user feedback on an AI response.

        Args:
            request: Feedback submission request

        Returns:
            SubmitFeedbackResponse: Response with submission status
        """
        try:
            logger.info(f"[Feedback API] Submitting feedback: user={request.user_id}, "
                       f"type={request.feedback_type.value}, category={request.category.value}")

            response = self.feedback_service.submit_feedback(request)

            if response.success:
                logger.info(f"[Feedback API] Feedback submitted successfully: id={response.feedback_id}")
            else:
                logger.warning(f"[Feedback API] Feedback submission failed: {response.message}")

            return response

        except Exception as e:
            logger.error(f"[Feedback API] Failed to submit feedback: {str(e)}")
            return SubmitFeedbackResponse(
                feedback_id="",
                success=False,
                message=f"Failed to submit feedback: {str(e)}",
                processed_immediately=False
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
            logger.info(f"[Feedback API] Retrieving feedback history: user={request.user_id}, "
                       f"limit={request.limit}, offset={request.offset}")

            response = self.feedback_service.get_feedback_history(request)

            logger.info(f"[Feedback API] Retrieved {len(response.feedbacks)} feedback items "
                       f"for user {request.user_id}")

            return response

        except Exception as e:
            logger.error(f"[Feedback API] Failed to get feedback history: {str(e)}")
            return GetFeedbackHistoryResponse(
                user_id=request.user_id,
                feedbacks=[],
                total_count=0,
                has_more=False
            )

    def submit_correction(self, request: SubmitCorrectionRequest) -> SubmitCorrectionResponse:
        """
        Submit a correction for a wrong AI response.

        Args:
            request: Correction submission request

        Returns:
            SubmitCorrectionResponse: Response with correction status
        """
        try:
            logger.info(f"[Feedback API] Submitting correction: user={request.user_id}, "
                       f"type={request.correction_type.value}")

            response = self.feedback_service.submit_correction(request)

            if response.success:
                logger.info(f"[Feedback API] Correction submitted successfully: id={response.correction_id}")
            else:
                logger.warning(f"[Feedback API] Correction submission failed: {response.message}")

            return response

        except Exception as e:
            logger.error(f"[Feedback API] Failed to submit correction: {str(e)}")
            return SubmitCorrectionResponse(
                correction_id="",
                success=False,
                message=f"Failed to submit correction: {str(e)}",
                validation_status="failed"
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
            logger.info(f"[Feedback API] Retrieving preferences: user={request.user_id}")

            response = self.feedback_service.get_user_preferences(request)

            logger.info(f"[Feedback API] Retrieved preferences for user {request.user_id}: "
                       f"confidence={response.confidence_level:.3f}")

            return response

        except Exception as e:
            logger.error(f"[Feedback API] Failed to get user preferences: {str(e)}")
            # Return default preferences on error
            from models.feedback import PreferenceProfile
            default_prefs = PreferenceProfile(user_id=request.user_id)
            return GetPreferencesResponse(
                user_id=request.user_id,
                preferences=default_prefs,
                last_updated=default_prefs.last_updated,
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
            logger.info(f"[Feedback API] Rating response: user={request.user_id}, "
                       f"response_id={request.response_id}, rating={request.rating}")

            response = self.feedback_service.rate_response(request)

            if response.success:
                logger.info(f"[Feedback API] Response rated successfully, "
                           f"preferences_updated={response.updated_preferences}")
            else:
                logger.warning(f"[Feedback API] Response rating failed: {response.message}")

            return response

        except Exception as e:
            logger.error(f"[Feedback API] Failed to rate response: {str(e)}")
            return RateResponseResponse(
                success=False,
                message=f"Failed to rate response: {str(e)}",
                updated_preferences=False
            )

    def get_feedback_analytics(self, user_id: Optional[str] = None, response_id: Optional[str] = None):
        """
        Get feedback analytics and insights.

        Args:
            user_id: Optional user ID filter
            response_id: Optional response ID for analysis

        Returns:
            Dict with feedback analytics
        """
        try:
            logger.info(f"[Feedback API] Getting feedback analytics: user={user_id}, response={response_id}")

            analytics = {
                'metrics': self.feedback_service.get_metrics(),
                'timestamp': os.times()[4] if hasattr(os, 'times') else 0
            }

            # Add response-specific analysis if requested
            if response_id:
                analysis = self.feedback_service.analyze_feedback_patterns(response_id)
                analytics['response_analysis'] = {
                    'response_id': response_id,
                    'total_feedbacks': analysis.total_feedbacks,
                    'average_rating': analysis.average_rating,
                    'sentiment_score': analysis.sentiment_score,
                    'common_issues': analysis.common_issues,
                    'improvement_suggestions': analysis.improvement_suggestions
                }

            # Add collaborative insights
            analytics['collaborative_insights'] = self.feedback_service.get_collaborative_insights()

            logger.info(f"[Feedback API] Retrieved feedback analytics: "
                       f"{analytics['metrics']['total_feedbacks']} total feedbacks")

            return analytics

        except Exception as e:
            logger.error(f"[Feedback API] Failed to get feedback analytics: {str(e)}")
            return {
                'error': str(e),
                'metrics': self.feedback_service.get_metrics() if self.feedback_service else {},
                'timestamp': os.times()[4] if hasattr(os, 'times') else 0
            }

    def submit_error_report(self, request: SubmitErrorReportRequest) -> SubmitErrorReportResponse:
        """
        Submit an error report from the frontend.

        Args:
            request: Error report submission request

        Returns:
            SubmitErrorReportResponse: Response with submission status
        """
        try:
            logger.info(f"[Feedback API] Submitting error report: user={request.user_id}, "
                       f"severity={request.severity}, component={request.component}")

            # For now, we'll store error reports as feedback with ERROR_REPORT type
            # In a real implementation, you might want a separate error reporting system
            feedback_request = SubmitFeedbackRequest(
                user_id=request.user_id,
                session_id=request.session_id,
                prompt_id="error_report",  # Use a placeholder
                response_id="error_report",  # Use a placeholder
                feedback_type=FeedbackType.ERROR_REPORT,
                category=FeedbackCategory.USEFULNESS,  # Default category
                comment=f"Error: {request.error_message}\n\nStack Trace:\n{request.stack_trace or 'N/A'}\n\nContext:\n{request.context or 'N/A'}\n\nComponent: {request.component or 'N/A'}\nSeverity: {request.severity}"
            )

            feedback_response = self.feedback_service.submit_feedback(feedback_request)

            if feedback_response.success:
                logger.info(f"[Feedback API] Error report submitted successfully: id={feedback_response.feedback_id}")
                return SubmitErrorReportResponse(
                    error_report_id=feedback_response.feedback_id,
                    success=True,
                    message="Error report submitted successfully",
                    processed_immediately=feedback_response.processed_immediately
                )
            else:
                logger.warning(f"[Feedback API] Error report submission failed: {feedback_response.message}")
                return SubmitErrorReportResponse(
                    error_report_id="",
                    success=False,
                    message=feedback_response.message,
                    processed_immediately=False
                )

        except Exception as e:
            logger.error(f"[Feedback API] Failed to submit error report: {str(e)}")
            return SubmitErrorReportResponse(
                error_report_id="",
                success=False,
                message=f"Failed to submit error report: {str(e)}",
                processed_immediately=False
            )

    def get_feedback_insights(self, category_filter: Optional[str] = None):
        """
        Get feedback insights and recommendations.

        Args:
            category_filter: Optional category to filter insights

        Returns:
            Dict with feedback insights
        """
        try:
            logger.info(f"[Feedback API] Getting feedback insights: category={category_filter}")

            insights = {
                'collaborative_patterns': self.feedback_service.get_collaborative_insights(),
                'system_metrics': self.feedback_service.get_metrics(),
                'recommendations': []
            }

            # Generate recommendations based on feedback patterns
            metrics = insights['system_metrics']
            if metrics.total_feedbacks > 0:
                if metrics.user_engagement_rate < 0.1:
                    insights['recommendations'].append({
                        'type': 'engagement',
                        'priority': 'high',
                        'message': 'Low user engagement with feedback system. Consider improving feedback collection UX.'
                    })

                if metrics.correction_adoption_rate > 0.3:
                    insights['recommendations'].append({
                        'type': 'learning',
                        'priority': 'medium',
                        'message': 'High correction rate indicates potential issues with response quality.'
                    })

                if metrics.preference_learning_accuracy > 0.8:
                    insights['recommendations'].append({
                        'type': 'personalization',
                        'priority': 'low',
                        'message': 'Strong preference learning detected. Personalization is working well.'
                    })

            logger.info(f"[Feedback API] Generated {len(insights['recommendations'])} recommendations")

            return insights

        except Exception as e:
            logger.error(f"[Feedback API] Failed to get feedback insights: {str(e)}")
            return {
                'error': str(e),
                'collaborative_patterns': [],
                'system_metrics': {},
                'recommendations': []
            }