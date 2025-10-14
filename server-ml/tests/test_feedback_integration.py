#!/usr/bin/env python3
"""
Test script for User Feedback Integration

This script tests the basic functionality of the feedback integration system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from services.feedback_service import FeedbackService
from models.feedback import (
    SubmitFeedbackRequest, FeedbackType, FeedbackCategory,
    SubmitCorrectionRequest, CorrectionType
)

def test_feedback_service():
    """Test basic feedback service functionality"""
    print("Testing Feedback Service...")

    # Initialize service
    feedback_service = FeedbackService()

    # Test submitting feedback
    request = SubmitFeedbackRequest(
        user_id="test_user_123",
        prompt_id="test_prompt_789",
        response_id="test_response_456",
        feedback_type=FeedbackType.RATING,
        category=FeedbackCategory.ACCURACY,
        rating=4.5,
        comment="Good response, very accurate information"
    )

    response = feedback_service.submit_feedback(request)
    print(f"✓ Feedback submission: {response.success}")
    assert response.success, "Feedback submission should succeed"

    # Test getting feedback history
    from models.feedback import GetFeedbackHistoryRequest
    history_request = GetFeedbackHistoryRequest(
        user_id="test_user_123",
        limit=10
    )

    history = feedback_service.get_feedback_history(history_request)
    print(f"✓ Feedback history retrieval: {len(history.feedbacks)} feedbacks found")
    assert len(history.feedbacks) == 1, "Should have 1 feedback in history"

    # Test submitting correction
    correction_request = SubmitCorrectionRequest(
        user_id="test_user_123",
        prompt_id="test_prompt_789",
        response_id="test_response_456",
        corrected_response="This is the corrected and improved response with better accuracy.",
        correction_type=CorrectionType.FACTUAL_ERROR,
        explanation="The original response contained factual inaccuracies"
    )

    correction_response = feedback_service.submit_correction(correction_request)
    print(f"✓ Correction submission: {correction_response.success}")
    assert correction_response.success, "Correction submission should succeed"

    # Test getting user preferences
    from models.feedback import GetPreferencesRequest
    prefs_request = GetPreferencesRequest(
        user_id="test_user_123",
        include_history=False
    )

    preferences = feedback_service.get_user_preferences(prefs_request)
    print(f"✓ User preferences retrieval: user_id={preferences.user_id}")
    assert preferences.user_id == "test_user_123", "Should return correct user preferences"

    # Test collaborative insights
    insights = feedback_service.get_collaborative_insights()
    print(f"✓ Collaborative insights: {len(insights)} patterns found")
    # May be empty initially, which is fine

    # Test feedback analysis
    analysis = feedback_service.analyze_feedback_patterns("test_response_456")
    print(f"✓ Feedback analysis: {analysis.total_feedbacks} feedbacks analyzed")
    assert analysis.total_feedbacks >= 1, "Should have at least 1 feedback for analysis"

    print("✅ All feedback service tests passed!")

def test_feedback_integration():
    """Test integration with other services"""
    print("\nTesting Feedback Integration...")

    # Test that feedback service can be imported and used
    from services.reward_service import RewardService
    from services.meta_learning_service import MetaLearningService
    from services.analytics_service import AnalyticsService

    # Create services
    feedback_service = FeedbackService()
    reward_service = RewardService(feedback_service=feedback_service)

    print("✓ Services created successfully with feedback integration")

    # Test reward calculation with user feedback
    reward = reward_service.calculate_multi_objective_reward(
        prompt="Test prompt",
        response="Test response",
        action=1,
        user_id="test_user_123",
        response_id="test_response_456"
    )

    print(f"✓ Personalized reward calculation: {reward.get('personalized', False)}")
    assert 'personalized' in reward, "Reward should indicate if it was personalized"

    print("✅ Feedback integration tests passed!")

if __name__ == "__main__":
    try:
        test_feedback_service()
        test_feedback_integration()
        print("\n🎉 All feedback integration tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)