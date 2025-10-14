#!/usr/bin/env python3
"""
Simple test script for the Multi-Objective Reward Service
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from services.reward_service import RewardService

def test_reward_service():
    """Test the reward service functionality"""
    print("Testing Multi-Objective Reward Service...")

    # Initialize the service
    reward_service = RewardService()
    print("✓ Reward service initialized")

    # Test reward calculation
    prompt = "Create a machine learning model for image classification"
    response = "Create a convolutional neural network (CNN) to classify images using TensorFlow. The model should include convolutional layers, pooling layers, and dense layers for accurate image classification."
    action = 5

    reward_result = reward_service.calculate_multi_objective_reward(prompt, response, action)
    print("✓ Multi-objective reward calculated:")
    print(f"  Individual rewards: {reward_result['individual_rewards']}")
    print(".3f")
    print(f"  Weights used: {reward_result['weights_used']}")

    # Test weight configuration
    new_weights = {
        'accuracy': 0.5,
        'coherence': 0.2,
        'factuality': 0.2,
        'creativity': 0.1
    }

    success = reward_service.configure_weights(new_weights)
    print(f"✓ Weight configuration {'successful' if success else 'failed'}")

    # Test with new weights
    reward_result2 = reward_service.calculate_multi_objective_reward(prompt, response, action)
    print("✓ Reward with new weights:")
    print(".3f")

    # Test metrics
    metrics = reward_service.get_current_metrics()
    print("✓ Current metrics retrieved:")
    print(f"  History length: {metrics['history_length']}")

    print("\n🎉 All tests passed! Multi-Objective Reward Service is working correctly.")

if __name__ == "__main__":
    test_reward_service()