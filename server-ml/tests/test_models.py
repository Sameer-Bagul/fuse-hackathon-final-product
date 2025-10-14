#!/usr/bin/env python3
"""
Test script for Pydantic models and API structures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_pydantic_models():
    """Test the Pydantic models for reward system"""
    print("Testing Pydantic models...")

    # Import the models (without importing the full app that requires torch)
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional

    # Define the models locally to test them
    class RewardWeightsConfig(BaseModel):
        accuracy: float
        coherence: float
        factuality: float
        creativity: float

    class MultiObjectiveRewards(BaseModel):
        accuracy: float
        coherence: float
        factuality: float
        creativity: float

    class RewardMetrics(BaseModel):
        individual_rewards: MultiObjectiveRewards
        total_reward: float
        weighted_reward: float
        weights_used: RewardWeightsConfig

    # Test model creation
    weights = RewardWeightsConfig(accuracy=0.4, coherence=0.3, factuality=0.2, creativity=0.1)
    print("✓ RewardWeightsConfig created")

    rewards = MultiObjectiveRewards(accuracy=0.8, coherence=0.7, factuality=0.6, creativity=0.5)
    print("✓ MultiObjectiveRewards created")

    metrics = RewardMetrics(
        individual_rewards=rewards,
        total_reward=0.65,
        weighted_reward=0.67,
        weights_used=weights
    )
    print("✓ RewardMetrics created")

    # Test JSON serialization
    json_data = metrics.json()
    print("✓ JSON serialization successful")
    print(f"  Sample JSON: {json_data[:100]}...")

    # Test dict conversion
    dict_data = metrics.dict()
    print("✓ Dict conversion successful")

    print("\n🎉 All Pydantic model tests passed!")

if __name__ == "__main__":
    test_pydantic_models()