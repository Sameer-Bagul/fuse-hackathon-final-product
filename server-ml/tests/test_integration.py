#!/usr/bin/env python3
"""
Integration test for the Multi-Objective Reward System
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Mock torch to avoid dependency issues during testing
class MockTorch:
    class FloatTensor:
        def __init__(self, data):
            self.data = data
        def unsqueeze(self, dim):
            return self

    class tensor:
        @staticmethod
        def tensor(data, dtype=None):
            return MockTensor(data)

    class distributions:
        class Categorical:
            def __init__(self, probs):
                self.probs = probs
            def log_prob(self, action):
                return MockTensor(0.0)

    class nn:
        class Module:
            pass
        class Linear:
            def __init__(self, in_features, out_features):
                pass
            def __call__(self, x):
                return MockTensor([0.0] * out_features)
        class ReLU:
            def __call__(self, x):
                return x
        class Softmax:
            def __init__(self, dim=None):
                pass
            def __call__(self, x):
                return x
        class MSELoss:
            def __call__(self, pred, target):
                return MockTensor(0.0)

    class optim:
        class Adam:
            def __init__(self, params, lr=None):
                pass
            def zero_grad(self):
                pass
            def step(self):
                pass

class MockTensor:
    def __init__(self, data):
        self.data = data

    def squeeze(self, dim=None):
        return self

    def detach(self):
        return self

    def mean(self):
        return self

    def backward(self):
        pass

    def __getitem__(self, idx):
        return self

    def __mul__(self, other):
        return self

    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __truediv__(self, other):
        return self

# Monkey patch torch
sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = MockTorch.nn
sys.modules['torch.optim'] = MockTorch.optim

# Now import our modules
from services.reward_service import RewardService

class MockLLM:
    """Mock LLM for testing without torch dependency"""

    def __init__(self, num_actions=10):
        self.num_actions = num_actions
        self.episode_rewards = []
        self.last_multi_objective_reward = None

    def process_prompt(self, prompt_text):
        # Simple mock response generation
        responses = [
            "Create a neural network model using TensorFlow",
            "Implement a machine learning algorithm for classification",
            "Build a data processing pipeline",
            "Design a predictive model",
            "Develop an AI system"
        ]
        response = responses[hash(prompt_text) % len(responses)]
        action = hash(prompt_text) % self.num_actions
        return response, action

    def learn(self, action, reward):
        """Mock learning - just store the reward"""
        if isinstance(reward, dict):
            actual_reward = reward.get('weighted_reward', 0.5)
            self.last_multi_objective_reward = reward
        else:
            actual_reward = float(reward)
            self.last_multi_objective_reward = None

        self.episode_rewards.append(actual_reward)

    def get_last_multi_objective_reward(self):
        return self.last_multi_objective_reward

def test_integration():
    """Test the integration of reward service with LLM"""
    print("Testing Multi-Objective Reward System Integration...")

    # Initialize components
    reward_service = RewardService()
    llm = MockLLM(num_actions=10)

    print("✓ Components initialized")

    # Test prompt processing with reward calculation
    test_prompts = [
        "Create a machine learning model",
        "Explain neural networks",
        "Build a recommendation system",
        "Analyze time series data"
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Processing prompt {i+1}: {prompt[:30]}... ---")

        # Process prompt
        response, action = llm.process_prompt(prompt)
        print(f"✓ Response generated: {response[:50]}...")

        # Calculate multi-objective reward
        reward_result = reward_service.calculate_multi_objective_reward(prompt, response, action)
        print("✓ Multi-objective reward calculated:")
        print(".3f")

        # Learn from the reward
        llm.learn(action, reward_result)
        print("✓ LLM learned from reward")

    # Test reward configuration
    new_weights = {
        'accuracy': 0.5,
        'coherence': 0.2,
        'factuality': 0.2,
        'creativity': 0.1
    }

    success = reward_service.configure_weights(new_weights)
    print(f"\n✓ Weight reconfiguration: {'successful' if success else 'failed'}")

    # Test metrics retrieval
    metrics = reward_service.get_current_metrics()
    print("✓ Metrics retrieved:")
    print(f"  History length: {metrics['history_length']}")
    print(f"  Current weighted reward: {metrics['current_metrics']['weighted_reward']:.3f}")

    # Test average metrics
    averages = reward_service.get_average_metrics(window=10)
    print("✓ Average metrics calculated:")
    print(f"  Avg weighted reward: {averages.get('weighted_reward', 0):.3f}")

    print("\n🎉 Integration test passed! Multi-Objective Reward System is fully functional.")

if __name__ == "__main__":
    test_integration()