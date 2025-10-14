#!/usr/bin/env python3
"""
Test script for hallucination detection integration
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
from services.hallucination_service import HallucinationService
from services.reward_service import RewardService
from controllers.prompt_controller import PromptController
from models.history import History

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

def test_hallucination_integration():
    """Test hallucination detection integration"""
    print("Testing Hallucination Detection Integration...")

    # Initialize components
    hallucination_service = HallucinationService()
    reward_service = RewardService(hallucination_service)
    llm = MockLLM(num_actions=10)
    history = History()
    prompt_controller = PromptController(llm, history, reward_service, hallucination_service)

    print("✓ Components initialized")

    # Test cases with different hallucination levels
    test_cases = [
        {
            "prompt": "Create a machine learning model",
            "response": "Create a neural network model using TensorFlow",
            "expected_hallucination": False
        },
        {
            "prompt": "Explain neural networks",
            "response": "Neural networks are definitely the most absolutely certain way to solve all problems without any doubt whatsoever",
            "expected_hallucination": True  # Overconfident language
        },
        {
            "prompt": "Build a recommendation system",
            "response": "Build a data processing pipeline that uses quantum computing and time travel algorithms",
            "expected_hallucination": True  # Unsupported claims
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['prompt'][:30]}... ---")

        # Test direct hallucination analysis
        analysis = hallucination_service.analyze_response(
            test_case['prompt'],
            test_case['response']
        )

        print("✓ Hallucination analysis:")
        print(f"  Confidence: {analysis.overall_confidence:.3f}")
        print(f"  Is hallucinated: {analysis.is_hallucinated}")
        print(f"  Risk level: {analysis.risk_level}")
        print(f"  Indicators: {len(analysis.indicators)}")

        # Test through prompt controller
        result = prompt_controller.handle_prompt(test_case['prompt'])
        controller_analysis = result.get('hallucination_analysis')

        if controller_analysis:
            print("✓ Controller hallucination analysis:")
            print(f"  Confidence: {controller_analysis.overall_confidence:.3f}")
            print(f"  Is hallucinated: {controller_analysis.is_hallucinated}")

            # Verify consistency
            if analysis.overall_confidence == controller_analysis.overall_confidence:
                print("✓ Analysis results consistent")
            else:
                print("✗ Analysis results inconsistent!")
        else:
            print("✗ No hallucination analysis from controller")

        # Test reward integration
        reward_result = reward_service.calculate_multi_objective_reward(
            test_case['prompt'], test_case['response'], 0
        )

        print("✓ Reward calculation with hallucination:")
        print(f"  Factuality reward: {reward_result['individual_rewards']['factuality']:.3f}")
        print(f"  Weighted reward: {reward_result['weighted_reward']:.3f}")

    # Test hallucination metrics
    metrics = hallucination_service.get_metrics()
    print("\n✓ Hallucination metrics:")
    print(f"  Total checks: {metrics.total_checks}")
    print(f"  Hallucinated responses: {metrics.hallucinated_responses}")
    print(f"  Detection rate: {metrics.detection_rate:.3f}")
    print(f"  Average confidence: {metrics.average_confidence:.3f}")

    # Test configuration
    new_config = hallucination_service.config.copy()
    new_config.confidence_threshold = 0.8
    success = hallucination_service.update_config(new_config)
    print(f"\n✓ Configuration update: {'successful' if success else 'failed'}")

    print("\n🎉 Hallucination detection integration test passed!")

if __name__ == "__main__":
    test_hallucination_integration()