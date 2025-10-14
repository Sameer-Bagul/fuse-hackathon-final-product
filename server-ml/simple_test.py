#!/usr/bin/env python3

# Simple test to verify the refactored MVC architecture works

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.learner import LLM
from models.prompt import Prompt
from models.history import History
from models.task_generator import TaskGenerator
from controllers.evaluator import Evaluator
from controllers.scheduler import Scheduler
from controllers.prompt_controller import PromptController
from view.visualizer import Visualizer

def test_basic_functionality():
    print("Testing basic functionality...")

    # Initialize components
    llm = LLM(num_actions=5)
    history = History()
    task_gen = TaskGenerator(num_actions=5)
    evaluator = Evaluator(history)
    scheduler = Scheduler(task_gen)
    prompt_ctrl = PromptController(llm, history)
    visualizer = Visualizer()

    # Test prompt handling
    prompt_text = "Solve this math problem"
    response = prompt_ctrl.handle_prompt(prompt_text)
    print(f"Prompt: {prompt_text}")
    print(f"Response: {response}")

    # Test task generation
    prompt = Prompt("Generate a task for learning")
    tasks = scheduler.schedule(3, prompt)
    print(f"Generated {len(tasks)} tasks")

    # Test evaluation
    if tasks:
        avg_reward = evaluator.evaluate(llm, tasks[0], prompt, num_episodes=10)
        print(f"Average reward: {avg_reward}")

    # Test metrics
    metrics = prompt_ctrl.get_learning_metrics()
    print(f"Learning metrics: {metrics}")

    # Test visualization data
    chart_data = visualizer.get_chartjs_data([0.1, 0.2, 0.3], metrics)
    print("Chart.js data generated successfully")

    print("All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()