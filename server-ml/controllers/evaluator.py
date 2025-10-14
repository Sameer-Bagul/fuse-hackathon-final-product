import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.history import History

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, history=None):
        try:
            self.history = history or History()
            logger.info("Evaluator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Evaluator: {str(e)}")
            raise

    def evaluate(self, llm, task, prompt=None, num_episodes=100):
        try:
            if llm is None:
                raise ValueError("LLM instance is required for evaluation")

            if task is None:
                raise ValueError("Task is required for evaluation")

            if not isinstance(num_episodes, int) or num_episodes <= 0:
                raise ValueError("num_episodes must be a positive integer")

            if num_episodes > 10000:
                raise ValueError("num_episodes cannot exceed 10,000 for safety reasons")

            logger.info(f"Starting evaluation with {num_episodes} episodes")

            total_reward = 0
            for episode in range(num_episodes):
                try:
                    if prompt:
                        response, action = llm.process_prompt(prompt.text)
                        # Simulate reward based on action quality
                        reward = self._simulate_reward(action)
                    else:
                        action = llm.choose_action()
                        if isinstance(task, list) and len(task) > action:
                            reward = task[action]
                        else:
                            raise ValueError(f"Invalid task format or action index: {action}")

                    llm.learn(action, reward)
                    total_reward += reward

                    if prompt:
                        self.history.add_interaction(prompt, response, reward)

                except Exception as e:
                    logger.warning(f"Error in episode {episode}: {str(e)}")
                    # Continue with next episode rather than failing completely
                    continue

            # Adapt LLM based on history
            try:
                llm.adapt_from_history()
            except Exception as e:
                logger.warning(f"Failed to adapt LLM from history: {str(e)}")
                # Don't fail the evaluation for adaptation issues

            average_reward = total_reward / num_episodes
            logger.info(f"Evaluation completed. Average reward: {average_reward:.3f}")
            return average_reward

        except ValueError as e:
            logger.warning(f"Validation error in evaluate: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in evaluate: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")

    def _simulate_reward(self, action):
        try:
            # Simple reward simulation based on action
            # Assume action 0 is suboptimal, others are better
            if action == 0:
                return 0.3
            else:
                return 0.7
        except Exception as e:
            logger.warning(f"Error in reward simulation: {str(e)}")
            return 0.5  # Default reward