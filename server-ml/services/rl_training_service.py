import asyncio
import threading
import time
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from models.ppo_agent import PPOAgent
from models.curriculum_generator import CurriculumGenerator
from models.curriculum import CurriculumTree, CurriculumProgress, DifficultyLevel
from models.task_generator import TaskGenerator
from models.prompt import Prompt
from controllers.prompt_controller import PromptController
from services.reward_service import RewardService
from utils.logging_config import get_logger

logger = get_logger(__name__)

class RLTrainingService:
    """Service for curriculum-driven RL training with PPO"""

    def __init__(self, prompt_controller: PromptController,
                 reward_service: Optional[RewardService] = None,
                 state_dim: int = 10, action_dim: int = 10,
                 curriculum_generator: Optional[CurriculumGenerator] = None):
        self.prompt_controller = prompt_controller
        self.reward_service = reward_service
        self.curriculum_generator = curriculum_generator or CurriculumGenerator()

        # PPO Agent
        self.ppo_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            epsilon=0.2
        )

        # Curriculum components
        self.curriculum_tree = CurriculumTree.create_default_curriculum()
        self.task_generator = TaskGenerator(curriculum_tree=self.curriculum_tree)

        # Training state
        self.is_training = False
        self.training_thread = None
        self.current_learner_id = "rl_trainer"
        self.learner_progress = CurriculumProgress(self.current_learner_id)

        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'episode_rewards': [],
            'curriculum_progress': [],
            'policy_updates': 0,
            'success_rate': [],
            'learning_curves': []
        }

        # Training configuration
        self.config = {
            'max_episodes': 1000,
            'episode_length': 50,
            'update_frequency': 10,  # Update policy every N steps
            'curriculum_check_frequency': 25,  # Check curriculum progress every N episodes
            'save_frequency': 100,  # Save model every N episodes
            'model_path': 'models/ppo_curriculum_agent.pth'
        }

        logger.info("RL Training Service initialized")

    def start_training(self):
        """Start the RL training loop"""
        if self.is_training:
            logger.warning("RL training is already running")
            return

        logger.info("🚀 Starting curriculum-driven RL training...")

        self.is_training = True
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

        logger.info("✅ RL training started successfully")

    def stop_training(self):
        """Stop the RL training loop"""
        if not self.is_training:
            logger.info("RL training is not running")
            return

        logger.info("🛑 Stopping RL training...")

        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=10)

        # Save final model
        self._save_model()

        logger.info("✅ RL training stopped")

    def _training_loop(self):
        """Main RL training loop with curriculum integration"""
        logger.info("Starting curriculum-driven PPO training loop")

        episode = 0
        total_steps = 0

        while self.is_training and episode < self.config['max_episodes']:
            episode += 1
            episode_reward = 0
            episode_steps = 0
            episode_successes = 0

            # Reset episode state
            self._reset_episode()

            logger.info(f"Episode {episode} started")

            # Generate initial curriculum prompt
            current_prompt = self._generate_curriculum_prompt()
            if not current_prompt:
                logger.warning(f"No curriculum prompt available for episode {episode}")
                time.sleep(1)
                continue

            # Episode loop
            for step in range(self.config['episode_length']):
                if not self.is_training:
                    break

                # Get current state
                state = self._get_current_state(current_prompt)

                # Select action using PPO
                action, log_prob, value = self.ppo_agent.select_action(state)

                # Execute action (process prompt with selected action strategy)
                reward, next_prompt, done, success = self._execute_action(action, current_prompt)

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if success:
                    episode_successes += 1

                # Get next state
                next_state = self._get_current_state(next_prompt) if not done else np.zeros_like(state)

                # Store transition
                self.ppo_agent.store_transition(state, action, log_prob, reward, value, done)

                # Update curriculum progress
                if success:
                    self._update_curriculum_progress(current_prompt, reward)

                # Check for policy update
                if total_steps % self.config['update_frequency'] == 0:
                    self.ppo_agent.update_policy()
                    self.training_stats['policy_updates'] += 1

                current_prompt = next_prompt

                if done:
                    break

                time.sleep(0.1)  # Small delay between steps

            # Episode completed
            self._end_episode(episode, episode_reward, episode_steps, episode_successes)

            # Curriculum adaptation check
            if episode % self.config['curriculum_check_frequency'] == 0:
                self._adapt_curriculum()

            # Save model periodically
            if episode % self.config['save_frequency'] == 0:
                self._save_model()

        logger.info("Training loop completed")

    def _reset_episode(self):
        """Reset episode-specific state"""
        # Reset any episode-specific variables
        pass

    def _generate_curriculum_prompt(self) -> Optional[Prompt]:
        """Generate a curriculum-aware prompt for training"""
        try:
            # Get available skills based on current progress
            available_skills = self.learner_progress.get_available_skills(self.curriculum_tree)

            if not available_skills:
                # Reset to basic skills if no progress
                available_skills = [skill_id for skill_id, skill in self.curriculum_tree.skills.items()
                                  if not skill.prerequisites]

            if not available_skills:
                return None

            # Select skill based on curriculum progress
            target_skill_id = self._select_curriculum_skill(available_skills)
            target_skill = self.curriculum_tree.skills[target_skill_id]

            # Generate prompt for the skill
            prompt_text = self._generate_skill_prompt(target_skill)

            # Update PPO curriculum state
            difficulty_value = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}[target_skill.difficulty.value]
            self.ppo_agent.update_curriculum_state(
                target_skill_id,
                difficulty_value,
                self.learner_progress.get_skill_mastery_level(target_skill_id),
                False  # Reset success streak for new episode
            )

            return Prompt(prompt_text)

        except Exception as e:
            logger.error(f"Failed to generate curriculum prompt: {str(e)}")
            return None

    def _select_curriculum_skill(self, available_skills: List[str]) -> str:
        """Select a skill for the current training episode"""
        if not available_skills:
            return random.choice(list(self.curriculum_tree.skills.keys()))

        # Weight skills by mastery level (prefer less mastered skills)
        skill_weights = []
        for skill_id in available_skills:
            mastery = self.learner_progress.get_skill_mastery_level(skill_id)
            # Lower mastery = higher weight
            weight = max(0.1, 1.0 - mastery)
            skill_weights.append(weight)

        # Normalize weights
        total_weight = sum(skill_weights)
        if total_weight == 0:
            return random.choice(available_skills)

        skill_weights = [w / total_weight for w in skill_weights]

        # Sample skill
        r = random.random()
        cumulative = 0
        for i, weight in enumerate(skill_weights):
            cumulative += weight
            if r <= cumulative:
                return available_skills[i]

        return available_skills[-1]

    def _generate_skill_prompt(self, skill) -> str:
        """Generate a training prompt for a specific skill"""
        skill_templates = {
            "python_basics": [
                "Write a Python function to calculate the sum of numbers from 1 to n",
                "Create a Python script that checks if a number is prime",
                "Implement a Python function to reverse a string"
            ],
            "data_structures": [
                "Implement a stack data structure with push and pop operations",
                "Create a Python function to find duplicates in a list",
                "Write code to implement a simple queue using lists"
            ],
            "numpy_basics": [
                "Use NumPy to create a 2D array and perform matrix multiplication",
                "Write code to calculate mean and standard deviation using NumPy",
                "Create a NumPy array and demonstrate array slicing"
            ],
            "pandas_basics": [
                "Load a CSV file into a Pandas DataFrame and display basic statistics",
                "Write code to clean missing values in a Pandas DataFrame",
                "Create a Pandas DataFrame and perform groupby operations"
            ],
            "ml_concepts": [
                "Explain the difference between supervised and unsupervised learning",
                "Describe what overfitting means and how to prevent it",
                "Explain the bias-variance tradeoff in machine learning"
            ],
            "linear_regression": [
                "Implement simple linear regression from scratch using Python",
                "Write code to calculate the cost function for linear regression",
                "Explain the gradient descent algorithm for linear regression"
            ]
        }

        templates = skill_templates.get(skill.skill_id, [
            f"Explain the concept of {skill.name}",
            f"Write code demonstrating {skill.name}",
            f"Describe how to implement {skill.name}"
        ])

        return random.choice(templates)

    def _get_current_state(self, prompt: Optional[Prompt]) -> np.ndarray:
        """Get current state representation for PPO agent"""
        if prompt is None:
            return np.zeros(self.ppo_agent.state_dim)

        curriculum_context = self.ppo_agent.get_curriculum_context()
        return self.ppo_agent.get_state_representation(prompt.text, curriculum_context)

    def _execute_action(self, action: int, current_prompt: Prompt) -> Tuple[float, Optional[Prompt], bool, bool]:
        """Execute selected action and return reward, next state, done, success"""
        try:
            # Process the prompt with the selected action strategy
            result = self.prompt_controller.handle_prompt(
                current_prompt.text,
                source="rl_training",
                user_id=self.current_learner_id
            )

            response = result['response']
            reward_value = 0.0

            # Calculate reward based on response quality
            if self.reward_service:
                reward_result = result.get('reward_result')
                if reward_result:
                    reward_value = reward_result.get('weighted_reward', 0.0)
                else:
                    # Fallback reward calculation
                    reward_value = self._calculate_basic_reward(response, current_prompt)
            else:
                reward_value = self._calculate_basic_reward(response, current_prompt)

            # Determine if episode should end
            done = self._should_end_episode(reward_value, result)

            # Determine success
            success = reward_value > 0.6  # Threshold for success

            # Generate next prompt (could be follow-up or new curriculum prompt)
            next_prompt = None
            if not done and success:
                next_prompt = self._generate_followup_prompt(current_prompt, response)
            elif done:
                next_prompt = None
            else:
                next_prompt = self._generate_curriculum_prompt()

            return reward_value, next_prompt, done, success

        except Exception as e:
            logger.error(f"Failed to execute action {action}: {str(e)}")
            return -0.5, None, True, False  # Penalty for failure

    def _calculate_basic_reward(self, response: str, prompt: Prompt) -> float:
        """Calculate basic reward when reward service is not available"""
        if not response or len(response.strip()) == 0:
            return 0.0

        reward = 0.5  # Base reward

        # Reward response length (up to a point)
        response_length = len(response.split())
        if response_length > 10:
            reward += 0.2
        if response_length > 50:
            reward += 0.1

        # Reward code-like responses for programming prompts
        if any(keyword in prompt.text.lower() for keyword in ['function', 'code', 'implement', 'write']):
            if any(code_indicator in response.lower() for code_indicator in ['def ', 'class ', 'import ', 'return ']):
                reward += 0.3

        # Penalize very short responses
        if response_length < 5:
            reward -= 0.2

        return max(0.0, min(1.0, reward))

    def _should_end_episode(self, reward: float, result: Dict[str, Any]) -> bool:
        """Determine if the episode should end"""
        # End episode on very low reward (failure)
        if reward < 0.2:
            return True

        # End episode if hallucination detected
        if result.get('hallucination_analysis') and result['hallucination_analysis'].is_hallucinated:
            return True

        # Random early termination for exploration
        if random.random() < 0.05:  # 5% chance
            return True

        return False

    def _generate_followup_prompt(self, original_prompt: Prompt, response: str) -> Optional[Prompt]:
        """Generate a follow-up prompt based on the response"""
        followup_templates = [
            "Can you explain that in more detail?",
            "Can you provide an example?",
            "How would you modify this approach?",
            "What are the limitations of this solution?",
            "Can you show me how to implement this?"
        ]

        followup_text = random.choice(followup_templates)
        return Prompt(followup_text)

    def _update_curriculum_progress(self, prompt: Prompt, reward: float):
        """Update curriculum progress based on performance"""
        # Extract skill from current PPO state
        curriculum_context = self.ppo_agent.get_curriculum_context()
        current_skill = curriculum_context.get('current_skill')

        if current_skill and current_skill in self.curriculum_tree.skills:
            # Update progress
            performance_score = float(reward)
            time_spent = 30  # Assume 30 seconds per interaction

            self.learner_progress.update_skill_progress(current_skill, performance_score, time_spent)

            # Update PPO agent curriculum state
            success = performance_score > 0.7
            difficulty = curriculum_context.get('difficulty_level', 0)
            progress = self.learner_progress.get_skill_mastery_level(current_skill)

            self.ppo_agent.update_curriculum_state(current_skill, difficulty, progress, success)

    def _end_episode(self, episode: int, reward: float, steps: int, successes: int):
        """Handle end of episode"""
        success_rate = successes / steps if steps > 0 else 0

        # Update training statistics
        self.training_stats['episodes'] = episode
        self.training_stats['total_steps'] += steps
        self.training_stats['episode_rewards'].append(reward)
        self.training_stats['success_rate'].append(success_rate)

        # Keep only recent stats
        if len(self.training_stats['episode_rewards']) > 100:
            self.training_stats['episode_rewards'] = self.training_stats['episode_rewards'][-100:]
            self.training_stats['success_rate'] = self.training_stats['success_rate'][-100:]

        logger.info(f"Episode {episode} completed | Reward: {reward:.3f} | Steps: {steps} | Success Rate: {success_rate:.2f}")

    def _adapt_curriculum(self):
        """Adapt curriculum based on training progress"""
        # Check if learner should advance difficulty
        current_difficulty = self.learner_progress.get_recommended_difficulty()
        recent_success_rate = sum(self.training_stats['success_rate'][-10:]) / len(self.training_stats['success_rate'][-10:]) if self.training_stats['success_rate'] else 0

        if recent_success_rate > 0.7 and current_difficulty.value < DifficultyLevel.EXPERT.value:
            # Increase difficulty
            new_difficulty = DifficultyLevel(current_difficulty.value + 1)
            self.learner_progress.current_difficulty = new_difficulty
            logger.info(f"Advanced curriculum difficulty to {new_difficulty.value}")
        elif recent_success_rate < 0.3 and current_difficulty.value > DifficultyLevel.EASY.value:
            # Decrease difficulty
            new_difficulty = DifficultyLevel(current_difficulty.value - 1)
            self.learner_progress.current_difficulty = new_difficulty
            logger.info(f"Reduced curriculum difficulty to {new_difficulty.value}")

        # Update curriculum progress tracking
        self.training_stats['curriculum_progress'].append({
            'episode': self.training_stats['episodes'],
            'difficulty': current_difficulty.value,
            'success_rate': recent_success_rate,
            'completed_skills': len(self.learner_progress.completed_skills)
        })

    def _save_model(self):
        """Save the PPO model"""
        try:
            self.ppo_agent.save_model(self.config['model_path'])
            logger.info(f"Model saved to {self.config['model_path']}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and statistics"""
        recent_rewards = self.training_stats['episode_rewards'][-20:] if self.training_stats['episode_rewards'] else []
        recent_success = self.training_stats['success_rate'][-20:] if self.training_stats['success_rate'] else []

        return {
            'is_training': self.is_training,
            'episodes': self.training_stats['episodes'],
            'total_steps': self.training_stats['total_steps'],
            'policy_updates': self.training_stats['policy_updates'],
            'current_avg_reward': sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0,
            'current_success_rate': sum(recent_success) / len(recent_success) if recent_success else 0,
            'curriculum_difficulty': self.learner_progress.current_difficulty.value,
            'completed_skills': len(self.learner_progress.completed_skills),
            'available_skills': len(self.learner_progress.get_available_skills(self.curriculum_tree)),
            'recent_rewards': recent_rewards,
            'recent_success_rates': recent_success
        }

    def generate_curriculum_from_prompt(self, user_prompt: str) -> CurriculumTree:
        """Generate a curriculum from a user prompt"""
        prompt = Prompt(user_prompt)
        analysis = self.curriculum_generator.analyze_prompt(prompt)
        curriculum = self.curriculum_generator.generate_curriculum(analysis)

        logger.info(f"Generated curriculum with {len(curriculum.skills)} skills from prompt analysis")
        return curriculum

    def get_progressive_tasks(self, user_prompt: str, num_tasks: int = 5) -> List[Dict[str, Any]]:
        """Generate progressive tasks from a user prompt"""
        curriculum = self.generate_curriculum_from_prompt(user_prompt)
        return self.curriculum_generator.generate_progressive_tasks(curriculum, num_tasks=num_tasks)