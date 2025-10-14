import asyncio
import threading
import time
import random
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, Any
from models.prompt import Prompt
from models.task_generator import TaskGenerator
from models.curriculum_generator import CurriculumGenerator
from models.ppo_agent import PPOAgent
from services.rl_training_service import RLTrainingService
from services.meta_learning_service import MetaLearningService
from controllers.prompt_controller import PromptController
from controllers.evaluator import Evaluator
from controllers.scheduler import Scheduler
from models.curriculum import CurriculumProgress, DifficultyLevel, CurriculumTree
from utils.logging_config import (
    get_logger, log_learning_event, log_curriculum_status,
    start_progress_tracker, update_progress, complete_progress
)

logger = get_logger(__name__)

class LearningLoopService:
    def __init__(self, prompt_controller, evaluator, scheduler, task_generator=None, feedback_service=None, reward_service=None, meta_learning_service=None):
        self.prompt_controller = prompt_controller or None  # Allow None initially
        self.evaluator = evaluator
        self.scheduler = scheduler
        self.task_generator = task_generator or TaskGenerator()
        self.feedback_service = feedback_service
        self.reward_service = reward_service
        self.meta_learning_service = meta_learning_service

        # Curriculum and RL components
        self.curriculum_generator = CurriculumGenerator()
        self.ppo_agent = PPOAgent(
            state_dim=10,  # State representation dimension
            action_dim=10,  # Number of possible actions
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            epsilon=0.2
        )
        self.rl_training_service = RLTrainingService(
            prompt_controller=prompt_controller,
            reward_service=reward_service,
            curriculum_generator=self.curriculum_generator
        )

        # Curriculum state
        self.curriculum_tree = CurriculumTree.create_default_curriculum()
        self.current_curriculum_tasks = []
        self.current_task_index = 0
        self.user_curriculum_tasks = []  # Tasks generated from user prompts
        self.use_user_curriculum = False  # Flag to use user-generated curriculum

        self.is_running = False
        self.loop_thread = None
        self.initial_prompt_received = False  # NEW: Flag to track if initial prompt received
        self.waiting_for_initial_prompt = True  # NEW: System waits for user input
        self.initial_prompt_data = None  # NEW: Store initial prompt data
        self.learning_stats = {
            'iterations': 0,
            'total_rewards': [],
            'average_rewards': [],
            'generated_prompts': [],
            'success_rates': [],
            'curriculum_progress': {},
            'feedback_influence': [],  # Track how feedback influences learning
            'learning_outcomes': [],  # Track detailed learning outcomes per iteration
            'ppo_training_stats': {  # Track PPO-specific metrics
                'ppo_action_distribution': {},
                'average_ppo_reward': [],
                'policy_updates': 0,
                'total_transitions': 0
            }
        }
        # Default learner ID for autonomous learning
        self.default_learner_id = "autonomous_agent"
        self._ensure_default_learner()

    def _ensure_default_learner(self):
        """Ensure the default autonomous learner exists"""
        if not self.scheduler.get_learner_progress(self.default_learner_id):
            self.scheduler.register_learner(self.default_learner_id)
            logger.info(f"Created default autonomous learner: {self.default_learner_id}")

    def set_user_curriculum_tasks(self, curriculum_tasks):
        """Set curriculum tasks generated from a user's prompt"""
        try:
            if not curriculum_tasks:
                logger.warning("No curriculum tasks provided")
                return

            self.user_curriculum_tasks = curriculum_tasks
            self.use_user_curriculum = True
            self.current_task_index = 0  # Reset task index for new curriculum

            logger.info(f"Set {len(curriculum_tasks)} user-generated curriculum tasks for autonomous learning")
        except Exception as e:
            logger.error(f"Failed to set user curriculum tasks: {str(e)}")
            raise

    def receive_initial_prompt(self, prompt_text: str, result: dict, user_id: str = None):
        """
        Receive initial prompt from user and trigger autonomous learning loop.
        This is the entry point for starting the autonomous learning system.
        """
        if self.initial_prompt_received:
            logger.info("ℹ️  Initial prompt already received, learning loop is active")
            return

        logger.info(f"🎯 Received initial user prompt: '{prompt_text[:50]}...' | User: {user_id or 'unknown'}")

        # Store initial prompt data
        self.initial_prompt_data = {
            'prompt_text': prompt_text,
            'result': result,
            'user_id': user_id or self.default_learner_id,
            'timestamp': time.time()
        }

        # Mark that we've received the initial prompt
        self.initial_prompt_received = True
        self.waiting_for_initial_prompt = False

        # Log the initialization
        log_learning_event("initial_prompt_received", {
            'prompt_text': prompt_text[:100],
            'user_id': user_id,
            'prompt_length': len(prompt_text),
            'will_start_loop': not self.is_running
        })

        # Start the autonomous learning loop if not already running
        if not self.is_running:
            logger.info("🚀 Initial prompt received! Starting autonomous learning loop...")
            self.start_learning_loop()
        else:
            logger.info("✅ Learning loop already running, initial prompt processed")

    def start_learning_loop(self):
        """Start the autonomous learning loop in a background thread"""
        if self.is_running:
            logger.warning("⚠️  Learning loop is already running")
            return

        # Check if we're waiting for initial prompt
        if self.waiting_for_initial_prompt and not self.initial_prompt_received:
            logger.warning("⏳ Cannot start learning loop: Waiting for initial user prompt from UI")
            logger.info("💡 Please submit a prompt through the UI to begin autonomous learning")
            return

        logger.info("🚀 Starting autonomous learning loop...")

        # Start progress tracking
        self.loop_tracker_id = start_progress_tracker("autonomous_learning_loop")

        self.is_running = True
        self.loop_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.loop_thread.start()

        # Log learning event
        log_learning_event("learning_loop_started", {
            'tracker_id': self.loop_tracker_id,
            'learner_id': self.default_learner_id,
            'thread_id': self.loop_thread.ident,
            'triggered_by_initial_prompt': self.initial_prompt_received
        })

        logger.info("✅ Autonomous learning loop started successfully")

    def stop_learning_loop(self):
        """Stop the learning loop"""
        if not self.is_running:
            logger.info("ℹ️  Learning loop is not running")
            return

        logger.info("🛑 Stopping autonomous learning loop...")

        # Get final stats before stopping
        final_stats = {
            'total_iterations': self.learning_stats['iterations'],
            'final_avg_reward': self.learning_stats['average_rewards'][-1] if self.learning_stats['average_rewards'] else 0,
            'total_rewards_count': len(self.learning_stats['total_rewards']),
            'curriculum_skills_practiced': len(self.learning_stats['curriculum_progress'])
        }

        self.is_running = False
        if self.loop_thread:
            self.loop_thread.join(timeout=5)

        # Complete progress tracking
        complete_progress(self.loop_tracker_id, success=True,
                         message=f"Completed {final_stats['total_iterations']} iterations")

        # Log learning event
        log_learning_event("learning_loop_stopped", {
            'tracker_id': self.loop_tracker_id,
            'final_stats': final_stats,
            'total_runtime': time.time() - getattr(self, 'loop_start_time', time.time())
        })

        logger.info(f"✅ Autonomous learning loop stopped | Total iterations: {final_stats['total_iterations']} | "
                   f"Final avg reward: {final_stats['final_avg_reward']:.3f}")

    def get_loop_status(self):
        """Get current status of the learning loop"""
        return {
            'is_running': self.is_running,
            'waiting_for_initial_prompt': self.waiting_for_initial_prompt,
            'initial_prompt_received': self.initial_prompt_received,
            'iterations': self.learning_stats['iterations'],
            'thread_alive': self.loop_thread.is_alive() if self.loop_thread else False,
            'initial_prompt_data': {
                'prompt_text': self.initial_prompt_data['prompt_text'][:100] if self.initial_prompt_data else None,
                'user_id': self.initial_prompt_data['user_id'] if self.initial_prompt_data else None,
                'timestamp': self.initial_prompt_data['timestamp'] if self.initial_prompt_data else None
            } if self.initial_prompt_data else None
        }

    def _learning_loop(self):
        """Main learning loop that runs continuously with curriculum awareness"""
        logger.info("🚀 Starting autonomous learning loop with curriculum integration")

        # Log initial prompt context if available
        if self.initial_prompt_data:
            logger.info(f"📝 Initial prompt context: '{self.initial_prompt_data['prompt_text'][:100]}...' | "
                       f"User: {self.initial_prompt_data['user_id']}")

        iteration_count = 0
        loop_start_time = time.time()

        while self.is_running:
            iteration_count += 1
            iteration_start = time.time()

            try:
                # Log iteration start
                log_learning_event("learning_iteration_started", {
                    'iteration': iteration_count,
                    'total_runtime': time.time() - loop_start_time,
                    'learner_id': self.default_learner_id
                })

                # Generate curriculum-aware prompt with feedback influence
                new_prompt_text, target_skill, feedback_influence = self._generate_curriculum_prompt()

                if new_prompt_text is None:
                    # This should never happen now with fallback prompts, but keep as safety
                    logger.warning(f"⚠️  Iteration {iteration_count}: No suitable prompt available, skipping")
                    # Still count this as an iteration attempt
                    self._update_stats(0.0, None, {}, None, None, None, None)
                    time.sleep(5)  # Wait before checking again
                    continue

                prompt = Prompt(new_prompt_text)
                logger.info(f"📝 Iteration {iteration_count}: Generated prompt for skill '{target_skill}' | "
                            f"Length: {len(new_prompt_text)} chars")

                # Process the prompt (simplified without PPO for now)
                if self.prompt_controller is None:
                    logger.error("Prompt controller not available, skipping iteration")
                    time.sleep(5)
                    continue

                result = self.prompt_controller.handle_prompt(
                    prompt.text, source="ai", user_id=self.default_learner_id
                )
                response, action_taken = result['response'], result['action']

                # Evaluate and get reward with feedback considerations
                reward = self._evaluate_response_with_feedback(prompt, response, action_taken, result)

                # PPO operations (wrapped in try-catch to prevent failures)
                try:
                    curriculum_context = self.ppo_agent.get_curriculum_context()
                    state = self.ppo_agent.get_state_representation(prompt.text, curriculum_context)
                    
                    # 🔍 INSTRUMENTATION: Validate state shape
                    if state.shape[0] != self.ppo_agent.state_dim:
                        logger.error(f"🔴 State shape error at iteration {iteration_count}!")
                        logger.error(f"   Expected state_dim={self.ppo_agent.state_dim}, got shape={state.shape}")
                        logger.error(f"   Prompt length: {len(prompt.text)}, Curriculum: {curriculum_context}")
                        # Fix state dimension
                        if state.shape[0] < self.ppo_agent.state_dim:
                            padded = np.zeros(self.ppo_agent.state_dim)
                            padded[:state.shape[0]] = state
                            state = padded
                        else:
                            state = state[:self.ppo_agent.state_dim]
                    
                    ppo_action, log_prob, value = self.ppo_agent.select_action(state)

                    # Store PPO transition for learning
                    next_state = self.ppo_agent.get_state_representation(response, curriculum_context)
                    
                    # 🔍 Validate next_state shape
                    if next_state.shape[0] != self.ppo_agent.state_dim:
                        logger.error(f"🔴 Next state shape error at iteration {iteration_count}!")
                        logger.error(f"   Expected state_dim={self.ppo_agent.state_dim}, got shape={next_state.shape}")
                        if next_state.shape[0] < self.ppo_agent.state_dim:
                            padded = np.zeros(self.ppo_agent.state_dim)
                            padded[:next_state.shape[0]] = next_state
                            next_state = padded
                        else:
                            next_state = next_state[:self.ppo_agent.state_dim]
                    
                    done = self._should_end_ppo_episode(reward, result, iteration_count)
                    self.ppo_agent.store_transition(state, ppo_action, log_prob, reward, value, done)

                    # Collect PPO feedback for meta-learning adaptation
                    if self.meta_learning_service:
                        try:
                            difficulty_value = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}
                            skill_difficulty = difficulty_value.get(
                                self.curriculum_tree.skills[target_skill].difficulty.value, 1
                            ) if target_skill and target_skill in self.curriculum_tree.skills else 1

                            self.meta_learning_service.collect_ppo_feedback(
                                state, ppo_action, reward, next_state, done,
                                self.default_learner_id, target_skill, skill_difficulty
                            )
                        except Exception as e:
                            logger.warning(f"Failed to collect PPO feedback: {str(e)}")

                    # PPO Learning: Update policy if enough transitions accumulated
                    if iteration_count % 10 == 0:  # Update every 10 iterations
                        self.ppo_agent.update_policy()
                        self.learning_stats['ppo_training_stats']['policy_updates'] = self.learning_stats['ppo_training_stats'].get('policy_updates', 0) + 1

                    # Update PPO curriculum state based on performance
                    if target_skill:
                        success = reward > 0.7
                        difficulty_value = {'easy': 0, 'medium': 1, 'hard': 2, 'expert': 3}
                        skill_difficulty = difficulty_value.get(
                            self.curriculum_tree.skills[target_skill].difficulty.value, 1
                        )
                        progress = self.scheduler.get_learner_progress(self.default_learner_id).get_skill_mastery_level(target_skill)
                        self.ppo_agent.update_curriculum_state(target_skill, skill_difficulty, progress, success)

                except Exception as ppo_error:
                    logger.warning(f"PPO operations failed: {str(ppo_error)}, continuing without PPO")
                    ppo_action = 0

                # Log learning interaction
                log_learning_event("learning_interaction_completed", {
                    'iteration': iteration_count,
                    'prompt_length': len(new_prompt_text),
                    'response_length': len(response),
                    'action': action_taken,
                    'reward': reward,
                    'target_skill': target_skill,
                    'hallucinated': result.get('hallucination_analysis').is_hallucinated if result.get('hallucination_analysis') else False,
                    'feedback_influenced': bool(feedback_influence and any(feedback_influence.values())),
                    'processing_time': time.time() - iteration_start
                })

                # Update curriculum progress based on performance
                if target_skill:
                    performance_score = self._calculate_performance_score(reward, result)
                    self.scheduler.update_learner_progress(
                        self.default_learner_id,
                        target_skill,
                        performance_score,
                        time_spent=30  # Estimate 30 seconds per task
                    )

                    # Log curriculum progress update
                    log_curriculum_status(self.default_learner_id, "skill_practiced", {
                        'skill': target_skill,
                        'performance_score': performance_score,
                        'reward': reward,
                        'iteration': iteration_count
                    })

                # Learn from the interaction (original LLM learning)
                self.prompt_controller.llm.learn(action_taken, reward)

                # Execute curriculum feedback loop periodically
                if self.meta_learning_service and iteration_count % 25 == 0:  # Every 25 iterations
                    try:
                        feedback_result = self.meta_learning_service.execute_curriculum_feedback_loop(self.default_learner_id)
                        if feedback_result.get('success', False):
                            logger.info(f"🔄 Curriculum feedback loop executed | Adaptations: {len(feedback_result.get('adaptations_applied', []))}")
                        else:
                            logger.warning(f"⚠️  Curriculum feedback loop failed: {feedback_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.warning(f"Failed to execute curriculum feedback loop: {str(e)}")

                # Update learning stats with feedback influence and learning outcomes
                self._update_stats(reward, target_skill, feedback_influence, prompt, response, action_taken, result)

                # Log iteration completion
                iteration_time = time.time() - iteration_start
                logger.info(f"✅ Iteration {iteration_count} completed | Reward: {reward:.3f} | "
                            f"Skill: {target_skill or 'N/A'} | Time: {iteration_time:.2f}s")

                # Sleep to control loop speed
                time.sleep(1)  # 1 second between iterations

            except Exception as e:
                iteration_time = time.time() - iteration_start
                logger.error(f"❌ Error in learning loop iteration {iteration_count}: {str(e)} | "
                             f"Time: {iteration_time:.2f}s")
                log_learning_event("learning_iteration_failed", {
                    'iteration': iteration_count,
                    'error': str(e),
                    'iteration_time': iteration_time
                })
                # Update stats even on failure to track attempts
                self._update_stats(0.0, None, {}, None, None, None, None)
                time.sleep(5)  # Wait longer on error

    def _generate_curriculum_prompt(self):
        """Generate a curriculum-aware prompt using curriculum generator with feedback influence"""
        try:
            # Get learner progress
            learner_progress = self.scheduler.get_learner_progress(self.default_learner_id)

            # Get feedback insights to influence prompt generation
            feedback_influence = {}
            if self.feedback_service:
                try:
                    collaborative_insights = self.feedback_service.get_collaborative_insights()
                    user_prefs = self.feedback_service.get_user_preferences(
                        type('Request', (), {'user_id': self.default_learner_id, 'include_history': False})()
                    ).preferences

                    feedback_influence = {
                        'collaborative_insights': len(collaborative_insights),
                        'user_preferences': hasattr(user_prefs, 'preferred_styles') and bool(user_prefs.preferred_styles),
                        'correction_patterns': self._get_correction_influence()
                    }
                except Exception as e:
                    logger.warning(f"Failed to get feedback influence: {str(e)}")

            # Check if we should use user-generated curriculum tasks
            if self.use_user_curriculum and self.user_curriculum_tasks:
                # Use user-generated curriculum tasks
                if self.current_task_index >= len(self.user_curriculum_tasks):
                    # All user curriculum tasks completed, switch back to default curriculum
                    logger.info("User curriculum tasks completed, switching back to default curriculum")
                    self.use_user_curriculum = False
                    self.current_task_index = 0
                else:
                    # Get current user curriculum task
                    current_task = self.user_curriculum_tasks[self.current_task_index]
                    target_skill = current_task.get('skill_id', 'unknown_skill')

                    # Generate prompt based on the curriculum task
                    prompt_text = self._generate_curriculum_task_prompt(current_task, feedback_influence)

                    # Move to next task for next iteration
                    self.current_task_index += 1

                    logger.info(f"Using user curriculum task {self.current_task_index}/{len(self.user_curriculum_tasks)}: {target_skill}")
                    return prompt_text, target_skill, feedback_influence

            # Check if we need to generate new curriculum tasks (default behavior)
            if not self.current_curriculum_tasks or self.current_task_index >= len(self.current_curriculum_tasks):
                # Generate progressive curriculum tasks
                self.current_curriculum_tasks = self.curriculum_generator.generate_progressive_tasks(
                    self.curriculum_tree,
                    learner_progress,
                    num_tasks=10  # Generate a batch of tasks
                )
                self.current_task_index = 0
                logger.info(f"Generated {len(self.current_curriculum_tasks)} new default curriculum tasks")

            if not self.current_curriculum_tasks:
                logger.warning("No curriculum tasks available, falling back to general prompt generation")
                # Fall back to general prompt generation instead of returning None
                fallback_prompt = self._generate_fallback_prompt(feedback_influence)
                return fallback_prompt, 'general_learning', feedback_influence

            # Get current task
            current_task = self.current_curriculum_tasks[self.current_task_index]
            target_skill = current_task['skill_id']

            # Generate prompt based on the curriculum task
            prompt_text = self._generate_curriculum_task_prompt(current_task, feedback_influence)

            # Move to next task for next iteration
            self.current_task_index += 1

            return prompt_text, target_skill, feedback_influence

        except Exception as e:
            logger.warning(f"Failed to generate curriculum prompt: {str(e)}")
            # Always return a fallback prompt instead of None
            fallback_prompt = self._generate_fallback_prompt({})
            return fallback_prompt, 'general_learning', {}

    def _generate_fallback_prompt(self, feedback_influence):
        """Generate a fallback prompt when curriculum generation fails"""
        try:
            # Use the conversational prompt generation as fallback
            topics = ['machine learning', 'programming', 'data science', 'algorithms', 'artificial intelligence']
            main_topic = random.choice(topics)

            base_prompt = f"Tell me about {main_topic} and explain a key concept related to it."

            # Apply feedback influence if available
            if feedback_influence.get('user_preferences'):
                try:
                    user_prefs = self.feedback_service.get_user_preferences(
                        type('Request', (), {'user_id': self.default_learner_id, 'include_history': False})()
                    ).preferences

                    if hasattr(user_prefs, 'preferred_styles') and user_prefs.preferred_styles:
                        if 'technical' in user_prefs.preferred_styles:
                            base_prompt += " Provide a technical explanation with examples."
                        elif 'simple' in user_prefs.preferred_styles:
                            base_prompt += " Explain this in simple terms for beginners."
                except Exception as e:
                    logger.warning(f"Failed to apply user preferences to fallback prompt: {str(e)}")

            if feedback_influence.get('correction_patterns'):
                base_prompt += " Focus on accuracy and avoid common mistakes."

            if feedback_influence.get('collaborative_insights', 0) > 0:
                base_prompt += " Consider best practices and common patterns."

            return base_prompt

        except Exception as e:
            logger.warning(f"Failed to generate fallback prompt: {str(e)}")
            # Ultimate fallback - always return something
            return "Explain a basic programming concept and provide an example."

    def _generate_skill_based_prompt(self, skill):
        """Generate a prompt that exercises a specific skill"""
        skill_name = skill.name.lower()

        # Skill-specific prompt templates
        skill_templates = {
            "python_basics": [
                "Write a Python function to calculate the factorial of a number",
                "Create a Python script that reads a file and counts word frequencies",
                "Implement a simple Python class for a bank account"
            ],
            "data_structures": [
                "Implement a stack data structure in Python",
                "Create a function that finds duplicates in a list",
                "Write code to merge two sorted lists"
            ],
            "numpy_basics": [
                "Use NumPy to create and manipulate a 2D array",
                "Write code to perform matrix operations with NumPy",
                "Create a NumPy array and perform statistical operations"
            ],
            "pandas_basics": [
                "Load a CSV file into a Pandas DataFrame and perform basic analysis",
                "Write code to clean and preprocess data using Pandas",
                "Create a Pandas DataFrame and perform grouping operations"
            ],
            "ml_concepts": [
                "Explain the difference between supervised and unsupervised learning",
                "Describe how a decision tree algorithm works",
                "What is overfitting and how can it be prevented?"
            ],
            "linear_regression": [
                "Implement simple linear regression from scratch",
                "Explain the cost function used in linear regression",
                "Write code to train a linear regression model"
            ]
        }

        # Get templates for this skill or use general templates
        templates = skill_templates.get(skill.skill_id, [
            f"Create a {skill_name} implementation",
            f"Explain how {skill_name} works",
            f"Write code demonstrating {skill_name}"
        ])

        return random.choice(templates)

    def _generate_new_prompt(self):
        """Generate a new prompt for learning based on user's initial prompt topic"""
        try:
            # Get the initial user prompt topic
            initial_topic = self._get_initial_prompt_topic()

            if initial_topic:
                # Generate conversational prompts based on the initial topic
                prompt = self._generate_conversational_prompt(initial_topic)
                self.learning_stats['generated_prompts'].append(f"Topic-based: {prompt}")
                return prompt
            else:
                # No initial topic available, skip generation
                logger.info("No initial user topic available, skipping prompt generation")
                return None
        except Exception as e:
            logger.warning(f"Failed to generate topic-based prompt: {str(e)}")
            return None

    def _get_initial_prompt_topic(self):
        """Extract the topic from the user's initial prompt"""
        user_interactions = [inter for inter in self.prompt_controller.history.interactions
                           if inter.get('source') == 'user']

        if not user_interactions:
            return None

        # Get the first user prompt
        initial_prompt = user_interactions[0]['prompt']
        initial_text = initial_prompt.text if hasattr(initial_prompt, 'text') else str(initial_prompt)

        # Extract key topics/keywords from the initial prompt
        # Simple extraction: split into words and filter common words
        words = initial_text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

        key_topics = [word for word in words if len(word) > 2 and word not in stop_words][:5]  # Limit to 5 key topics

        return key_topics if key_topics else None

    def _generate_conversational_prompt(self, topics):
        """Generate a conversational prompt based on the given topics"""
        if not topics:
            return "Tell me something interesting"

        main_topic = random.choice(topics)

        # Conversational templates that build on the topic
        conversational_templates = [
            f"Can you elaborate more on {main_topic}?",
            f"What are your thoughts on {main_topic}?",
            f"How would you explain {main_topic} to a beginner?",
            f"What are some practical applications of {main_topic}?",
            f"Can you give me an example related to {main_topic}?",
            f"Why is {main_topic} important?",
            f"What challenges are there with {main_topic}?",
            f"How has {main_topic} evolved over time?",
            f"Can you compare {main_topic} with something similar?",
            f"What would you recommend for learning more about {main_topic}?",
            f"Tell me a story or scenario involving {main_topic}",
            f"How does {main_topic} relate to everyday life?",
            f"What are the pros and cons of {main_topic}?",
            f"Can you break down {main_topic} into simpler terms?",
            f"What questions do people often ask about {main_topic}?"
        ]

        # Sometimes include multiple topics for richer conversation
        if len(topics) > 1 and random.random() < 0.3:
            secondary_topic = random.choice([t for t in topics if t != main_topic])
            extended_templates = [
                f"How do {main_topic} and {secondary_topic} relate to each other?",
                f"Can you compare {main_topic} with {secondary_topic}?",
                f"What happens when you combine {main_topic} and {secondary_topic}?",
                f"Which is more important: {main_topic} or {secondary_topic}?"
            ]
            conversational_templates.extend(extended_templates)

        return random.choice(conversational_templates)

    def _calculate_performance_score(self, reward, result):
        """Calculate a performance score for curriculum tracking"""
        # Use the reward as base performance score
        if isinstance(reward, dict):
            # Multi-objective reward
            performance = reward.get('weighted_reward', 0.5)
        else:
            # Single reward
            performance = float(reward)

        # Adjust based on other factors
        if result.get('hallucination_analysis'):
            hallucination_confidence = result['hallucination_analysis'].overall_confidence
            # Reduce performance if high hallucination confidence
            performance *= (1.0 - hallucination_confidence * 0.3)

        # Factor in response quality
        response_length = len(result.get('response', ''))
        if response_length > 200:
            performance *= 1.1  # Bonus for detailed responses
        elif response_length < 20:
            performance *= 0.8  # Penalty for too short responses

        return max(0.0, min(1.0, performance))

    def _evaluate_response(self, prompt, response, action):
        """Evaluate the response quality and return a reward"""
        # Simple evaluation based on response length and action diversity
        base_reward = 0.5

        # Reward longer, more detailed responses
        if len(response) > 50:
            base_reward += 0.2
        if len(response) > 100:
            base_reward += 0.1

        # Reward varied actions (exploration)
        action_history = [inter['action'] for inter in self.prompt_controller.history.interactions[-10:] if inter.get('action') is not None]
        if action_history.count(action) < len(action_history) * 0.7:  # Not too repetitive
            base_reward += 0.1

        # Add some randomness to simulate real evaluation
        reward = base_reward + random.uniform(-0.1, 0.1)
        reward = max(0.0, min(1.0, reward))  # Clamp to [0,1]

        return reward

    def _update_stats(self, reward, target_skill=None, feedback_influence=None, prompt=None, response=None, action=None, result=None):
        """Update learning statistics with curriculum tracking and feedback influence"""
        self.learning_stats['iterations'] += 1
        self.learning_stats['total_rewards'].append(reward)

        # Track learning outcomes for this iteration
        learning_outcome = {
            'iteration': self.learning_stats['iterations'],
            'timestamp': time.time(),
            'prompt': prompt.text if prompt else None,
            'response': response,
            'action': action,
            'reward': reward,
            'target_skill': target_skill,
            'hallucination_detected': result.get('hallucination_analysis').is_hallucinated if result and result.get('hallucination_analysis') else False,
            'processing_time': result.get('processing_time') if result else None,
            'feedback_influence': feedback_influence
        }
        self.learning_stats['learning_outcomes'].append(learning_outcome)

        # Keep only recent learning outcomes (last 20 iterations)
        if len(self.learning_stats['learning_outcomes']) > 20:
            self.learning_stats['learning_outcomes'] = self.learning_stats['learning_outcomes'][-20:]

        # Track feedback influence
        if feedback_influence:
            self.learning_stats['feedback_influence'].append({
                'iteration': self.learning_stats['iterations'],
                'influence': feedback_influence,
                'reward': reward,
                'timestamp': time.time()
            })

            # Keep only recent feedback influence data
            if len(self.learning_stats['feedback_influence']) > 50:
                self.learning_stats['feedback_influence'] = self.learning_stats['feedback_influence'][-50:]

        # Keep only recent rewards for average
        recent_rewards = self.learning_stats['total_rewards'][-100:]
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        self.learning_stats['average_rewards'].append(avg_reward)

        # Success rate (rewards > 0.6 considered success)
        success_rate = sum(1 for r in recent_rewards if r > 0.6) / len(recent_rewards) if recent_rewards else 0
        self.learning_stats['success_rates'].append(success_rate)

        # Track curriculum progress
        if target_skill:
            if target_skill not in self.learning_stats['curriculum_progress']:
                self.learning_stats['curriculum_progress'][target_skill] = {
                    'attempts': 0,
                    'total_reward': 0.0,
                    'best_reward': 0.0,
                    'last_attempt': 0,
                    'feedback_influenced': 0
                }

            skill_stats = self.learning_stats['curriculum_progress'][target_skill]
            skill_stats['attempts'] += 1
            skill_stats['total_reward'] += reward
            skill_stats['best_reward'] = max(skill_stats['best_reward'], reward)
            skill_stats['last_attempt'] = time.time()

            # Track feedback influence on skill learning
            if feedback_influence and any(feedback_influence.values()):
                skill_stats['feedback_influenced'] += 1

        # Track PPO-specific statistics
        ppo_stats = self.learning_stats['ppo_training_stats']
        ppo_stats['total_transitions'] = len(self.ppo_agent.states) if hasattr(self.ppo_agent, 'states') else 0

        # Track PPO action distribution
        if action is not None:
            action_str = str(action)
            if action_str not in ppo_stats['ppo_action_distribution']:
                ppo_stats['ppo_action_distribution'][action_str] = 0
            ppo_stats['ppo_action_distribution'][action_str] += 1

        # Track PPO reward averages
        if len(self.learning_stats['total_rewards']) >= 10:
            recent_ppo_rewards = self.learning_stats['total_rewards'][-10:]
            avg_ppo_reward = sum(recent_ppo_rewards) / len(recent_ppo_rewards)
            ppo_stats['average_ppo_reward'].append(avg_ppo_reward)

            # Keep only recent averages
            if len(ppo_stats['average_ppo_reward']) > 50:
                ppo_stats['average_ppo_reward'] = ppo_stats['average_ppo_reward'][-50:]

    def get_learning_progress(self):
        """Get current learning progress data with curriculum integration"""
        base_progress = {
            'iterations': self.learning_stats['iterations'],
            'current_avg_reward': self.learning_stats['average_rewards'][-1] if self.learning_stats['average_rewards'] else 0,
            'recent_rewards': self.learning_stats['total_rewards'][-50:],  # Last 50 rewards
            'success_rate': self.learning_stats['success_rates'][-1] if self.learning_stats['success_rates'] else 0,
            'generated_prompts_count': len(self.learning_stats['generated_prompts']),
            'is_running': self.is_running
        }

        # Add curriculum progress for the autonomous learner
        curriculum_status = self.scheduler.get_curriculum_status(self.default_learner_id)
        if "error" not in curriculum_status:
            base_progress['curriculum'] = {
                'completed_skills': curriculum_status['curriculum_summary']['completed_skills'],
                'total_skills': curriculum_status['curriculum_summary']['total_skills'],
                'completion_percentage': curriculum_status['curriculum_summary']['completion_percentage'],
                'current_difficulty': curriculum_status['curriculum_summary']['current_difficulty'],
                'recommended_difficulty': curriculum_status['curriculum_summary']['recommended_difficulty'],
                'available_skills': curriculum_status['available_skills'][:5],  # Top 5 available skills
                'skill_progress': curriculum_status['progress']['skill_progress']
            }

        # Add curriculum skill statistics
        base_progress['curriculum_skill_stats'] = self.learning_stats['curriculum_progress']

        # Add feedback influence statistics
        if self.learning_stats['feedback_influence']:
            recent_feedback = self.learning_stats['feedback_influence'][-10:]
            base_progress['feedback_influence'] = {
                'recent_influences': recent_feedback,
                'total_influenced_iterations': len(self.learning_stats['feedback_influence']),
                'avg_influence_score': sum(fi['influence'].get('collaborative_insights', 0) for fi in recent_feedback) / len(recent_feedback) if recent_feedback else 0
            }

        # Add learning outcomes for detailed tracking
        base_progress['learning_outcomes'] = self.learning_stats['learning_outcomes']

        # Add PPO training statistics
        ppo_stats = self.learning_stats.get('ppo_training_stats', {})
        base_progress['ppo_stats'] = {
            'policy_updates': ppo_stats.get('policy_updates', 0),
            'total_transitions': len(self.ppo_agent.states) if hasattr(self.ppo_agent, 'states') else 0,
            'current_avg_ppo_reward': ppo_stats.get('average_ppo_reward', [])[-1] if ppo_stats.get('average_ppo_reward') else 0,
            'ppo_action_distribution': ppo_stats.get('ppo_action_distribution', {}),
            'curriculum_context': self.ppo_agent.get_curriculum_context() if hasattr(self.ppo_agent, 'get_curriculum_context') else {}
        }

        return base_progress

    def _generate_curriculum_task_prompt(self, task_info, feedback_influence):
        """Generate a prompt based on curriculum task information with feedback influence"""
        task_description = task_info['task_description']
        skill_name = task_info['skill_name']
        difficulty = task_info['difficulty']
        learning_objectives = task_info.get('learning_objectives', [])

        # Base prompt from task description
        base_prompt = task_description

        # Add context about skill and difficulty
        difficulty_names = {1: "basic", 2: "intermediate", 3: "advanced", 4: "expert"}
        difficulty_name = difficulty_names.get(difficulty, "intermediate")

        enhanced_prompt = f"{difficulty_name.capitalize()} {skill_name} task: {base_prompt}"

        # Add learning objectives if available
        if learning_objectives:
            objectives_text = " ".join(f"• {obj}" for obj in learning_objectives[:3])  # Limit to 3
            enhanced_prompt += f"\n\nLearning Objectives:\n{objectives_text}"

        # Apply feedback influence
        if feedback_influence.get('user_preferences'):
            try:
                user_prefs = self.feedback_service.get_user_preferences(
                    type('Request', (), {'user_id': self.default_learner_id, 'include_history': False})()
                ).preferences

                if hasattr(user_prefs, 'preferred_styles') and user_prefs.preferred_styles:
                    if 'technical' in user_prefs.preferred_styles:
                        enhanced_prompt += " Provide a technical implementation with detailed code."
                    elif 'simple' in user_prefs.preferred_styles:
                        enhanced_prompt += " Explain this in simple, easy-to-understand terms."
            except Exception as e:
                logger.warning(f"Failed to apply user preferences to prompt: {str(e)}")

        if feedback_influence.get('correction_patterns'):
            enhanced_prompt += " Focus on accuracy and avoid common mistakes."

        if feedback_influence.get('collaborative_insights', 0) > 0:
            enhanced_prompt += " Consider best practices and common patterns."

        return enhanced_prompt

    def _generate_new_prompt_with_feedback(self, feedback_influence):
        """Generate a new prompt with feedback influence"""
        base_prompt = self._generate_new_prompt()

        if base_prompt and feedback_influence.get('collaborative_insights', 0) > 0:
            # Enhance prompt with collaborative insights
            base_prompt += " Consider best practices and common patterns."

        return base_prompt

    def _evaluate_response_with_feedback(self, prompt, response, action, result):
        """Evaluate response quality with feedback considerations"""
        # Get base evaluation
        base_reward = self._evaluate_response(prompt, response, action)

        # Apply feedback-based adjustments
        if self.feedback_service:
            try:
                # Check for similar past corrections that might apply
                correction_adjustments = self.feedback_service.get_feedback_for_reward_adjustment(
                    self.default_learner_id, getattr(result, 'response_id', None)
                )

                if 'penalty' in correction_adjustments:
                    base_reward *= (1.0 - correction_adjustments['penalty'] * 0.2)

                if 'boost' in correction_adjustments:
                    base_reward *= (1.0 + correction_adjustments['boost'] * 0.1)

                # Apply collaborative learning insights
                if self.reward_service:
                    collaborative_penalty = self.reward_service.get_correction_based_adjustments(response)
                    for obj, penalty in collaborative_penalty.items():
                        if obj in ['creativity', 'coherence', 'factuality']:
                            base_reward *= (1.0 - penalty * 0.1)

            except Exception as e:
                logger.warning(f"Failed to apply feedback adjustments to evaluation: {str(e)}")

        return max(0.0, min(1.0, base_reward))

    def _get_correction_influence(self):
        """Get influence from past corrections for prompt generation"""
        if not self.feedback_service:
            return False

        try:
            # Check if there are recent corrections that should influence learning
            feedback_history = self.feedback_service.get_feedback_history(
                type('Request', (), {
                    'user_id': self.default_learner_id,
                    'limit': 5,
                    'offset': 0,
                    'category_filter': None,
                    'feedback_type_filter': None
                })()
            )

            # Look for correction-type feedback
            corrections = [fb for fb in feedback_history.feedbacks
                          if hasattr(fb, 'correction_type') and fb.correction_type]

            return len(corrections) > 0

        except Exception as e:
            logger.warning(f"Failed to get correction influence: {str(e)}")
            return False

    def _should_end_ppo_episode(self, reward: float, result: Dict[str, Any], iteration_count: int) -> bool:
        """Determine if the PPO episode should end"""
        # End episode on very low reward (failure)
        if reward < 0.2:
            return True

        # End episode if hallucination detected
        if result.get('hallucination_analysis') and result['hallucination_analysis'].is_hallucinated:
            return True

        # End episode every 50 iterations for curriculum progression
        if iteration_count % 50 == 0:
            return True

        # Random early termination for exploration (5% chance)
        if random.random() < 0.05:
            return True

        return False