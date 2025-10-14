import sys
import os
import time
import logging
from typing import TYPE_CHECKING, Dict, Any, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.prompt import Prompt
from models.history import History
from models.learner import LLM
from models.curriculum_generator import CurriculumGenerator
from services.reward_service import RewardService
from services.hallucination_service import HallucinationService
from services.external_llm_service import ExternalLLMService
from utils.config import get_external_llm_config

if TYPE_CHECKING:
    from services.rl_training_service import RLTrainingService

logger = logging.getLogger(__name__)

class PromptController:
    def __init__(self, llm=None, history=None, reward_service=None, hallucination_service=None, external_llm_service=None, learning_loop_service=None):
        try:
            self.llm = llm or LLM(num_actions=10)
            self.history = history or History()
            self.reward_service = reward_service  # Can be None for backward compatibility
            self.hallucination_service = hallucination_service  # Can be None for backward compatibility

            # External LLM integration
            self.external_llm_config = get_external_llm_config()
            self.external_llm_service = external_llm_service
            if self.external_llm_config.enabled and not self.external_llm_service:
                self.external_llm_service = ExternalLLMService()

            # Track current LLM mode
            self.use_external_llm = self.external_llm_config.enabled and not self.external_llm_config.fallback_to_internal

            # Curriculum-driven learning components
            self.curriculum_generator = CurriculumGenerator()
            self._rl_training_service = None  # Lazy initialization to avoid circular import
            self.learning_loop_service = learning_loop_service  # Reference to learning loop service

            # Curriculum learning state
            self.curriculum_mode = False
            self.current_curriculum = None
            self.task_sequence = []
            self.current_task_index = 0

            logger.info("PromptController initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PromptController: {str(e)}")
            raise

    @property
    def rl_training_service(self) -> 'RLTrainingService':
        """Lazy initialization of RL training service to avoid circular imports"""
        if self._rl_training_service is None:
            from services.rl_training_service import RLTrainingService
            self._rl_training_service = RLTrainingService(
                prompt_controller=self,
                reward_service=self.reward_service
            )
        return self._rl_training_service

    def handle_prompt(self, prompt_text, source="user", use_external_llm=None, provider=None, model=None, user_id=None):
        # Initialize variables that will be used in history even on failure
        prompt = None
        response = ""
        action = 0
        reward_value = 0.0
        external_llm_response = None
        hallucination_analysis = None
        reward_result = None
        processing_error = None

        try:
            if not prompt_text or not isinstance(prompt_text, str):
                raise ValueError("Invalid prompt_text: must be a non-empty string")

            if len(prompt_text.strip()) == 0:
                raise ValueError("Prompt text cannot be empty or whitespace only")

            # Determine whether to use external LLM
            use_external = use_external_llm if use_external_llm is not None else self.use_external_llm

            logger.info(f"[ML Backend] Processing prompt from {source}: {prompt_text[:50]}...")
            logger.info(f"[ML Backend] Using external LLM: {use_external}")

            prompt = Prompt(prompt_text)

            if use_external and self.external_llm_service:
                try:
                    # Use external LLM
                    external_llm_response = self.external_llm_service.generate_response(
                        prompt=prompt.text,
                        provider=provider,
                        model=model
                    )

                    if asyncio.iscoroutine(external_llm_response):
                        import asyncio
                        external_llm_response = asyncio.run(external_llm_response)

                    if external_llm_response.error:
                        logger.warning(f"[ML Backend] External LLM error: {external_llm_response.error}")
                        if self.external_llm_config.fallback_to_internal:
                            logger.info("[ML Backend] Falling back to internal LLM")
                            use_external = False
                        else:
                            raise RuntimeError(f"External LLM failed: {external_llm_response.error}")

                    if not external_llm_response.error:
                        response = external_llm_response.response
                        # For external LLMs, we use a default action (could be improved)
                        action = 0

                except Exception as e:
                    logger.error(f"[ML Backend] External LLM failed: {str(e)}")
                    if self.external_llm_config.fallback_to_internal:
                        logger.info("[ML Backend] Falling back to internal LLM")
                        use_external = False
                    else:
                        raise RuntimeError(f"External LLM unavailable: {str(e)}")

            if not use_external:
                # Use internal LLM
                response, action = self.llm.process_prompt(prompt.text)

            # Calculate reward using multi-objective reward service if available
            if self.reward_service:
                # Generate response ID for feedback tracking
                import uuid
                response_id = str(uuid.uuid4())[:8] if user_id else None

                reward_result = self.reward_service.calculate_multi_objective_reward(
                    prompt.text, response, action, user_id=user_id, response_id=response_id
                )
                reward_value = reward_result['weighted_reward']  # For history/logging
            else:
                # Fallback to random reward for backward compatibility
                import random
                reward_value = random.random()

            # Perform hallucination analysis if service is available
            if self.hallucination_service:
                try:
                    hallucination_analysis = self.hallucination_service.analyze_response(
                        prompt.text, response
                    )
                    logger.info(f"[ML Backend] Hallucination analysis: confidence={hallucination_analysis.overall_confidence:.3f}, "
                                f"hallucinated={hallucination_analysis.is_hallucinated}")
                except Exception as e:
                    logger.warning(f"[ML Backend] Hallucination analysis failed: {str(e)}")

            # Meta-learning integration with external LLM awareness
            if hasattr(self, 'meta_learning_service') and self.meta_learning_service:
                try:
                    # Prepare context for meta-learning
                    context = {
                        'external_llm_used': use_external,
                        'source': source,
                        'timestamp': time.time()
                    }

                    # Extract external LLM info for meta-learning
                    external_llm_info = None
                    if external_llm_response and not external_llm_response.error:
                        external_llm_info = {
                            'provider': external_llm_response.provider,
                            'model': external_llm_response.model,
                            'cost': external_llm_response.cost,
                            'latency': external_llm_response.latency,
                            'input_tokens': external_llm_response.input_tokens,
                            'output_tokens': external_llm_response.output_tokens
                        }

                    # Monitor performance with meta-learning
                    self.meta_learning_service.monitor_performance(
                        prompt_text, response, action, reward_result or {'weighted_reward': reward_value},
                        context, external_llm_info
                    )
                    logger.info("[ML Backend] Meta-learning performance monitoring completed")
                except Exception as e:
                    logger.warning(f"[ML Backend] Meta-learning monitoring failed: {str(e)}")

            # Learn from the interaction (only for internal LLM)
            if not use_external:
                self.llm.learn(action, reward_value, prompt_text, response)

            # Hybrid learning: also learn from external LLM responses if enabled
            if use_external and self.external_llm_config.hybrid_learning_enabled and external_llm_response:
                try:
                    # Create a synthetic action based on external response quality
                    synthetic_action = self._calculate_synthetic_action(external_llm_response)
                    # Learn from external response (with reduced weight)
                    hybrid_reward = reward_result['weighted_reward'] * 0.3 if self.reward_service else reward_value * 0.3
                    self.llm.learn(synthetic_action, hybrid_reward, prompt_text, response)
                    logger.info("[ML Backend] Hybrid learning: updated internal model with external response")
                except Exception as e:
                    logger.warning(f"[ML Backend] Hybrid learning failed: {str(e)}")

            # Trigger curriculum generation for user prompts to start autonomous learning
            if source in ["user", "user_dashboard"] and not self.curriculum_mode:
                try:
                    logger.info(f"[ML Backend] User prompt detected, triggering curriculum generation from: {prompt_text[:50]}...")
                    curriculum_info = self.start_curriculum_learning(prompt_text, num_tasks=10)
                    logger.info(f"[ML Backend] Curriculum generated with {len(self.task_sequence)} tasks covering {len(curriculum_info['skills_covered'])} skills")

                    # Set the curriculum tasks for the learning loop service
                    # NOTE: Loop will be started automatically by receive_initial_prompt() in app.py
                    if hasattr(self, 'learning_loop_service') and self.learning_loop_service:
                        self.learning_loop_service.set_user_curriculum_tasks(self.task_sequence)
                        logger.info("[ML Backend] Curriculum tasks set for autonomous learning loop")
                    else:
                        logger.warning("[ML Backend] Learning loop service not available, curriculum generated but tasks not set")

                except Exception as e:
                    logger.error(f"[ML Backend] Failed to start curriculum learning from user prompt: {str(e)}")
                    # Don't fail the entire prompt processing if curriculum generation fails

            logger.info(f"[ML Backend] Prompt processed successfully, action: {action}, reward: {reward_value:.3f}, external: {use_external}")

        except ValueError as e:
            logger.warning(f"[ML Backend] Validation error in handle_prompt: {str(e)}")
            processing_error = str(e)
            raise
        except Exception as e:
            logger.error(f"[ML Backend] Unexpected error in handle_prompt: {str(e)}")
            processing_error = str(e)
            response = f"Error: {processing_error}"  # Set error message as response for history
            reward_value = 0.0  # Failed processing gets zero reward
            raise RuntimeError(f"Failed to process prompt: {str(e)}")
        finally:
            # Always add interaction to history, even on failure
            if prompt:
                if processing_error:
                    # For failed prompts, store error info
                    response = f"Processing failed: {processing_error}"
                    reward_value = 0.0
                    action = -1  # Special action indicating failure

                self.history.add_interaction(prompt, response, reward_value, action, source)
                logger.info(f"[ML Backend] History now has {len(self.history.interactions)} interactions")

            # Return response data including external LLM metadata
            result = {
                'response': response,
                'action': action,
                'hallucination_analysis': hallucination_analysis,
                'reward_result': reward_result if self.reward_service else None,
                'external_llm_used': use_external if 'use_external' in locals() else False,
                'external_llm_response': external_llm_response,
                'processing_error': processing_error
            }

            return result

    def _calculate_synthetic_action(self, external_response):
        """Calculate a synthetic action based on external LLM response quality."""
        # Simple heuristic: better responses get higher actions
        quality_score = 0

        if external_response.cost > 0:  # Successful response
            quality_score += 1

        if external_response.latency < 5.0:  # Fast response
            quality_score += 1

        if len(external_response.response.split()) > 10:  # Substantial response
            quality_score += 1

        # Map quality score to action (0-9 range)
        return min(quality_score, 9)

    def switch_llm_mode(self, use_external: bool) -> bool:
        """Switch between internal and external LLM modes."""
        try:
            if use_external and not self.external_llm_service:
                logger.warning("[ML Backend] External LLM service not available")
                return False

            self.use_external_llm = use_external
            logger.info(f"[ML Backend] Switched LLM mode to: {'external' if use_external else 'internal'}")
            return True
        except Exception as e:
            logger.error(f"[ML Backend] Failed to switch LLM mode: {str(e)}")
            return False

    def get_external_llm_models(self):
        """Get available external LLM models."""
        try:
            if not self.external_llm_service:
                return {}
            return self.external_llm_service.get_available_models()
        except Exception as e:
            logger.error(f"[ML Backend] Failed to get external LLM models: {str(e)}")
            return {}

    def get_external_llm_costs(self, provider=None):
        """Get external LLM cost metrics."""
        try:
            if not self.external_llm_service:
                return {}
            return self.external_llm_service.get_cost_metrics(provider)
        except Exception as e:
            logger.error(f"[ML Backend] Failed to get external LLM costs: {str(e)}")
            return {}

    async def compare_external_llm_models(self, prompt_text: str, providers_models: list):
        """Compare responses from multiple external LLM models."""
        try:
            if not self.external_llm_service:
                raise RuntimeError("External LLM service not available")

            return await self.external_llm_service.compare_models(prompt_text, providers_models)
        except Exception as e:
            logger.error(f"[ML Backend] Failed to compare external LLM models: {str(e)}")
            raise RuntimeError(f"Failed to compare models: {str(e)}")

    def get_llm_status(self):
        """Get current LLM configuration and status."""
        try:
            status = {
                'current_mode': 'external' if self.use_external_llm else 'internal',
                'external_llm_enabled': self.external_llm_config.enabled,
                'external_llm_available': self.external_llm_service is not None,
                'fallback_to_internal': self.external_llm_config.fallback_to_internal,
                'hybrid_learning_enabled': self.external_llm_config.hybrid_learning_enabled,
                'model_comparison_enabled': self.external_llm_config.model_comparison_enabled
            }

            if self.external_llm_service:
                status['provider_status'] = self.external_llm_service.get_provider_status()
                status['default_provider'] = self.external_llm_config.default_provider

            return status
        except Exception as e:
            logger.error(f"[ML Backend] Failed to get LLM status: {str(e)}")
            return {'error': str(e)}

    def get_learning_metrics(self):
        try:
            metrics = {
                'success_rate': self.history.get_success_rate(),
                'pattern_frequency': self.history.get_pattern_frequency(),
                'total_interactions': len(self.history.interactions)
            }

            # Add external LLM metrics if available
            if self.external_llm_service:
                external_costs = self.get_external_llm_costs()
                if external_costs:
                    metrics['external_llm_costs'] = external_costs

            # Add curriculum learning metrics
            if hasattr(self, 'rl_training_service') and self.rl_training_service:
                rl_status = self.rl_training_service.get_training_status()
                metrics['curriculum_learning'] = {
                    'is_training': rl_status.get('is_training', False),
                    'episodes': rl_status.get('episodes', 0),
                    'avg_reward': rl_status.get('current_avg_reward', 0),
                    'success_rate': rl_status.get('current_success_rate', 0),
                    'completed_skills': rl_status.get('completed_skills', 0),
                    'available_skills': rl_status.get('available_skills', 0)
                }

            logger.info(f"[ML Backend] Retrieved learning metrics: success_rate={metrics['success_rate']:.3f}, total_interactions={metrics['total_interactions']}")
            return metrics
        except Exception as e:
            logger.error(f"[ML Backend] Failed to get learning metrics: {str(e)}")
            raise RuntimeError(f"Failed to retrieve learning metrics: {str(e)}")

    def start_curriculum_learning(self, user_prompt: str, num_tasks: int = 5) -> Dict[str, Any]:
        """Start curriculum-driven learning from a user prompt"""
        try:
            logger.info(f"[ML Backend] Starting curriculum learning for prompt: {user_prompt[:50]}...")

            # Generate curriculum from user prompt
            self.current_curriculum = self.rl_training_service.generate_curriculum_from_prompt(user_prompt)

            # Generate progressive task sequence
            self.task_sequence = self.rl_training_service.get_progressive_tasks(user_prompt, num_tasks)
            self.current_task_index = 0
            self.curriculum_mode = True

            curriculum_info = {
                'total_tasks': len(self.task_sequence),
                'skills_covered': list(set(task['skill_id'] for task in self.task_sequence)),
                'difficulty_range': [task['difficulty'] for task in self.task_sequence],
                'estimated_completion_time': sum(task['estimated_time'] for task in self.task_sequence)
            }

            logger.info(f"[ML Backend] Curriculum generated: {len(self.task_sequence)} tasks, "
                       f"{len(curriculum_info['skills_covered'])} skills")
            return curriculum_info

        except Exception as e:
            logger.error(f"[ML Backend] Failed to start curriculum learning: {str(e)}")
            raise RuntimeError(f"Failed to start curriculum learning: {str(e)}")

    def get_next_curriculum_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task in the curriculum sequence"""
        if not self.curriculum_mode or not self.task_sequence:
            return None

        if self.current_task_index >= len(self.task_sequence):
            # Curriculum completed
            self.curriculum_mode = False
            return None

        task = self.task_sequence[self.current_task_index]
        self.current_task_index += 1

        logger.info(f"[ML Backend] Serving curriculum task {self.current_task_index}/{len(self.task_sequence)}: "
                   f"{task['task_description'][:50]}...")
        return task

    def process_curriculum_response(self, response: str, task_info: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        """Process a response to a curriculum task and update progress"""
        try:
            # Create a prompt for evaluation
            task_prompt = Prompt(task_info['task_description'])

            # Process through the normal pipeline but with curriculum context
            result = self.handle_prompt(
                task_prompt.text,
                source="curriculum_learning",
                user_id=user_id or "curriculum_learner"
            )

            # Update curriculum progress based on performance
            reward = 0.0
            if self.reward_service and result.get('reward_result'):
                reward = result['reward_result'].get('weighted_reward', 0.0)
            elif result.get('response'):
                # Basic reward calculation
                reward = min(1.0, len(result['response'].split()) / 50.0)  # Simple length-based reward

            # Update RL training service with curriculum progress
            if hasattr(self, 'rl_training_service'):
                curriculum_context = {
                    'current_skill': task_info.get('skill_id'),
                    'difficulty_level': ['easy', 'medium', 'hard', 'expert'].index(task_info.get('difficulty', 'easy')),
                    'task_progress': (self.current_task_index - 1) / len(self.task_sequence),
                    'success_streak': 1 if reward > 0.6 else 0
                }

                self.rl_training_service.ppo_agent.update_curriculum_state(
                    curriculum_context['current_skill'],
                    curriculum_context['difficulty_level'],
                    curriculum_context['task_progress'],
                    reward > 0.6
                )

            # Add curriculum-specific metadata
            result['curriculum_progress'] = {
                'task_completed': self.current_task_index,
                'total_tasks': len(self.task_sequence),
                'skill_practiced': task_info.get('skill_id'),
                'performance_score': reward,
                'learning_objectives': task_info.get('learning_objectives', [])
            }

            logger.info(f"[ML Backend] Curriculum task processed: performance={reward:.3f}, "
                       f"task {self.current_task_index}/{len(self.task_sequence)}")
            return result

        except Exception as e:
            logger.error(f"[ML Backend] Failed to process curriculum response: {str(e)}")
            raise RuntimeError(f"Failed to process curriculum response: {str(e)}")

    def get_curriculum_status(self) -> Dict[str, Any]:
        """Get current curriculum learning status"""
        if not self.curriculum_mode:
            return {'active': False}

        return {
            'active': True,
            'current_task': self.current_task_index,
            'total_tasks': len(self.task_sequence),
            'completion_percentage': (self.current_task_index / len(self.task_sequence)) * 100 if self.task_sequence else 0,
            'remaining_tasks': len(self.task_sequence) - self.current_task_index,
            'skills_covered': list(set(task['skill_id'] for task in self.task_sequence[:self.current_task_index])),
            'next_task': self.task_sequence[self.current_task_index] if self.current_task_index < len(self.task_sequence) else None
        }

    def stop_curriculum_learning(self):
        """Stop curriculum-driven learning"""
        self.curriculum_mode = False
        self.current_curriculum = None
        self.task_sequence = []
        self.current_task_index = 0
        logger.info("[ML Backend] Curriculum learning stopped")

    def start_rl_training(self):
        """Start RL training for curriculum learning"""
        if hasattr(self, 'rl_training_service'):
            self.rl_training_service.start_training()
            logger.info("[ML Backend] RL training started for curriculum learning")

    def stop_rl_training(self):
        """Stop RL training"""
        if hasattr(self, 'rl_training_service'):
            self.rl_training_service.stop_training()
            logger.info("[ML Backend] RL training stopped")

    def get_rl_training_status(self) -> Dict[str, Any]:
        """Get RL training status"""
        if hasattr(self, 'rl_training_service'):
            return self.rl_training_service.get_training_status()
        return {'error': 'RL training service not available'}