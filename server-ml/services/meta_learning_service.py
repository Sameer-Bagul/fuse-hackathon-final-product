import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import threading

from models.meta_learning import (
    MetaLearner, LearningStrategy, MetaMetrics,
    AdaptationRule
)
from services.reward_service import RewardService
from services.hallucination_service import HallucinationService
from controllers.scheduler import Scheduler
from models.curriculum import DifficultyLevel
from models.task_generator import TaskGenerator
from models.curriculum_generator import CurriculumGenerator
from models.ppo_agent import PPOAgent
from utils.logging_config import get_logger, log_curriculum_status

logger = get_logger(__name__)

class MetaLearningService:
    """
    Meta-Learning Service for autonomous curriculum learning engine.

    This service monitors learning performance across different strategies,
    adapts learning parameters based on performance, and implements
    learning-to-learn capabilities with transfer learning between tasks.
    """

    def __init__(self, reward_service: RewardService,
                 hallucination_service: HallucinationService,
                 scheduler: Scheduler, feedback_service=None,
                 task_generator=None, curriculum_generator=None, ppo_agent=None):
        self.reward_service = reward_service
        self.hallucination_service = hallucination_service
        self.scheduler = scheduler
        self.feedback_service = feedback_service
        self.task_generator = task_generator or TaskGenerator()
        self.curriculum_generator = curriculum_generator or CurriculumGenerator()
        self.ppo_agent = ppo_agent

        # Core meta-learning components
        self.meta_learner = MetaLearner()

        # Performance monitoring
        self.performance_window = deque(maxlen=100)  # Recent performance data
        self.strategy_performance = defaultdict(list)  # Strategy -> performance history
        self.parameter_adaptation_history = []

        # Curriculum performance tracking
        self.curriculum_performance = defaultdict(list)  # skill_id -> performance history
        self.difficulty_performance = defaultdict(list)  # difficulty -> performance history
        self.task_generation_feedback = []  # Track task generation effectiveness
        self.ppo_feedback_history = deque(maxlen=50)  # PPO feedback for adaptation

        # Learning context tracking
        self.current_context = {
            'difficulty': 'medium',
            'curriculum_completion': 0.0,
            'task_type': 'general',
            'learner_state': 'exploring'
        }

        # Adaptation scheduling
        self.adaptation_interval = 30  # Adapt every 30 seconds
        self.last_adaptation = time.time()
        self.is_adapting = False

        # Transfer learning memory
        self.task_transfer_memory = {}  # task_type -> learned_parameters
        self.skill_transfer_weights = {}  # skill_id -> transfer effectiveness

        logger.info("Meta-Learning Service initialized")

    def monitor_performance(self, prompt: str, response: str, action: int,
                           reward: Dict[str, Any], context: Dict[str, Any],
                           external_llm_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Monitor learning performance and update meta-learning metrics.

        Args:
            prompt: Input prompt text
            response: Generated response
            action: Action taken by LLM
            reward: Multi-objective reward dictionary
            context: Current learning context

        Returns:
            Performance analysis and adaptation recommendations
        """
        try:
            # Update current context
            self.current_context.update(context)

            # Calculate comprehensive performance metrics
            performance_metrics = self._calculate_performance_metrics(
                prompt, response, action, reward, external_llm_info
            )

            # Incorporate user feedback if available
            if self.feedback_service and context.get('user_id'):
                feedback_metrics = self._calculate_feedback_metrics(
                    context.get('user_id'), context.get('response_id'), response
                )
                performance_metrics.update(feedback_metrics)

            # Store performance data
            # Ensure reward is stored as a dict so downstream consumers can safely access keys
            if isinstance(reward, dict):
                reward_snapshot = reward.copy()
            else:
                # Normalize scalar rewards to a dict format
                try:
                    scalar_reward = float(reward)
                except Exception:
                    scalar_reward = 0.0
                reward_snapshot = {
                    'weighted_reward': scalar_reward,
                    'total_reward': scalar_reward,
                    'individual_rewards': {'single_reward': scalar_reward}
                }

            self.performance_window.append({
                'timestamp': time.time(),
                'metrics': performance_metrics,
                'context': context.copy() if isinstance(context, dict) else {},
                'reward': reward_snapshot,
                'external_llm_info': external_llm_info.copy() if isinstance(external_llm_info, dict) else external_llm_info
            })

            # Update strategy performance
            current_strategy = self.meta_learner.current_strategy
            self.strategy_performance[current_strategy.value].append(performance_metrics)

            # Keep only recent strategy performance (last 50 entries)
            if len(self.strategy_performance[current_strategy.value]) > 50:
                self.strategy_performance[current_strategy.value] = \
                    self.strategy_performance[current_strategy.value][-50:]

            # Update meta-learner metrics
            self.meta_learner.metrics.update_strategy_performance(
                current_strategy, performance_metrics
            )

            # Check if adaptation is needed
            adaptation_needed = self._should_adapt()

            analysis = {
                'performance_metrics': performance_metrics,
                'current_strategy': current_strategy.value,
                'adaptation_needed': adaptation_needed,
                'context_awareness': self.current_context.copy(),
                'strategy_effectiveness': self._calculate_strategy_effectiveness()
            }

            # Log meta-learning performance monitoring
            weighted_reward = performance_metrics.get('weighted_reward', 0.0)
            factuality_score = performance_metrics.get('factuality_score', 0.0)
            logger.info(f"🎯 Meta-Learning Performance | Strategy: {current_strategy.value} | "
                       f"Reward: {weighted_reward:.3f} | Factuality: {factuality_score:.3f} | "
                       f"Adaptation: {'Needed' if adaptation_needed else 'Not needed'}")

            # Trigger adaptation if needed
            if adaptation_needed and not self.is_adapting:
                logger.info("🔄 Triggering automatic meta-learning adaptation...")
                threading.Thread(
                    target=self._perform_adaptation,
                    args=(performance_metrics,),
                    daemon=True
                ).start()

            return analysis

        except Exception as e:
            logger.error(f"❌ Error monitoring performance: {str(e)}")
            return {
                'performance_metrics': {},
                'current_strategy': self.meta_learner.current_strategy.value,
                'adaptation_needed': False,
                'error': str(e)
            }

    def _calculate_performance_metrics(self, prompt: str, response: str,
                                       action: int, reward: Dict[str, Any],
                                       external_llm_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}

        # Handle both dict and float reward types
        if isinstance(reward, dict):
            # Multi-objective reward
            metrics['weighted_reward'] = reward.get('weighted_reward', 0.0)
            metrics['total_reward'] = reward.get('total_reward', 0.0)
            metrics['individual_rewards'] = reward.get('individual_rewards', {})
        else:
            # Single float reward - convert to dict format
            metrics['weighted_reward'] = float(reward)
            metrics['total_reward'] = float(reward)
            metrics['individual_rewards'] = {'single_reward': float(reward)}

        # Response quality metrics
        metrics['response_length'] = len(response.split())
        metrics['response_quality'] = self._assess_response_quality(prompt, response)

        # Learning efficiency metrics
        metrics['action_diversity'] = self._calculate_action_diversity(action)
        metrics['learning_progress'] = self._calculate_learning_progress()

        # Context-aware metrics
        metrics['curriculum_alignment'] = self._calculate_curriculum_alignment()
        metrics['task_complexity_match'] = self._assess_task_complexity_match(prompt)

        # Hallucination and factuality metrics
        hallucination_analysis = self.hallucination_service.analyze_response(prompt, response)
        metrics['hallucination_confidence'] = hallucination_analysis.overall_confidence
        metrics['factuality_score'] = 1.0 - hallucination_analysis.overall_confidence

        # External LLM specific metrics
        if external_llm_info:
            metrics['external_llm_used'] = True
            metrics['external_provider'] = external_llm_info.get('provider')
            metrics['external_model'] = external_llm_info.get('model')
            metrics['api_cost'] = external_llm_info.get('cost', 0.0)
            metrics['api_latency'] = external_llm_info.get('latency', 0.0)
            metrics['input_tokens'] = external_llm_info.get('input_tokens', 0)
            metrics['output_tokens'] = external_llm_info.get('output_tokens', 0)

            # Cost efficiency metric
            if metrics['api_cost'] > 0:
                metrics['cost_per_token'] = metrics['api_cost'] / (metrics['input_tokens'] + metrics['output_tokens'])
                metrics['cost_efficiency'] = 1.0 / (1.0 + metrics['api_cost'])  # Higher is better
            else:
                metrics['cost_per_token'] = 0.0
                metrics['cost_efficiency'] = 1.0

            # Adjust success indicators for external LLMs
            if metrics['api_latency'] < 2.0:  # Fast response
                metrics['success_indicators']['fast_response'] = True
            else:
                metrics['success_indicators']['fast_response'] = False

            if metrics['api_cost'] < 0.01:  # Cost-effective
                metrics['success_indicators']['cost_effective'] = True
            else:
                metrics['success_indicators']['cost_effective'] = False
        else:
            metrics['external_llm_used'] = False
            metrics['api_cost'] = 0.0
            metrics['api_latency'] = 0.0
            metrics['cost_efficiency'] = 1.0

        # Success indicators
        metrics['success_indicators'] = self._calculate_success_indicators(metrics)

        return metrics

    def _assess_response_quality(self, prompt: str, response: str) -> float:
        """Assess the quality of the response relative to the prompt"""
        # Simple heuristic-based quality assessment
        quality_score = 0.5  # Base score

        # Length appropriateness
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            quality_score += 0.2
        elif word_count < 5:
            quality_score -= 0.2

        # Keyword relevance
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap_ratio = len(prompt_words.intersection(response_words)) / max(len(prompt_words), 1)
        quality_score += overlap_ratio * 0.3

        # Structure indicators
        if '?' in response or '!' in response:
            quality_score += 0.1  # Shows engagement

        return max(0.0, min(1.0, quality_score))

    def _calculate_action_diversity(self, current_action: int) -> float:
        """Calculate action diversity based on recent actions"""
        if not self.performance_window:
            return 0.5

        recent_actions = [entry['metrics'].get('action', 0)
                         for entry in list(self.performance_window)[-10:]]

        if not recent_actions:
            return 0.5

        # Calculate diversity as 1 - (most_common_action_count / total_actions)
        action_counts = defaultdict(int)
        for action in recent_actions:
            action_counts[action] += 1

        most_common_count = max(action_counts.values())
        diversity = 1.0 - (most_common_count / len(recent_actions))

        return diversity

    def _calculate_learning_progress(self) -> float:
        """Calculate overall learning progress"""
        if len(self.performance_window) < 5:
            return 0.0

        recent_entries = list(self.performance_window)[-10:]
        recent_rewards = [entry['reward'].get('weighted_reward', 0.0) for entry in recent_entries]

        # Simple progress metric: improvement over time
        if len(recent_rewards) >= 2:
            early_avg = sum(recent_rewards[:len(recent_rewards)//2]) / (len(recent_rewards)//2)
            late_avg = sum(recent_rewards[len(recent_rewards)//2:]) / (len(recent_rewards)//2)
            progress = late_avg - early_avg
            return max(-1.0, min(1.0, progress))  # Clamp to [-1, 1]

        return 0.0

    def _calculate_curriculum_alignment(self) -> float:
        """Calculate how well current learning aligns with curriculum goals"""
        try:
            learner_id = getattr(self.scheduler, 'default_learner_id', 'autonomous_agent')
            progress = self.scheduler.get_learner_progress(learner_id)

            if progress and hasattr(progress, 'completed_skills'):
                total_skills = len(self.scheduler.curriculum_tree.skills)
                completed_skills = len(progress.completed_skills)
                alignment = completed_skills / max(total_skills, 1)
                return min(alignment, 1.0)

        except Exception as e:
            logger.debug(f"Could not calculate curriculum alignment: {e}")

        return 0.0

    def _assess_task_complexity_match(self, prompt: str) -> float:
        """Assess if task complexity matches learner capability"""
        # Simple heuristic based on prompt length and keywords
        complexity_indicators = ['advanced', 'complex', 'difficult', 'expert', 'optimize', 'design']
        prompt_lower = prompt.lower()

        complexity_score = 0.0
        for indicator in complexity_indicators:
            if indicator in prompt_lower:
                complexity_score += 0.2

        # Length-based complexity
        word_count = len(prompt.split())
        if word_count > 20:
            complexity_score += 0.3
        elif word_count > 10:
            complexity_score += 0.1

        return min(complexity_score, 1.0)

    def _calculate_feedback_metrics(self, user_id: str, response_id: str, response: str) -> Dict[str, Any]:
        """Calculate feedback-based performance metrics"""
        feedback_metrics = {}

        try:
            # Get user preferences for personalization insights
            preferences = self.feedback_service.get_user_preferences(
                type('Request', (), {'user_id': user_id, 'include_history': False})()
            ).preferences

            # Calculate preference alignment score
            feedback_metrics['preference_alignment'] = self._calculate_preference_alignment(
                response, preferences
            )

            # Get feedback history for user engagement metrics
            feedback_history = self.feedback_service.get_feedback_history(
                type('Request', (), {
                    'user_id': user_id,
                    'offset': 0,
                    'limit': 10,
                    'category_filter': None,
                    'feedback_type_filter': None
                })()
            )

            if feedback_history.feedbacks:
                # Calculate user engagement score
                feedback_metrics['user_engagement'] = min(len(feedback_history.feedbacks) / 10.0, 1.0)

                # Calculate average user satisfaction
                ratings = [fb.rating for fb in feedback_history.feedbacks if fb.rating]
                if ratings:
                    feedback_metrics['user_satisfaction'] = sum(ratings) / len(ratings) / 5.0  # Normalize to 0-1

                # Calculate feedback consistency (how consistent user ratings are)
                if len(ratings) > 1:
                    rating_variance = sum((r - sum(ratings)/len(ratings))**2 for r in ratings) / len(ratings)
                    feedback_metrics['feedback_consistency'] = max(0.0, 1.0 - rating_variance / 4.0)  # Lower variance = more consistent

            # Get collaborative insights
            collaborative_insights = self.feedback_service.get_collaborative_insights()
            feedback_metrics['collaborative_insights_count'] = len(collaborative_insights)

            # Calculate collaborative learning effectiveness
            if collaborative_insights:
                avg_confidence = sum(ci.confidence_score for ci in collaborative_insights) / len(collaborative_insights)
                feedback_metrics['collaborative_learning_effectiveness'] = avg_confidence

        except Exception as e:
            logger.warning(f"Error calculating feedback metrics: {str(e)}")
            feedback_metrics['feedback_error'] = str(e)

        return feedback_metrics

    def _calculate_preference_alignment(self, response: str, preferences) -> float:
        """Calculate how well the response aligns with user preferences"""
        if not hasattr(preferences, 'preferred_styles') or not preferences.preferred_styles:
            return 0.5  # Neutral score if no preferences

        alignment_score = 0.0
        response_lower = response.lower()

        # Check style alignment
        for style in preferences.preferred_styles:
            style_indicators = {
                'concise': ['brief', 'short', 'concise', 'to the point'],
                'detailed': ['detailed', 'comprehensive', 'thorough', 'in-depth'],
                'formal': ['therefore', 'however', 'moreover', 'consequently'],
                'casual': ['hey', 'okay', 'sure', 'you know'],
                'technical': ['algorithm', 'function', 'method', 'implementation'],
                'simple': ['easy', 'simple', 'basic', 'straightforward']
            }

            if style in style_indicators:
                matches = sum(1 for indicator in style_indicators[style] if indicator in response_lower)
                if matches > 0:
                    alignment_score += 0.2

        # Check disliked patterns (reduce alignment)
        if hasattr(preferences, 'disliked_patterns') and preferences.disliked_patterns:
            for pattern in preferences.disliked_patterns:
                if pattern.lower() in response_lower:
                    alignment_score -= 0.3

        return max(0.0, min(1.0, alignment_score))

    def analyze_curriculum_performance(self, learner_id: str, skill_id: str = None) -> Dict[str, Any]:
        """Analyze curriculum performance for adaptation decisions"""
        try:
            learner_progress = self.scheduler.get_learner_progress(learner_id)
            if not learner_progress:
                return {'error': 'No learner progress found'}

            analysis = {
                'overall_performance': {},
                'skill_performance': {},
                'difficulty_performance': {},
                'learning_trends': {},
                'adaptation_recommendations': {}
            }

            # Overall curriculum performance
            total_skills = len(self.scheduler.curriculum_tree.skills)
            completed_skills = len(learner_progress.completed_skills)
            analysis['overall_performance'] = {
                'completion_rate': completed_skills / max(total_skills, 1),
                'current_difficulty': learner_progress.current_difficulty.value,
                'recommended_difficulty': learner_progress.get_recommended_difficulty().value,
                'average_mastery': sum(learner_progress.get_skill_mastery_level(s) for s in self.scheduler.curriculum_tree.skills.keys()) / max(total_skills, 1)
            }

            # Skill-specific performance analysis
            if skill_id:
                skill_performance = self._analyze_skill_performance(learner_progress, skill_id)
                analysis['skill_performance'][skill_id] = skill_performance
            else:
                # Analyze all skills
                for s_id in self.scheduler.curriculum_tree.skills.keys():
                    analysis['skill_performance'][s_id] = self._analyze_skill_performance(learner_progress, s_id)

            # Difficulty-based performance analysis
            analysis['difficulty_performance'] = self._analyze_difficulty_performance(learner_progress)

            # Learning trends analysis
            analysis['learning_trends'] = self._analyze_learning_trends(learner_progress)

            # Generate adaptation recommendations
            analysis['adaptation_recommendations'] = self._generate_curriculum_adaptations(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing curriculum performance: {str(e)}")
            return {'error': str(e)}

    def _analyze_skill_performance(self, learner_progress, skill_id: str) -> Dict[str, Any]:
        """Analyze performance on a specific skill"""
        mastery_level = learner_progress.get_skill_mastery_level(skill_id)
        attempts = learner_progress.skill_attempts.get(skill_id, 0)
        recent_performance = self.curriculum_performance[skill_id][-10:] if skill_id in self.curriculum_performance else []

        # Calculate performance trend
        trend = 'stable'
        if len(recent_performance) >= 3:
            early_avg = sum(recent_performance[:len(recent_performance)//2]) / (len(recent_performance)//2)
            late_avg = sum(recent_performance[len(recent_performance)//2:]) / (len(recent_performance)//2)
            if late_avg > early_avg + 0.1:
                trend = 'improving'
            elif late_avg < early_avg - 0.1:
                trend = 'declining'

        return {
            'mastery_level': mastery_level,
            'attempts': attempts,
            'recent_performance': recent_performance,
            'performance_trend': trend,
            'needs_focus': mastery_level < 0.6 or trend == 'declining'
        }

    def _analyze_difficulty_performance(self, learner_progress) -> Dict[str, Any]:
        """Analyze performance across difficulty levels"""
        difficulty_stats = {}

        for difficulty in DifficultyLevel:
            # Get skills at this difficulty
            skills_at_difficulty = [s for s, skill in self.scheduler.curriculum_tree.skills.items()
                                  if skill.difficulty == difficulty]

            if skills_at_difficulty:
                mastery_levels = [learner_progress.get_skill_mastery_level(s) for s in skills_at_difficulty]
                avg_mastery = sum(mastery_levels) / len(mastery_levels)

                difficulty_stats[difficulty.value] = {
                    'average_mastery': avg_mastery,
                    'skills_count': len(skills_at_difficulty),
                    'completed_count': sum(1 for s in skills_at_difficulty if learner_progress.is_skill_mastered(s)),
                    'performance_trend': self.difficulty_performance[difficulty.value][-5:] if difficulty.value in self.difficulty_performance else []
                }

        return difficulty_stats

    def _analyze_learning_trends(self, learner_progress) -> Dict[str, Any]:
        """Analyze learning trends for curriculum adaptation"""
        trends = {
            'plateau_detected': False,
            'rapid_improvement': False,
            'struggling_areas': [],
            'ready_for_advance': False
        }

        # Check for performance plateaus
        recent_mastery = []
        for skill_id in self.scheduler.curriculum_tree.skills.keys():
            recent_perf = self.curriculum_performance[skill_id][-5:] if skill_id in self.curriculum_performance else []
            if recent_perf:
                recent_mastery.extend(recent_perf)

        if len(recent_mastery) >= 10:
            # Check if performance has plateaued (variance < 0.05 in recent performance)
            variance = sum((x - sum(recent_mastery)/len(recent_mastery))**2 for x in recent_mastery) / len(recent_mastery)
            trends['plateau_detected'] = variance < 0.05

        # Check for rapid improvement
        if len(recent_mastery) >= 5:
            early_avg = sum(recent_mastery[:len(recent_mastery)//2]) / (len(recent_mastery)//2)
            late_avg = sum(recent_mastery[len(recent_mastery)//2:]) / (len(recent_mastery)//2)
            trends['rapid_improvement'] = late_avg > early_avg + 0.15

        # Identify struggling areas
        for skill_id, skill in self.scheduler.curriculum_tree.skills.items():
            mastery = learner_progress.get_skill_mastery_level(skill_id)
            if mastery < 0.4:
                trends['struggling_areas'].append({
                    'skill_id': skill_id,
                    'mastery': mastery,
                    'difficulty': skill.difficulty.value
                })

        # Check if ready for difficulty advancement
        current_difficulty = learner_progress.current_difficulty
        next_difficulty = DifficultyLevel(min(current_difficulty.value + 1, 4))  # Max difficulty 4
        if next_difficulty != current_difficulty:
            next_difficulty_skills = [s for s, skill in self.scheduler.curriculum_tree.skills.items()
                                    if skill.difficulty == next_difficulty]
            if next_difficulty_skills:
                avg_next_mastery = sum(learner_progress.get_skill_mastery_level(s) for s in next_difficulty_skills) / len(next_difficulty_skills)
                trends['ready_for_advance'] = avg_next_mastery > 0.7

        return trends

    def _generate_curriculum_adaptations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate curriculum adaptation recommendations"""
        recommendations = {
            'difficulty_adjustments': [],
            'skill_focus_changes': [],
            'task_generation_changes': [],
            'strategy_modifications': []
        }

        # Difficulty adjustments
        overall_perf = analysis['overall_performance']
        if overall_perf['average_mastery'] > 0.8 and overall_perf['current_difficulty'] < overall_perf['recommended_difficulty']:
            recommendations['difficulty_adjustments'].append({
                'action': 'increase_difficulty',
                'reason': 'High mastery levels indicate readiness for harder tasks'
            })
        elif overall_perf['average_mastery'] < 0.4:
            recommendations['difficulty_adjustments'].append({
                'action': 'decrease_difficulty',
                'reason': 'Low mastery levels suggest need for easier tasks'
            })

        # Skill focus changes
        for skill_id, skill_perf in analysis['skill_performance'].items():
            if skill_perf['needs_focus']:
                recommendations['skill_focus_changes'].append({
                    'skill_id': skill_id,
                    'action': 'increase_focus',
                    'reason': f"Low mastery ({skill_perf['mastery_level']:.2f}) or declining trend"
                })

        # Task generation changes based on learning trends
        trends = analysis['learning_trends']
        if trends['plateau_detected']:
            recommendations['task_generation_changes'].append({
                'action': 'increase_variety',
                'reason': 'Performance plateau detected, need more diverse tasks'
            })
        if trends['rapid_improvement']:
            recommendations['task_generation_changes'].append({
                'action': 'accelerate_progression',
                'reason': 'Rapid improvement suggests faster curriculum advancement'
            })

        return recommendations

    def collect_ppo_feedback(self, state: np.ndarray, action: int, reward: float,
                           next_state: np.ndarray, done: bool, learner_id: str,
                           skill_id: str = None, difficulty: str = None):
        """Collect PPO feedback for curriculum adaptation analysis"""
        feedback_entry = {
            'timestamp': time.time(),
            'learner_id': learner_id,
            'skill_id': skill_id,
            'difficulty': difficulty,
            'state': state.tolist() if hasattr(state, 'tolist') else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist() if hasattr(next_state, 'tolist') else next_state,
            'done': done,
            'ppo_action_probs': self.ppo_agent.get_action_probabilities(state) if self.ppo_agent else None,
            'value_estimate': self.ppo_agent.get_value_estimate(state) if self.ppo_agent else None
        }

        self.ppo_feedback_history.append(feedback_entry)

        # Update curriculum performance tracking
        if skill_id:
            self.curriculum_performance[skill_id].append(reward)
            # Keep only recent performance data
            if len(self.curriculum_performance[skill_id]) > 20:
                self.curriculum_performance[skill_id] = self.curriculum_performance[skill_id][-20:]

        if difficulty:
            self.difficulty_performance[difficulty].append(reward)
            # Keep only recent performance data
            if len(self.difficulty_performance[difficulty]) > 20:
                self.difficulty_performance[difficulty] = self.difficulty_performance[difficulty][-20:]

    def analyze_ppo_feedback_for_adaptation(self) -> Dict[str, Any]:
        """Analyze PPO feedback to inform curriculum adaptation decisions"""
        if not self.ppo_feedback_history:
            return {'error': 'No PPO feedback available'}

        analysis = {
            'action_effectiveness': {},
            'state_transitions': {},
            'reward_patterns': {},
            'curriculum_insights': {},
            'adaptation_signals': {}
        }

        # Analyze action effectiveness
        action_rewards = defaultdict(list)
        for feedback in self.ppo_feedback_history:
            action_rewards[feedback['action']].append(feedback['reward'])

        analysis['action_effectiveness'] = {
            action: {
                'avg_reward': sum(rewards) / len(rewards),
                'count': len(rewards),
                'consistency': 1.0 - (np.std(rewards) / max(abs(np.mean(rewards)), 0.1)) if rewards else 0.0
            }
            for action, rewards in action_rewards.items()
        }

        # Analyze reward patterns by skill and difficulty
        skill_rewards = defaultdict(list)
        difficulty_rewards = defaultdict(list)

        for feedback in self.ppo_feedback_history:
            if feedback.get('skill_id'):
                skill_rewards[feedback['skill_id']].append(feedback['reward'])
            if feedback.get('difficulty'):
                difficulty_rewards[feedback['difficulty']].append(feedback['reward'])

        analysis['reward_patterns'] = {
            'by_skill': {skill: sum(rewards)/len(rewards) for skill, rewards in skill_rewards.items()},
            'by_difficulty': {diff: sum(rewards)/len(rewards) for diff, rewards in difficulty_rewards.items()}
        }

        # Generate curriculum insights from PPO feedback
        analysis['curriculum_insights'] = self._extract_curriculum_insights_from_ppo()

        # Generate adaptation signals
        analysis['adaptation_signals'] = self._generate_ppo_adaptation_signals(analysis)

        return analysis

    def _extract_curriculum_insights_from_ppo(self) -> Dict[str, Any]:
        """Extract curriculum insights from PPO feedback patterns"""
        insights = {
            'skill_difficulty_mismatch': [],
            'action_preference_shifts': {},
            'learning_plateaus': [],
            'optimal_action_sequences': {}
        }

        # Identify skill-difficulty mismatches
        skill_difficulty_rewards = defaultdict(lambda: defaultdict(list))
        for feedback in self.ppo_feedback_history:
            if feedback.get('skill_id') and feedback.get('difficulty'):
                skill_difficulty_rewards[feedback['skill_id']][feedback['difficulty']].append(feedback['reward'])

        for skill_id, difficulty_rewards in skill_difficulty_rewards.items():
            for difficulty, rewards in difficulty_rewards.items():
                avg_reward = sum(rewards) / len(rewards)
                if avg_reward < 0.3:  # Low performance threshold
                    insights['skill_difficulty_mismatch'].append({
                        'skill_id': skill_id,
                        'difficulty': difficulty,
                        'avg_reward': avg_reward,
                        'recommendation': 'reduce_difficulty' if difficulty != 'easy' else 'focus_on_fundamentals'
                    })

        # Analyze action preference shifts over time
        recent_feedback = list(self.ppo_feedback_history)[-20:]  # Last 20 entries
        earlier_feedback = list(self.ppo_feedback_history)[:-20] if len(self.ppo_feedback_history) > 20 else []

        if recent_feedback and earlier_feedback:
            recent_action_dist = defaultdict(int)
            earlier_action_dist = defaultdict(int)

            for fb in recent_feedback:
                recent_action_dist[fb['action']] += 1
            for fb in earlier_feedback:
                earlier_action_dist[fb['action']] += 1

            # Calculate preference shifts
            for action in set(list(recent_action_dist.keys()) + list(earlier_action_dist.keys())):
                recent_pct = recent_action_dist[action] / len(recent_feedback)
                earlier_pct = earlier_action_dist[action] / len(earlier_feedback)
                shift = recent_pct - earlier_pct
                if abs(shift) > 0.1:  # Significant shift threshold
                    insights['action_preference_shifts'][action] = {
                        'shift': shift,
                        'recent_preference': recent_pct,
                        'earlier_preference': earlier_pct
                    }

        return insights

    def _generate_ppo_adaptation_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptation signals based on PPO analysis"""
        signals = {
            'curriculum_adjustments': [],
            'task_generation_modifications': [],
            'strategy_changes': [],
            'difficulty_recalibrations': []
        }

        # Curriculum adjustments based on skill-difficulty mismatches
        for mismatch in analysis['curriculum_insights']['skill_difficulty_mismatch']:
            signals['curriculum_adjustments'].append({
                'skill_id': mismatch['skill_id'],
                'action': mismatch['recommendation'],
                'reason': f'Low PPO reward ({mismatch["avg_reward"]:.2f}) at {mismatch["difficulty"]} difficulty'
            })

        # Task generation modifications based on action effectiveness
        action_effectiveness = analysis['action_effectiveness']
        if action_effectiveness:
            # Identify most and least effective actions
            sorted_actions = sorted(action_effectiveness.items(),
                                  key=lambda x: x[1]['avg_reward'], reverse=True)

            best_action = sorted_actions[0][0]
            worst_action = sorted_actions[-1][0]

            signals['task_generation_modifications'].extend([
                {
                    'action': 'boost_effective_actions',
                    'target_action': best_action,
                    'reason': f'Action {best_action} shows high effectiveness ({action_effectiveness[best_action]["avg_reward"]:.2f})'
                },
                {
                    'action': 'reduce_ineffective_actions',
                    'target_action': worst_action,
                    'reason': f'Action {worst_action} shows low effectiveness ({action_effectiveness[worst_action]["avg_reward"]:.2f})'
                }
            ])

        # Strategy changes based on action preference shifts
        for action, shift_data in analysis['curriculum_insights']['action_preference_shifts'].items():
            if shift_data['shift'] > 0.1:  # Increasing preference
                signals['strategy_changes'].append({
                    'action': 'increase_action_weight',
                    'target_action': action,
                    'reason': f'Increasing preference for action {action} (+{shift_data["shift"]:.2f})'
                })

        return signals

    def adapt_task_generation_from_feedback(self, curriculum_analysis: Dict[str, Any],
                                          ppo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt task generation strategies based on curriculum and PPO feedback"""
        adaptations = {
            'task_generator_changes': {},
            'curriculum_generator_changes': {},
            'ppo_agent_adjustments': {},
            'applied_changes': []
        }

        try:
            # Adapt task generator based on curriculum performance
            curriculum_recs = curriculum_analysis.get('adaptation_recommendations', {})

            # Apply skill focus changes
            for focus_change in curriculum_recs.get('skill_focus_changes', []):
                skill_id = focus_change['skill_id']
                if focus_change['action'] == 'increase_focus':
                    self.task_generator.update_skill_focus(skill_id, 2.0)  # Double the focus
                    adaptations['task_generator_changes'][f'skill_focus_{skill_id}'] = 2.0
                    adaptations['applied_changes'].append(f"Increased focus on skill: {skill_id}")

            # Apply difficulty adjustments
            for diff_adjust in curriculum_recs.get('difficulty_adjustments', []):
                if diff_adjust['action'] == 'increase_difficulty':
                    # This would be handled by the scheduler, but we can influence task generation
                    adaptations['task_generator_changes']['difficulty_boost'] = 1.2
                    adaptations['applied_changes'].append("Increased task difficulty scaling")
                elif diff_adjust['action'] == 'decrease_difficulty':
                    adaptations['task_generator_changes']['difficulty_boost'] = 0.8
                    adaptations['applied_changes'].append("Decreased task difficulty scaling")

            # Adapt based on PPO feedback
            ppo_signals = ppo_analysis.get('adaptation_signals', {})

            # Apply task generation modifications from PPO
            for mod in ppo_signals.get('task_generation_modifications', []):
                if mod['action'] == 'boost_effective_actions':
                    # This would require modifying the task generator's action mapping
                    adaptations['task_generator_changes'][f'boost_action_{mod["target_action"]}'] = 1.5
                    adaptations['applied_changes'].append(f"Boosted effective action: {mod['target_action']}")
                elif mod['action'] == 'reduce_ineffective_actions':
                    adaptations['task_generator_changes'][f'reduce_action_{mod["target_action"]}'] = 0.5
                    adaptations['applied_changes'].append(f"Reduced ineffective action: {mod['target_action']}")

            # Apply PPO agent adjustments if needed
            if self.ppo_agent and ppo_signals.get('strategy_changes'):
                for strategy_change in ppo_signals['strategy_changes']:
                    if strategy_change['action'] == 'increase_action_weight':
                        # This would modify PPO agent's action preferences
                        adaptations['ppo_agent_adjustments'][f'action_weight_{strategy_change["target_action"]}'] = 1.2
                        adaptations['applied_changes'].append(f"Adjusted PPO action weight: {strategy_change['target_action']}")

            logger.info(f"Applied {len(adaptations['applied_changes'])} task generation adaptations")
            return adaptations

        except Exception as e:
            logger.error(f"Error adapting task generation: {str(e)}")
            return {'error': str(e)}

    def execute_curriculum_feedback_loop(self, learner_id: str) -> Dict[str, Any]:
        """Execute the complete curriculum feedback loop for continuous adaptation"""
        feedback_loop_result = {
            'timestamp': time.time(),
            'learner_id': learner_id,
            'analyses_performed': [],
            'adaptations_applied': [],
            'success': False,
            'error': None
        }

        try:
            logger.info(f"🔄 Executing curriculum feedback loop for learner: {learner_id}")

            # Step 1: Analyze curriculum performance
            curriculum_analysis = self.analyze_curriculum_performance(learner_id)
            if 'error' not in curriculum_analysis:
                feedback_loop_result['analyses_performed'].append('curriculum_performance')
                logger.info("✅ Curriculum performance analysis completed")
            else:
                logger.warning(f"⚠️  Curriculum analysis failed: {curriculum_analysis['error']}")

            # Step 2: Analyze PPO feedback
            if self.ppo_agent and self.ppo_feedback_history:
                ppo_analysis = self.analyze_ppo_feedback_for_adaptation()
                if 'error' not in ppo_analysis:
                    feedback_loop_result['analyses_performed'].append('ppo_feedback')
                    logger.info("✅ PPO feedback analysis completed")
                else:
                    logger.warning(f"⚠️  PPO analysis failed: {ppo_analysis['error']}")
                    ppo_analysis = {}
            else:
                ppo_analysis = {}
                logger.info("ℹ️  No PPO agent or feedback available, skipping PPO analysis")

            # Step 3: Generate and apply adaptations
            if 'error' not in curriculum_analysis:
                adaptations = self.adapt_task_generation_from_feedback(curriculum_analysis, ppo_analysis)
                if 'error' not in adaptations:
                    feedback_loop_result['adaptations_applied'] = adaptations.get('applied_changes', [])
                    logger.info(f"✅ Applied {len(feedback_loop_result['adaptations_applied'])} adaptations")
                else:
                    logger.warning(f"⚠️  Adaptation application failed: {adaptations['error']}")

            # Step 4: Update meta-learning strategy if needed
            strategy_adapted = self._adapt_meta_learning_strategy(curriculum_analysis, ppo_analysis)
            if strategy_adapted:
                feedback_loop_result['adaptations_applied'].append("Meta-learning strategy updated")
                logger.info("✅ Meta-learning strategy adapted")

            # Step 5: Log curriculum status update
            log_curriculum_status(learner_id, "feedback_loop_executed", {
                'analyses': feedback_loop_result['analyses_performed'],
                'adaptations': len(feedback_loop_result['adaptations_applied']),
                'curriculum_completion': curriculum_analysis.get('overall_performance', {}).get('completion_rate', 0)
            })

            feedback_loop_result['success'] = True
            logger.info(f"🎯 Curriculum feedback loop completed successfully for learner: {learner_id}")

            return feedback_loop_result

        except Exception as e:
            error_msg = f"Error in curriculum feedback loop: {str(e)}"
            logger.error(f"❌ {error_msg}")
            feedback_loop_result['error'] = error_msg
            return feedback_loop_result

    def _adapt_meta_learning_strategy(self, curriculum_analysis: Dict[str, Any],
                                    ppo_analysis: Dict[str, Any]) -> bool:
        """Adapt meta-learning strategy based on curriculum and PPO analysis"""
        try:
            current_strategy = self.meta_learner.current_strategy
            strategy_changed = False

            # Analyze if strategy change is needed based on performance patterns
            curriculum_perf = curriculum_analysis.get('overall_performance', {})
            learning_trends = curriculum_analysis.get('learning_trends', {})

            # Strategy adaptation logic
            if learning_trends.get('plateau_detected', False):
                # Switch to exploration-focused strategy when plateau detected
                if current_strategy != LearningStrategy.EXPLORATION_FOCUSED:
                    self.meta_learner.current_strategy = LearningStrategy.EXPLORATION_FOCUSED
                    strategy_changed = True
                    logger.info("Switched to exploration-focused strategy due to performance plateau")

            elif learning_trends.get('rapid_improvement', False):
                # Switch to exploitation-focused when rapid improvement detected
                if current_strategy != LearningStrategy.EXPLOITATION_FOCUSED:
                    self.meta_learner.current_strategy = LearningStrategy.EXPLOITATION_FOCUSED
                    strategy_changed = True
                    logger.info("Switched to exploitation-focused strategy due to rapid improvement")

            elif curriculum_perf.get('average_mastery', 0) > 0.8:
                # Switch to reward-optimized for high performers
                if current_strategy != LearningStrategy.REWARD_OPTIMIZED:
                    self.meta_learner.current_strategy = LearningStrategy.REWARD_OPTIMIZED
                    strategy_changed = True
                    logger.info("Switched to reward-optimized strategy due to high mastery levels")

            # PPO-based strategy adaptation
            if ppo_analysis and 'action_effectiveness' in ppo_analysis:
                action_eff = ppo_analysis['action_effectiveness']
                if action_eff:
                    # If actions are highly consistent, switch to adaptive strategy
                    avg_consistency = sum(a['consistency'] for a in action_eff.values()) / len(action_eff)
                    if avg_consistency > 0.8 and current_strategy != LearningStrategy.ADAPTIVE:
                        self.meta_learner.current_strategy = LearningStrategy.ADAPTIVE
                        strategy_changed = True
                        logger.info("Switched to adaptive strategy due to high action consistency")

            return strategy_changed

        except Exception as e:
            logger.error(f"Error adapting meta-learning strategy: {str(e)}")
            return False

    def get_curriculum_feedback_loop_status(self) -> Dict[str, Any]:
        """Get the current status of the curriculum feedback loop system"""
        return {
            'curriculum_performance_tracking': {
                'skills_tracked': len(self.curriculum_performance),
                'total_performance_entries': sum(len(perf) for perf in self.curriculum_performance.values())
            },
            'difficulty_performance_tracking': {
                'difficulties_tracked': len(self.difficulty_performance),
                'total_entries': sum(len(perf) for perf in self.difficulty_performance.values())
            },
            'ppo_feedback_tracking': {
                'feedback_entries': len(self.ppo_feedback_history),
                'ppo_agent_available': self.ppo_agent is not None
            },
            'task_generation_adaptations': {
                'skill_focus_weights': dict(self.task_generator.skill_focus_weights) if hasattr(self.task_generator, 'skill_focus_weights') else {}
            },
            'meta_learning_strategy': self.meta_learner.current_strategy.value if self.meta_learner else None,
            'last_adaptation': self.last_adaptation
        }

    def _calculate_success_indicators(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Calculate boolean success indicators"""
        return {
            'high_reward': metrics.get('weighted_reward', 0.0) > 0.7,
            'good_factuality': metrics.get('factuality_score', 0.0) > 0.8,
            'appropriate_length': 10 <= metrics.get('response_length', 0) <= 100,
            'curriculum_aligned': metrics.get('curriculum_alignment', 0.0) > 0.5,
            'learning_progressing': metrics.get('learning_progress', 0.0) > 0.0
        }

    def _calculate_strategy_effectiveness(self) -> Dict[str, float]:
        """Calculate effectiveness scores for different strategies"""
        effectiveness = {}

        for strategy_name, performances in self.strategy_performance.items():
            if performances:
                avg_reward = sum(p.get('weighted_reward', 0.0) for p in performances) / len(performances)
                avg_factuality = sum(p.get('factuality_score', 0.0) for p in performances) / len(performances)
                effectiveness[strategy_name] = (avg_reward + avg_factuality) / 2.0
            else:
                effectiveness[strategy_name] = 0.0

        return effectiveness

    def _should_adapt(self) -> bool:
        """Determine if parameter adaptation is needed"""
        # Check time-based adaptation
        time_since_adaptation = time.time() - self.last_adaptation
        if time_since_adaptation >= self.adaptation_interval:
            return True

        # Check performance-based adaptation triggers
        if len(self.performance_window) >= 5:
            recent_performance = [entry['metrics'].get('weighted_reward', 0.0)
                                for entry in list(self.performance_window)[-5:]]
            avg_recent = sum(recent_performance) / len(recent_performance)

            # Adapt if performance is consistently low
            if avg_recent < 0.3:
                return True

            # Adapt if performance is declining
            if len(recent_performance) >= 3:
                trend = recent_performance[-1] - recent_performance[0]
                if trend < -0.2:  # Significant decline
                    return True

        return False

    def _perform_adaptation(self, current_metrics: Dict[str, Any]):
        """Perform parameter adaptation in background thread"""
        try:
            self.is_adapting = True
            logger.info("Starting meta-learning adaptation...")

            # Get current context
            context = self._get_current_context()

            # Select optimal strategy
            optimal_strategy = self.meta_learner.select_optimal_strategy(context)
            if optimal_strategy != self.meta_learner.current_strategy:
                logger.info(f"Switching strategy from {self.meta_learner.current_strategy.value} "
                          f"to {optimal_strategy.value}")
                self.meta_learner.current_strategy = optimal_strategy

            # Adapt parameters
            old_params = self.meta_learner.current_params.copy()
            new_params = self.meta_learner.adapt_parameters(current_metrics)

            # Apply parameter changes to learning components
            self._apply_parameter_changes(old_params, new_params)

            # Record adaptation
            self.parameter_adaptation_history.append({
                'timestamp': time.time(),
                'old_params': old_params,
                'new_params': new_params,
                'strategy': self.meta_learner.current_strategy.value,
                'trigger_metrics': current_metrics
            })

            self.last_adaptation = time.time()
            logger.info("Meta-learning adaptation completed")

        except Exception as e:
            logger.error(f"Error during adaptation: {str(e)}")
        finally:
            self.is_adapting = False

    def _get_current_context(self) -> Dict[str, Any]:
        """Get current learning context"""
        context = self.current_context.copy()

        # Add curriculum information
        try:
            learner_id = getattr(self.scheduler, 'default_learner_id', 'autonomous_agent')
            progress = self.scheduler.get_learner_progress(learner_id)
            if progress:
                context['curriculum_completion'] = len(progress.completed_skills) / \
                    max(len(self.scheduler.curriculum_tree.skills), 1)
                context['current_difficulty'] = progress.current_difficulty.value
        except:
            pass

        # Add performance trends
        if self.performance_window:
            recent_rewards = [entry['metrics'].get('weighted_reward', 0.0)
                            for entry in list(self.performance_window)[-10:]]
            context['recent_avg_reward'] = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0

        return context

    def _apply_parameter_changes(self, old_params: Dict[str, Any], new_params: Dict[str, Any]):
        """Apply parameter changes to learning components"""
        # This would be implemented to actually change parameters in the LLM and other components
        # For now, we'll log the changes
        changed_params = {}
        for key in new_params:
            if key in old_params and new_params[key] != old_params[key]:
                changed_params[key] = {'from': old_params[key], 'to': new_params[key]}

        if changed_params:
            logger.info(f"Applied parameter changes: {changed_params}")

    def get_meta_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning performance metrics"""
        return {
            'current_strategy': self.meta_learner.current_strategy.value,
            'current_params': self.meta_learner.current_params.copy(),
            'strategy_performance': dict(self.strategy_performance),
            'recent_performance': [
                {
                    'timestamp': entry['timestamp'],
                    'metrics': entry['metrics'],
                    'strategy': self.meta_learner.current_strategy.value
                }
                for entry in list(self.performance_window)[-10:]
            ],
            'adaptation_history': self.parameter_adaptation_history[-5:],  # Last 5 adaptations
            'context_awareness': self.current_context.copy(),
            'strategy_effectiveness': self._calculate_strategy_effectiveness(),
            'learning_transfer_memory': self.task_transfer_memory.copy()
        }

    def trigger_adaptation(self) -> Dict[str, Any]:
        """Manually trigger parameter adaptation"""
        if self.is_adapting:
            logger.warning("⚠️  Adaptation already in progress")
            return {'success': False, 'message': 'Adaptation already in progress'}

        try:
            logger.info("🔧 Manually triggering meta-learning adaptation...")

            # Get current metrics for adaptation
            if self.performance_window:
                current_metrics = self.performance_window[-1]['metrics']
            else:
                current_metrics = {}

            # Perform adaptation
            self._perform_adaptation(current_metrics)

            result = {
                'success': True,
                'message': 'Adaptation triggered successfully',
                'new_strategy': self.meta_learner.current_strategy.value,
                'parameter_changes': self.parameter_adaptation_history[-1] if self.parameter_adaptation_history else {}
            }

            logger.info(f"✅ Manual adaptation completed | New strategy: {result['new_strategy']}")
            return result

        except Exception as e:
            logger.error(f"❌ Error triggering adaptation: {str(e)}")
            return {'success': False, 'message': str(e)}

    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get information about available learning strategies"""
        strategies = []

        for strategy in LearningStrategy:
            performance = self.strategy_performance.get(strategy.value, [])
            avg_performance = 0.0
            if performance:
                avg_performance = sum(p.get('weighted_reward', 0.0) for p in performance) / len(performance)

            strategies.append({
                'name': strategy.value,
                'description': self._get_strategy_description(strategy),
                'current': strategy == self.meta_learner.current_strategy,
                'performance_history': len(performance),
                'average_performance': avg_performance,
                'last_used': self.meta_learner.metrics.strategy_performance.get(
                    strategy.value, {}).get('last_used', 0)
            })

        return strategies

    def _get_strategy_description(self, strategy: LearningStrategy) -> str:
        """Get description for a learning strategy"""
        descriptions = {
            LearningStrategy.EXPLORATION_FOCUSED: "Prioritizes exploration over exploitation, good for discovering new patterns",
            LearningStrategy.EXPLOITATION_FOCUSED: "Prioritizes exploitation of known good actions, good for optimization",
            LearningStrategy.BALANCED: "Balances exploration and exploitation for general learning",
            LearningStrategy.CURRICULUM_DRIVEN: "Follows curriculum structure for structured learning progression",
            LearningStrategy.REWARD_OPTIMIZED: "Optimizes for maximum reward across all objectives",
            LearningStrategy.ADAPTIVE: "Dynamically adapts based on performance and context"
        }
        return descriptions.get(strategy, "Unknown strategy")

    def switch_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Switch to a different learning strategy"""
        try:
            new_strategy = LearningStrategy(strategy_name)

            if new_strategy == self.meta_learner.current_strategy:
                logger.info(f"ℹ️  Already using strategy {strategy_name}")
                return {
                    'success': False,
                    'message': f'Already using strategy {strategy_name}'
                }

            old_strategy = self.meta_learner.current_strategy
            logger.info(f"🔄 Switching learning strategy from {old_strategy.value} to {new_strategy.value}")

            self.meta_learner.current_strategy = new_strategy

            # Record the strategy change
            self.meta_learner.metrics.record_parameter_change(
                {'strategy': old_strategy.value},
                {'strategy': new_strategy.value},
                f"Manual strategy switch to {new_strategy.value}"
            )

            # Log curriculum status update for strategy change
            log_curriculum_status("meta_learner", "strategy_changed", {
                'old_strategy': old_strategy.value,
                'new_strategy': new_strategy.value,
                'reason': 'manual_switch'
            })

            logger.info(f"✅ Successfully switched to strategy {strategy_name}")

            return {
                'success': True,
                'message': f'Successfully switched to {strategy_name}',
                'old_strategy': old_strategy.value,
                'new_strategy': new_strategy.value
            }

        except ValueError:
            logger.error(f"❌ Invalid strategy name: {strategy_name}")
            return {
                'success': False,
                'message': f'Invalid strategy name: {strategy_name}',
                'available_strategies': [s.value for s in LearningStrategy]
            }

    def update_transfer_learning(self, task_type: str, performance: float, learned_params: Dict[str, Any]):
        """Update transfer learning memory with task-specific learnings"""
        if task_type not in self.task_transfer_memory:
            self.task_transfer_memory[task_type] = {
                'performances': [],
                'parameter_sets': [],
                'transfer_weight': 1.0
            }

        memory = self.task_transfer_memory[task_type]
        memory['performances'].append(performance)
        memory['parameter_sets'].append(learned_params.copy())

        # Update transfer weight based on performance
        avg_performance = sum(memory['performances']) / len(memory['performances'])
        memory['transfer_weight'] = min(avg_performance * 2.0, 2.0)  # Cap at 2.0

        # Keep only recent data
        if len(memory['performances']) > 10:
            memory['performances'] = memory['performances'][-10:]
            memory['parameter_sets'] = memory['parameter_sets'][-10:]

    def get_transfer_learning_suggestions(self, current_task_type: str) -> List[Dict[str, Any]]:
        """Get transfer learning suggestions for current task"""
        suggestions = []

        for task_type, memory in self.task_transfer_memory.items():
            if task_type != current_task_type and memory['transfer_weight'] > 0.5:
                # Suggest parameters from similar successful tasks
                best_params = None
                best_performance = 0.0

                for i, perf in enumerate(memory['performances']):
                    if perf > best_performance:
                        best_performance = perf
                        best_params = memory['parameter_sets'][i]

                if best_params:
                    suggestions.append({
                        'source_task': task_type,
                        'suggested_params': best_params,
                        'expected_performance': best_performance,
                        'transfer_weight': memory['transfer_weight']
                    })

        # Sort by transfer weight
        suggestions.sort(key=lambda x: x['transfer_weight'], reverse=True)
        return suggestions[:3]  # Top 3 suggestions