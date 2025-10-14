import re
import math
from typing import Dict, List, Any, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class RewardService:
    """
    Multi-Objective Reward System for LLM Learning

    Provides multiple reward functions that evaluate different aspects of LLM responses:
    - Accuracy: Based on response correctness and factual accuracy
    - Coherence: Based on logical consistency and flow
    - Factuality: Based on factual accuracy with hallucination detection
    - Creativity: Based on response diversity and novelty
    """

    def __init__(self, hallucination_service=None, feedback_service=None):
        # Default reward weights
        self.reward_weights = {
            'accuracy': 0.4,
            'coherence': 0.3,
            'factuality': 0.2,
            'creativity': 0.1
        }

        # Track reward metrics
        self.reward_history = []
        self.current_metrics = {
            'accuracy': 0.0,
            'coherence': 0.0,
            'factuality': 0.0,
            'creativity': 0.0,
            'total_reward': 0.0,
            'weighted_reward': 0.0
        }

        # External services
        self.hallucination_service = hallucination_service
        self.feedback_service = feedback_service

    def configure_weights(self, weights: Dict[str, float]) -> bool:
        """
        Configure reward weights for different objectives.

        Args:
            weights: Dictionary with keys 'accuracy', 'coherence', 'factuality', 'creativity'
                    and float values that sum to 1.0

        Returns:
            bool: True if configuration successful, False otherwise
        """
        required_keys = {'accuracy', 'coherence', 'factuality', 'creativity'}

        if not all(key in weights for key in required_keys):
            logger.error("Missing required weight keys")
            return False

        weight_sum = sum(weights.values())
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
            logger.error(f"Weights must sum to 1.0, got {weight_sum}")
            return False

        if not all(0.0 <= w <= 1.0 for w in weights.values()):
            logger.error("All weights must be between 0.0 and 1.0")
            return False

        self.reward_weights = weights.copy()
        logger.info(f"Reward weights configured: {self.reward_weights}")
        return True

    def calculate_multi_objective_reward(self, prompt: str, response: str, action: int,
                                       user_id: Optional[str] = None, response_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate multi-objective reward for a prompt-response pair with user feedback integration.

        Args:
            prompt: The input prompt text
            response: The generated response
            action: The action taken by the LLM
            user_id: Optional user ID for personalized rewards
            response_id: Optional response ID for feedback-based adjustments

        Returns:
            Dict containing individual rewards, total reward, and weighted reward
        """
        try:
            # Calculate individual reward components
            accuracy_reward = self.accuracy_reward(prompt, response)
            coherence_reward = self.coherence_reward(prompt, response)
            factuality_reward = self.factuality_reward(prompt, response)
            creativity_reward = self.creativity_reward(prompt, response, action)

            # Get user-specific adjustments
            user_adjustments = {}
            personalized_weights = self.reward_weights.copy()

            if self.feedback_service and user_id and response_id:
                user_adjustments = self.feedback_service.get_feedback_for_reward_adjustment(user_id, response_id)

                # Apply user preference weights
                if 'weight_adjustments' in user_adjustments:
                    for category, weight in user_adjustments['weight_adjustments'].items():
                        if category in personalized_weights:
                            # Blend user preferences with default weights (70% user, 30% default)
                            personalized_weights[category] = 0.7 * weight + 0.3 * self.reward_weights[category]

            # Calculate base rewards
            individual_rewards = {
                'accuracy': accuracy_reward,
                'coherence': coherence_reward,
                'factuality': factuality_reward,
                'creativity': creativity_reward
            }

            # Apply feedback-based adjustments
            adjusted_rewards = individual_rewards.copy()

            # Apply penalties and boosts from user feedback
            if 'penalty' in user_adjustments:
                # Distribute penalty across all objectives
                penalty_per_objective = user_adjustments['penalty'] / len(adjusted_rewards)
                for obj in adjusted_rewards:
                    adjusted_rewards[obj] = max(0.0, adjusted_rewards[obj] - penalty_per_objective)

            if 'boost' in user_adjustments:
                # Distribute boost across all objectives
                boost_per_objective = user_adjustments['boost'] / len(adjusted_rewards)
                for obj in adjusted_rewards:
                    adjusted_rewards[obj] = min(1.0, adjusted_rewards[obj] + boost_per_objective)

            if 'rating_penalty' in user_adjustments:
                # Apply rating-based penalty to accuracy (most affected by user ratings)
                adjusted_rewards['accuracy'] = max(0.0, adjusted_rewards['accuracy'] - user_adjustments['rating_penalty'])

            # Apply correction-based learning adjustments
            correction_adjustments = self.get_correction_based_adjustments(response)
            for obj, adjustment in correction_adjustments.items():
                if obj in adjusted_rewards:
                    adjusted_rewards[obj] = max(0.0, min(1.0, adjusted_rewards[obj] + adjustment))

            # Calculate total and weighted rewards using personalized weights
            total_reward = sum(adjusted_rewards.values()) / len(adjusted_rewards)
            weighted_reward = sum(
                adjusted_rewards[obj] * personalized_weights[obj]
                for obj in adjusted_rewards
            )

            # Update current metrics
            self.current_metrics.update(adjusted_rewards)
            self.current_metrics['total_reward'] = total_reward
            self.current_metrics['weighted_reward'] = weighted_reward

            # Store in history
            reward_entry = {
                'prompt': prompt[:100],  # Truncate for storage
                'response': response[:100],
                'action': action,
                'user_id': user_id,
                'response_id': response_id,
                'rewards': individual_rewards.copy(),
                'adjusted_rewards': adjusted_rewards.copy(),
                'total_reward': total_reward,
                'weighted_reward': weighted_reward,
                'weights': personalized_weights.copy(),
                'user_adjustments': user_adjustments
            }
            self.reward_history.append(reward_entry)

            # Keep only last 1000 entries
            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-1000:]

            result = {
                'individual_rewards': adjusted_rewards,
                'base_rewards': individual_rewards,
                'total_reward': total_reward,
                'weighted_reward': weighted_reward,
                'weights_used': personalized_weights,
                'user_adjustments': user_adjustments,
                'personalized': user_id is not None and bool(user_adjustments)
            }

            logger.debug(f"Calculated multi-objective reward: personalized={result['personalized']}, "
                        f"weighted_reward={weighted_reward:.3f}")
            return result

        except Exception as e:
            logger.error(f"Error calculating multi-objective reward: {str(e)}")
            # Return default rewards on error
            return {
                'individual_rewards': {
                    'accuracy': 0.5,
                    'coherence': 0.5,
                    'factuality': 0.5,
                    'creativity': 0.5
                },
                'base_rewards': {
                    'accuracy': 0.5,
                    'coherence': 0.5,
                    'factuality': 0.5,
                    'creativity': 0.5
                },
                'total_reward': 0.5,
                'weighted_reward': 0.5,
                'weights_used': self.reward_weights.copy(),
                'user_adjustments': {},
                'personalized': False
            }

    def accuracy_reward(self, prompt: str, response: str) -> float:
        """
        Calculate accuracy reward based on response correctness.

        This is a heuristic-based implementation that evaluates:
        - Response length appropriateness
        - Presence of relevant keywords
        - Structure and completeness
        """
        try:
            prompt_words = set(prompt.lower().split())
            response_words = set(response.lower().split())

            # Base score from response length (prefer substantial responses)
            length_score = min(len(response.split()) / 50.0, 1.0)  # Max at 50 words

            # Keyword overlap score
            if prompt_words:
                overlap = len(prompt_words.intersection(response_words)) / len(prompt_words)
                keyword_score = min(overlap * 2.0, 1.0)  # Scale up overlap
            else:
                keyword_score = 0.5

            # Structure score (presence of complete sentences, proper grammar hints)
            sentences = re.split(r'[.!?]+', response)
            complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
            structure_score = min(complete_sentences / 3.0, 1.0)  # Max at 3 complete sentences

            # Combine scores with weights
            accuracy = (length_score * 0.3 + keyword_score * 0.4 + structure_score * 0.3)

            return max(0.0, min(1.0, accuracy))  # Clamp to [0,1]

        except Exception as e:
            logger.warning(f"Error in accuracy_reward: {str(e)}")
            return 0.5

    def coherence_reward(self, prompt: str, response: str) -> float:
        """
        Calculate coherence reward based on logical consistency and flow.

        Evaluates:
        - Logical flow between sentences
        - Consistency of terminology
        - Absence of contradictions
        """
        try:
            sentences = re.split(r'[.!?]+', response.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) < 2:
                return 0.3  # Minimum coherence for very short responses

            # Check for logical connectors
            connectors = ['and', 'or', 'but', 'however', 'therefore', 'thus', 'because', 'since']
            connector_count = sum(1 for sentence in sentences
                                for connector in connectors
                                if connector in sentence.lower())

            # Transition score based on connectors
            transition_score = min(connector_count / len(sentences), 1.0)

            # Consistency score (repeated keywords across sentences)
            all_words = [word.lower() for sentence in sentences for word in sentence.split()]
            word_counts = Counter(all_words)

            # Words that appear in multiple sentences get consistency points
            consistent_words = sum(1 for count in word_counts.values() if count > 1)
            consistency_score = min(consistent_words / len(set(all_words)), 1.0)

            # Flow score (sentence length variation - avoid monotonous lengths)
            if len(sentences) > 1:
                lengths = [len(s.split()) for s in sentences]
                avg_length = sum(lengths) / len(lengths)
                variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
                flow_score = min(variance / 25.0, 1.0)  # Prefer some variation
            else:
                flow_score = 0.5

            coherence = (transition_score * 0.4 + consistency_score * 0.4 + flow_score * 0.2)

            return max(0.0, min(1.0, coherence))

        except Exception as e:
            logger.warning(f"Error in coherence_reward: {str(e)}")
            return 0.5

    def factuality_reward(self, prompt: str, response: str) -> float:
        """
        Calculate factuality reward based on factual accuracy with hallucination detection.

        Uses hallucination detection service to assess response quality and applies penalties
        for detected hallucinations. Falls back to basic heuristics if service unavailable.

        Evaluates:
        - Hallucination detection results (primary)
        - Presence of specific facts/numbers (fallback)
        - Use of definitive language appropriately (fallback)
        - Absence of obvious contradictions (fallback)
        """
        try:
            # Use hallucination detection if available
            if self.hallucination_service:
                try:
                    analysis = self.hallucination_service.analyze_response(prompt, response)

                    # Base factuality from hallucination analysis
                    hallucination_penalty = self.hallucination_service.get_hallucination_penalty(analysis)

                    # Factuality is inverse of hallucination confidence and penalty
                    base_factuality = 1.0 - analysis.overall_confidence

                    # Apply additional penalty for detected hallucinations
                    factuality = base_factuality * (1.0 - hallucination_penalty)

                    # Boost factuality for high uncertainty (appropriate hedging)
                    uncertainty_boost = analysis.uncertainty_score * 0.2
                    factuality += uncertainty_boost

                    logger.debug(f"Hallucination-aware factuality: base={base_factuality:.3f}, "
                               f"penalty={hallucination_penalty:.3f}, uncertainty_boost={uncertainty_boost:.3f}, "
                               f"final={factuality:.3f}")

                    return max(0.0, min(1.0, factuality))

                except Exception as e:
                    logger.warning(f"Hallucination analysis failed, falling back to basic factuality: {str(e)}")

            # Fallback to basic implementation if hallucination service unavailable or failed
            # Look for numbers (potential facts)
            numbers = re.findall(r'\d+\.?\d*', response)
            number_score = min(len(numbers) / 5.0, 1.0)  # Max at 5 numbers

            # Look for definitive language (indicators of factual statements)
            definitive_words = ['is', 'are', 'was', 'were', 'has', 'have', 'contains', 'includes']
            definitive_count = sum(1 for word in definitive_words if word in response.lower())
            definitive_score = min(definitive_count / 10.0, 1.0)  # Max at 10 definitive statements

            # Check for hedging language (reduces factuality if overused)
            hedging_words = ['might', 'may', 'could', 'possibly', 'perhaps', 'maybe']
            hedging_count = sum(1 for word in hedging_words if word in response.lower())
            hedging_penalty = min(hedging_count / 5.0, 0.5)  # Max penalty of 0.5

            # Specificity score (longer, more detailed responses tend to be more factual)
            word_count = len(response.split())
            specificity_score = min(word_count / 100.0, 1.0)  # Max at 100 words

            factuality = (number_score * 0.2 + definitive_score * 0.3 +
                          specificity_score * 0.3 - hedging_penalty * 0.2)

            return max(0.0, min(1.0, factuality))

        except Exception as e:
            logger.warning(f"Error in factuality_reward: {str(e)}")
            return 0.5

    def creativity_reward(self, prompt: str, response: str, action: int) -> float:
        """
        Calculate creativity reward based on response diversity and novelty.

        Evaluates:
        - Use of varied vocabulary
        - Novel combinations of ideas
        - Unconventional approaches
        """
        try:
            words = response.lower().split()
            unique_words = set(words)

            if not words:
                return 0.0

            # Vocabulary diversity score
            diversity_score = len(unique_words) / len(words)

            # Novel word score (words not in prompt)
            prompt_words = set(prompt.lower().split())
            novel_words = unique_words - prompt_words
            novelty_score = len(novel_words) / len(unique_words) if unique_words else 0.0

            # Action diversity bonus (encourage different actions)
            # This is a simple heuristic - in practice, you'd track action diversity over time
            action_diversity = (action % 10) / 9.0  # Normalize action to [0,1]

            # Length creativity (avoiding very short or very long responses)
            word_count = len(words)
            if 10 <= word_count <= 75:
                length_creativity = 1.0
            elif word_count < 10:
                length_creativity = word_count / 10.0
            else:
                length_creativity = max(0.5, 1.0 - (word_count - 75) / 50.0)

            creativity = (diversity_score * 0.3 + novelty_score * 0.3 +
                         action_diversity * 0.2 + length_creativity * 0.2)

            return max(0.0, min(1.0, creativity))

        except Exception as e:
            logger.warning(f"Error in creativity_reward: {str(e)}")
            return 0.5

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current reward metrics.

        Returns:
            Dict containing current reward metrics and configuration
        """
        return {
            'current_metrics': self.current_metrics.copy(),
            'reward_weights': self.reward_weights.copy(),
            'history_length': len(self.reward_history)
        }

    def get_reward_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent reward history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of recent reward history entries
        """
        return self.reward_history[-limit:] if self.reward_history else []

    def get_average_metrics(self, window: int = 50) -> Dict[str, float]:
        """
        Calculate average reward metrics over recent history.

        Args:
            window: Number of recent entries to average over

        Returns:
            Dict with average metrics
        """
        if not self.reward_history:
            return {k: 0.0 for k in ['accuracy', 'coherence', 'factuality', 'creativity', 'total_reward', 'weighted_reward']}

        recent_entries = self.reward_history[-window:]
        averages = {}

        for key in ['accuracy', 'coherence', 'factuality', 'creativity', 'total_reward', 'weighted_reward']:
            values = [entry.get(key, entry.get('rewards', {}).get(key, 0.0)) for entry in recent_entries]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages

    def process_user_correction(self, original_response: str, corrected_response: str,
                               correction_type: str, user_id: str) -> Dict[str, Any]:
        """
        Process user correction to update reward learning.

        Args:
            original_response: The original AI response
            corrected_response: The user-corrected response
            correction_type: Type of correction (factual_error, logical_error, etc.)
            user_id: User who provided the correction

        Returns:
            Dict with correction analysis and reward adjustments
        """
        try:
            correction_analysis = {
                'correction_type': correction_type,
                'user_id': user_id,
                'original_length': len(original_response.split()),
                'corrected_length': len(corrected_response.split()),
                'length_difference': len(corrected_response.split()) - len(original_response.split())
            }

            # Analyze what aspects of the response needed correction
            adjustments = {}

            if correction_type == 'factual_error':
                # Major penalty to factuality, some to accuracy
                adjustments['factuality_penalty'] = 0.5
                adjustments['accuracy_penalty'] = 0.3
            elif correction_type == 'logical_error':
                # Penalty to coherence and accuracy
                adjustments['coherence_penalty'] = 0.4
                adjustments['accuracy_penalty'] = 0.3
            elif correction_type == 'incomplete_info':
                # Penalty to completeness (affects accuracy and coherence)
                adjustments['accuracy_penalty'] = 0.4
                adjustments['coherence_penalty'] = 0.2
            elif correction_type == 'style_improvement':
                # Minor adjustments to creativity and coherence
                adjustments['creativity_boost'] = 0.1
                adjustments['coherence_boost'] = 0.1
            elif correction_type == 'format_issue':
                # Minor penalty to coherence
                adjustments['coherence_penalty'] = 0.2

            # Store correction learning for future reward adjustments
            correction_entry = {
                'original_response': original_response[:200],
                'corrected_response': corrected_response[:200],
                'correction_type': correction_type,
                'user_id': user_id,
                'adjustments': adjustments,
                'timestamp': time.time()
            }

            # Add to reward history for learning
            if not hasattr(self, 'correction_history'):
                self.correction_history = []

            self.correction_history.append(correction_entry)

            # Keep only last 500 corrections
            if len(self.correction_history) > 500:
                self.correction_history = self.correction_history[-500:]

            logger.info(f"Processed user correction: type={correction_type}, user={user_id}")

            return {
                'success': True,
                'adjustments': adjustments,
                'analysis': correction_analysis,
                'correction_id': len(self.correction_history) - 1
            }

        except Exception as e:
            logger.error(f"Error processing user correction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'adjustments': {},
                'analysis': {}
            }

    def get_correction_based_adjustments(self, response_text: str) -> Dict[str, float]:
        """
        Get reward adjustments based on similar past corrections.

        Args:
            response_text: Current response to analyze

        Returns:
            Dict with recommended adjustments based on correction history
        """
        try:
            if not hasattr(self, 'correction_history') or not self.correction_history:
                return {}

            # Simple similarity matching (in production, use better NLP)
            adjustments = {'accuracy': 0.0, 'coherence': 0.0, 'factuality': 0.0, 'creativity': 0.0}

            for correction in self.correction_history[-50:]:  # Check last 50 corrections
                # Check if response has similar patterns to corrected responses
                original_words = set(correction['original_response'].lower().split())
                current_words = set(response_text.lower().split())

                similarity = len(original_words.intersection(current_words)) / len(original_words.union(current_words))

                if similarity > 0.3:  # 30% word overlap threshold
                    # Apply learned adjustments
                    corr_adjustments = correction.get('adjustments', {})
                    for key, value in corr_adjustments.items():
                        if key.endswith('_penalty'):
                            obj = key.replace('_penalty', '')
                            if obj in adjustments:
                                adjustments[obj] -= value * 0.1  # Small adjustment
                        elif key.endswith('_boost'):
                            obj = key.replace('_boost', '')
                            if obj in adjustments:
                                adjustments[obj] += value * 0.1

            # Normalize adjustments
            for obj in adjustments:
                adjustments[obj] = max(-0.3, min(0.3, adjustments[obj]))  # Limit adjustment range

            return adjustments

        except Exception as e:
            logger.error(f"Error getting correction-based adjustments: {str(e)}")
            return {}