"""
Hallucination Detection & Prevention Service

This service provides comprehensive hallucination detection capabilities including:
- Pattern-based detection of overconfident language and contradictions
- Confidence scoring and uncertainty metrics
- Basic fact-checking using knowledge patterns
- Integration with reward systems for hallucination penalties
"""

import re
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import time

from models.hallucination import (
    HallucinationAnalysis,
    HallucinationConfig,
    HallucinationMetrics,
    HallucinationIndicator,
    HallucinationType,
    HallucinationSeverity,
    HallucinationCheckRequest,
    HallucinationCheckResponse
)

logger = logging.getLogger(__name__)


class HallucinationService:
    """
    Service for detecting and analyzing hallucinations in LLM responses.

    Provides pattern-based detection, confidence scoring, fact-checking,
    and uncertainty metrics to identify potentially hallucinated content.
    """

    def __init__(self, config: Optional[HallucinationConfig] = None):
        self.config = config or HallucinationConfig()
        self.metrics = HallucinationMetrics()
        self.analysis_history: List[HallucinationAnalysis] = []
        logger.info("HallucinationService initialized with config")

    def update_config(self, config: HallucinationConfig) -> bool:
        """
        Update the service configuration.

        Args:
            config: New configuration to apply

        Returns:
            bool: True if update successful
        """
        try:
            self.config = config
            logger.info("Hallucination detection config updated")
            return True
        except Exception as e:
            logger.error(f"Failed to update config: {str(e)}")
            return False

    def analyze_response(self, prompt_text: str, response_text: str,
                        context: Optional[Dict[str, Any]] = None) -> HallucinationAnalysis:
        """
        Analyze a response for potential hallucinations.

        Args:
            prompt_text: The original prompt
            response_text: The response to analyze
            context: Additional context information

        Returns:
            HallucinationAnalysis: Complete analysis results
        """
        start_time = time.time()

        try:
            # Perform individual detection methods
            pattern_indicators = self._detect_pattern_based_hallucinations(response_text)
            contradiction_indicators = self._detect_contradictions(prompt_text, response_text)
            factuality_indicators = self._perform_fact_checking(response_text)

            # Combine all indicators
            all_indicators = pattern_indicators + contradiction_indicators + factuality_indicators

            # Calculate overall scores
            uncertainty_score = self._calculate_uncertainty_score(response_text)
            factuality_score = self._calculate_factuality_score(response_text, factuality_indicators)
            pattern_score = self._calculate_pattern_score(pattern_indicators)

            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                uncertainty_score, factuality_score, pattern_score, contradiction_indicators
            )

            # Determine if hallucinated and risk level
            is_hallucinated = overall_confidence >= self.config.confidence_threshold
            risk_level = self._determine_risk_level(overall_confidence, all_indicators)

            analysis = HallucinationAnalysis(
                overall_confidence=overall_confidence,
                is_hallucinated=is_hallucinated,
                indicators=all_indicators,
                uncertainty_score=uncertainty_score,
                factuality_score=factuality_score,
                risk_level=risk_level,
                analysis_metadata={
                    "processing_time": time.time() - start_time,
                    "pattern_score": pattern_score,
                    "num_indicators": len(all_indicators),
                    "context_provided": context is not None
                }
            )

            # Update metrics
            self._update_metrics(analysis)

            # Store in history
            self.analysis_history.append(analysis)
            if len(self.analysis_history) > 1000:  # Keep last 1000 analyses
                self.analysis_history = self.analysis_history[-1000:]

            logger.debug(f"Hallucination analysis completed: confidence={overall_confidence:.3f}, "
                        f"hallucinated={is_hallucinated}, indicators={len(all_indicators)}")

            return analysis

        except Exception as e:
            logger.error(f"Error in hallucination analysis: {str(e)}")
            # Return safe default analysis
            return HallucinationAnalysis(
                overall_confidence=0.0,
                is_hallucinated=False,
                indicators=[],
                uncertainty_score=0.5,
                factuality_score=0.5,
                risk_level=HallucinationSeverity.LOW,
                analysis_metadata={"error": str(e)}
            )

    def check_hallucination(self, request: HallucinationCheckRequest) -> HallucinationCheckResponse:
        """
        API method to check a response for hallucinations.

        Args:
            request: Check request with prompt, response, and optional config

        Returns:
            HallucinationCheckResponse: Complete response with analysis
        """
        start_time = time.time()

        # Use provided config override if available
        config_to_use = request.config_override or self.config

        # Temporarily update config if override provided
        original_config = None
        if request.config_override:
            original_config = self.config
            self.config = config_to_use

        try:
            analysis = self.analyze_response(
                request.prompt_text,
                request.response_text,
                request.context
            )

            response = HallucinationCheckResponse(
                analysis=analysis,
                processing_time=time.time() - start_time,
                config_used=config_to_use
            )

            return response

        finally:
            # Restore original config if it was overridden
            if original_config:
                self.config = original_config

    def get_hallucination_penalty(self, analysis: HallucinationAnalysis) -> float:
        """
        Calculate hallucination penalty for reward systems.

        Args:
            analysis: Hallucination analysis results

        Returns:
            float: Penalty value between 0.0 (no penalty) and 1.0 (maximum penalty)
        """
        if not analysis.is_hallucinated:
            return 0.0

        # Base penalty on confidence and severity
        base_penalty = analysis.overall_confidence

        # Severity multiplier
        severity_multipliers = {
            HallucinationSeverity.LOW: 0.2,
            HallucinationSeverity.MEDIUM: 0.5,
            HallucinationSeverity.HIGH: 0.8,
            HallucinationSeverity.CRITICAL: 1.0
        }

        severity_penalty = severity_multipliers.get(analysis.risk_level, 0.5)

        # Number of indicators penalty
        indicator_penalty = min(len(analysis.indicators) / 10.0, 0.5)

        total_penalty = base_penalty * (0.5 + severity_penalty * 0.3 + indicator_penalty * 0.2)
        return min(total_penalty, 1.0)

    def get_metrics(self) -> HallucinationMetrics:
        """Get current hallucination detection metrics."""
        return self.metrics

    def get_recent_analyses(self, limit: int = 50) -> List[HallucinationAnalysis]:
        """Get recent hallucination analyses."""
        return self.analysis_history[-limit:] if self.analysis_history else []

    def _detect_pattern_based_hallucinations(self, response_text: str) -> List[HallucinationIndicator]:
        """Detect hallucinations based on linguistic patterns."""
        indicators = []
        response_lower = response_text.lower()

        # Check for overconfident language
        overconfident_count = 0
        for phrase in self.config.overconfident_phrases:
            count = response_lower.count(phrase.lower())
            overconfident_count += count
            if count > 0:
                confidence = min(count / 3.0, 1.0)  # Scale confidence
                indicators.append(HallucinationIndicator(
                    type=HallucinationType.OVERCONFIDENT,
                    severity=HallucinationSeverity.MEDIUM if confidence > 0.5 else HallucinationSeverity.LOW,
                    confidence=confidence,
                    text_snippet=f"...{phrase}...",
                    explanation=f"Overconfident language detected: '{phrase}'"
                ))

        # Check for lack of hedging when making claims
        definitive_words = ["is", "are", "has", "have", "contains", "must", "always", "never"]
        hedging_count = sum(1 for phrase in self.config.hedging_phrases if phrase.lower() in response_lower)
        definitive_count = sum(1 for word in definitive_words if word in response_lower.split())

        if definitive_count > 5 and hedging_count == 0:
            confidence = min((definitive_count - 5) / 10.0, 1.0)
            indicators.append(HallucinationIndicator(
                type=HallucinationType.OVERCONFIDENT,
                severity=HallucinationSeverity.HIGH,
                confidence=confidence,
                text_snippet="Multiple definitive statements without hedging",
                explanation="Response makes definitive claims without appropriate uncertainty markers"
            ))

        return indicators

    def _detect_contradictions(self, prompt_text: str, response_text: str) -> List[HallucinationIndicator]:
        """Detect contradictions between prompt and response."""
        indicators = []

        # Simple contradiction detection based on keywords
        prompt_words = set(prompt_text.lower().split())
        response_words = set(response_text.lower().split())

        # Look for obvious contradictions (this is a basic implementation)
        contradiction_pairs = [
            (["yes", "true"], ["no", "false"]),
            (["good", "positive"], ["bad", "negative"]),
            (["increase", "higher"], ["decrease", "lower"]),
        ]

        for positive_words, negative_words in contradiction_pairs:
            has_positive = any(word in prompt_words for word in positive_words)
            has_negative = any(word in response_words for word in negative_words)

            if has_positive and has_negative:
                indicators.append(HallucinationIndicator(
                    type=HallucinationType.CONTRADICTION,
                    severity=HallucinationSeverity.HIGH,
                    confidence=0.8,
                    text_snippet="Contradictory terms detected",
                    explanation=f"Response contradicts prompt intent with opposing terms"
                ))
                break  # Only add one contradiction indicator

        return indicators

    def _perform_fact_checking(self, response_text: str) -> List[HallucinationIndicator]:
        """Perform basic fact-checking using knowledge patterns."""
        indicators = []

        if not self.config.enable_fact_checking:
            return indicators

        response_lower = response_text.lower()

        # Check against known knowledge patterns
        for category, known_facts in self.config.knowledge_patterns.items():
            mentioned_facts = []
            for fact in known_facts:
                if fact.lower() in response_lower:
                    mentioned_facts.append(fact)

            if mentioned_facts:
                # For this basic implementation, we assume mentioned facts are correct
                # In a real system, this would query external knowledge sources
                factuality_confidence = 0.9  # High confidence for known facts
                # We don't add indicators here since facts are verified
                # Could add indicators for unknown claims in the future

        # Check for numbers that might be factual claims
        numbers = re.findall(r'\d+\.?\d*', response_text)
        if len(numbers) > 3:  # Many numbers might indicate factual content
            # In a real system, we'd verify these numbers
            indicators.append(HallucinationIndicator(
                type=HallucinationType.UNSUPPORTED_CLAIM,
                severity=HallucinationSeverity.MEDIUM,
                confidence=0.6,
                text_snippet=f"Multiple numerical values: {', '.join(numbers[:3])}...",
                explanation="Response contains many numerical values that may need fact-checking"
            ))

        return indicators

    def _calculate_uncertainty_score(self, response_text: str) -> float:
        """Calculate uncertainty score based on hedging language."""
        response_lower = response_text.lower()
        total_words = len(response_text.split())

        if total_words == 0:
            return 0.5

        hedging_count = sum(1 for phrase in self.config.hedging_phrases
                          if phrase.lower() in response_lower)

        # Uncertainty increases with more hedging, but very high hedging might indicate caution
        uncertainty = min(hedging_count / (total_words / 50.0), 1.0)

        # Adjust for response length - longer responses might be more certain
        length_factor = min(total_words / 100.0, 1.0)
        uncertainty = uncertainty * (1 - length_factor * 0.3)

        return uncertainty

    def _calculate_factuality_score(self, response_text: str, factuality_indicators: List[HallucinationIndicator]) -> float:
        """Calculate factuality score based on indicators and content analysis."""
        # Start with high factuality assumption
        factuality = 0.8

        # Reduce factuality based on indicators
        for indicator in factuality_indicators:
            factuality -= indicator.confidence * 0.2

        # Check for definitive language without hedging
        definitive_words = ["is", "are", "has", "have", "must", "always", "never"]
        definitive_count = sum(1 for word in definitive_words if word in response_text.lower().split())

        if definitive_count > 3:
            hedging_count = sum(1 for phrase in self.config.hedging_phrases
                              if phrase.lower() in response_text.lower())
            if hedging_count < definitive_count / 3:
                factuality -= 0.2  # Reduce factuality for overconfident claims

        return max(0.0, min(1.0, factuality))

    def _calculate_pattern_score(self, pattern_indicators: List[HallucinationIndicator]) -> float:
        """Calculate pattern-based hallucination score."""
        if not pattern_indicators:
            return 0.0

        # Average confidence of pattern indicators
        total_confidence = sum(indicator.confidence for indicator in pattern_indicators)
        pattern_score = total_confidence / len(pattern_indicators)

        # Weight by number of indicators
        pattern_score *= min(len(pattern_indicators) / 3.0, 1.0)

        return pattern_score

    def _calculate_overall_confidence(self, uncertainty_score: float, factuality_score: float,
                                    pattern_score: float, contradiction_indicators: List[HallucinationIndicator]) -> float:
        """Calculate overall hallucination confidence."""
        # Base score from weighted components
        overall = (
            (1 - uncertainty_score) * 0.2 +  # Low uncertainty increases hallucination risk
            (1 - factuality_score) * self.config.factuality_weight +
            pattern_score * self.config.pattern_weight
        )

        # Add contradiction penalty
        if contradiction_indicators:
            contradiction_penalty = sum(ind.confidence for ind in contradiction_indicators) / len(contradiction_indicators)
            overall += contradiction_penalty * self.config.contradiction_weight

        return min(overall, 1.0)

    def _determine_risk_level(self, confidence: float, indicators: List[HallucinationIndicator]) -> HallucinationSeverity:
        """Determine risk level based on confidence and indicators."""
        if confidence >= 0.9 or any(ind.severity == HallucinationSeverity.CRITICAL for ind in indicators):
            return HallucinationSeverity.CRITICAL
        elif confidence >= 0.7 or any(ind.severity == HallucinationSeverity.HIGH for ind in indicators):
            return HallucinationSeverity.HIGH
        elif confidence >= 0.5 or any(ind.severity == HallucinationSeverity.MEDIUM for ind in indicators):
            return HallucinationSeverity.MEDIUM
        else:
            return HallucinationSeverity.LOW

    def _update_metrics(self, analysis: HallucinationAnalysis):
        """Update service metrics with new analysis."""
        self.metrics.total_checks += 1

        if analysis.is_hallucinated:
            self.metrics.hallucinated_responses += 1

        # Update averages
        self.metrics.average_confidence = (
            (self.metrics.average_confidence * (self.metrics.total_checks - 1)) + analysis.overall_confidence
        ) / self.metrics.total_checks

        self.metrics.average_uncertainty = (
            (self.metrics.average_uncertainty * (self.metrics.total_checks - 1)) + analysis.uncertainty_score
        ) / self.metrics.total_checks

        # Update detection rate
        self.metrics.detection_rate = self.metrics.hallucinated_responses / self.metrics.total_checks

        # Update distributions
        severity_key = analysis.risk_level.value
        self.metrics.severity_distribution[severity_key] = self.metrics.severity_distribution.get(severity_key, 0) + 1

        for indicator in analysis.indicators:
            type_key = indicator.type.value
            self.metrics.common_hallucination_types[type_key] = self.metrics.common_hallucination_types.get(type_key, 0) + 1