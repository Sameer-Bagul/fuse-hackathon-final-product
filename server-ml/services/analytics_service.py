"""
Advanced Analytics Service for LLM Learning System

Provides comprehensive analytics including:
- Real-time learning progress dashboards
- Skill gap analysis and recommendations
- Learning bottleneck detection algorithms
- Performance prediction models using historical data
- System health monitoring and anomaly detection
"""

import time
import logging

from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
import psutil

from models.analytics import (
    SkillGap, SkillGapAnalysis, SkillGapSeverity,
    LearningBottleneck, BottleneckDetection, BottleneckType,
    PerformancePrediction, PredictionConfidence,
    LearningInsight, LearningInsights, InsightPriority,
    SystemHealthMetrics, TrendAnalysis, AnomalyDetection,
    AnalyticsDashboard, AnalyticsConfig
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    Advanced Analytics Service for comprehensive learning system monitoring and insights.

    Integrates with all existing services to provide:
    - Real-time analytics and dashboards
    - Skill gap analysis and bottleneck detection
    - Performance prediction and learning insights
    - System health monitoring and alerting
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()

        # Service dependencies (to be injected)
        self.reward_service = None
        self.hallucination_service = None
        self.meta_learning_service = None
        self.curriculum_service = None
        self.learning_loop_service = None
        self.external_llm_service = None
        self.prompt_controller = None
        self.history = None
        self.feedback_service = None

        # Analytics data storage
        self.analytics_history = []
        self.system_health_history = []
        self.anomaly_history = []
        self.last_update = 0

        # Statistical baselines for anomaly detection
        self.metric_baselines = {}
        self.trend_windows = defaultdict(list)

        logger.info("AnalyticsService initialized")

    def set_dependencies(self, **services):
        """Set service dependencies for analytics integration."""
        for service_name, service in services.items():
            if hasattr(self, service_name):
                setattr(self, service_name, service)
                logger.info(f"Set {service_name} dependency")
            else:
                logger.warning(f"Unknown service dependency: {service_name}")

    def get_analytics_dashboard(self, learner_id: Optional[str] = None,
                               include_historical: bool = False) -> AnalyticsDashboard:
        """
        Generate comprehensive analytics dashboard data.

        Args:
            learner_id: Specific learner to analyze (None for system-wide)
            include_historical: Whether to include historical trend data

        Returns:
            Complete analytics dashboard with all metrics and insights
        """
        start_time = time.time()

        try:
            timestamp = time.time()

            # Gather all analytics components
            skill_gap_analysis = self.analyze_skill_gaps(learner_id)
            bottleneck_detection = self.detect_bottlenecks(learner_id)
            performance_prediction = self.predict_performance(learner_id) if learner_id else None
            learning_insights = self.generate_learning_insights(learner_id)

            # System monitoring
            system_health = self.monitor_system_health()
            trend_analysis = self.analyze_trends() if include_historical else []
            anomalies = self.detect_anomalies()

            # Calculate overall learning score
            overall_score = self._calculate_overall_learning_score(
                skill_gap_analysis, bottleneck_detection, system_health
            )

            # Generate key indicators
            key_indicators = self._generate_key_indicators(
                skill_gap_analysis, bottleneck_detection, system_health
            )

            # Generate alerts
            alerts = self._generate_alerts(anomalies, system_health)

            # Data freshness tracking
            data_freshness = {
                'skill_gaps': skill_gap_analysis.timestamp if skill_gap_analysis else timestamp,
                'bottlenecks': bottleneck_detection.timestamp if bottleneck_detection else timestamp,
                'predictions': performance_prediction.timestamp if performance_prediction else timestamp,
                'insights': learning_insights.timestamp if learning_insights else timestamp,
                'system_health': system_health.timestamp if system_health else timestamp
            }

            dashboard = AnalyticsDashboard(
                timestamp=timestamp,
                learner_id=learner_id,
                skill_gap_analysis=skill_gap_analysis,
                bottleneck_detection=bottleneck_detection,
                performance_prediction=performance_prediction,
                learning_insights=learning_insights,
                system_health=system_health,
                trend_analysis=trend_analysis,
                anomalies=anomalies,
                overall_learning_score=overall_score,
                key_indicators=key_indicators,
                alerts=alerts,
                data_freshness=data_freshness
            )

            processing_time = time.time() - start_time
            logger.info(f"Generated analytics dashboard in {processing_time:.2f}s")

            return dashboard

        except Exception as e:
            logger.error(f"Failed to generate analytics dashboard: {str(e)}")
            # Return minimal dashboard on error
            return AnalyticsDashboard(
                timestamp=time.time(),
                learner_id=learner_id,
                overall_learning_score=0.5,
                key_indicators={'error': 'Dashboard generation failed'},
                alerts=[{'type': 'error', 'message': str(e), 'severity': 'high'}]
            )

    def analyze_skill_gaps(self, learner_id: Optional[str] = None) -> Optional[SkillGapAnalysis]:
        """
        Analyze skill gaps in the learning system.

        Args:
            learner_id: Specific learner to analyze (None for system-wide)

        Returns:
            Skill gap analysis with identified gaps and recommendations
        """
        try:
            timestamp = time.time()

            # Check if there are any interactions - if not, return empty analysis for empty state
            has_interactions = False
            if self.history:
                interactions = self.history.get_all_interactions_chronological()
                has_interactions = len(interactions) > 0
            elif self.prompt_controller:
                metrics = self.prompt_controller.get_learning_metrics()
                has_interactions = metrics.get('total_interactions', 0) > 0

            # For empty state (no interactions), return empty skill gap analysis
            if not has_interactions:
                analysis = SkillGapAnalysis(
                    learner_id=learner_id or "system",
                    timestamp=timestamp,
                    skill_gaps=[],  # No gaps when no learning data exists
                    overall_gap_score=0.0,  # No gaps = perfect score
                    critical_gaps_count=0,
                    recommendations=["No learning data available yet. Start interacting with the system to generate skill gap analysis."]
                )
                logger.info("Empty state: No skill gaps to analyze (no learning data available)")
                return analysis

            # Get curriculum data to understand required skills
            skill_requirements = self._get_skill_requirements()

            # Get current skill levels
            current_skills = self._get_current_skill_levels(learner_id)

            skill_gaps = []
            total_gap_score = 0.0
            critical_gaps = 0

            for skill_id, required_level in skill_requirements.items():
                current_level = current_skills.get(skill_id, 0.0)
                gap_size = max(0.0, required_level - current_level)

                if gap_size > 0.1:  # Only consider meaningful gaps
                    # Determine severity
                    severity = self._calculate_gap_severity(gap_size, skill_id)

                    # Get affected tasks and suggestions
                    affected_tasks = self._get_affected_tasks(skill_id)
                    suggestions = self._generate_gap_recommendations(skill_id, gap_size)

                    skill_gap = SkillGap(
                        skill_id=skill_id,
                        skill_name=self._get_skill_name(skill_id),
                        current_level=current_level,
                        required_level=required_level,
                        gap_size=gap_size,
                        severity=severity,
                        affected_tasks=affected_tasks,
                        improvement_suggestions=suggestions,
                        estimated_time_to_close=self._estimate_gap_closure_time(gap_size, skill_id)
                    )

                    skill_gaps.append(skill_gap)
                    total_gap_score += gap_size

                    if severity in [SkillGapSeverity.CRITICAL, SkillGapSeverity.HIGH]:
                        critical_gaps += 1

            # Calculate overall gap score
            overall_gap_score = min(1.0, total_gap_score / max(1, len(skill_gaps)))

            # Generate recommendations
            recommendations = self._generate_overall_gap_recommendations(skill_gaps)

            analysis = SkillGapAnalysis(
                learner_id=learner_id or "system",
                timestamp=timestamp,
                skill_gaps=skill_gaps,
                overall_gap_score=overall_gap_score,
                critical_gaps_count=critical_gaps,
                recommendations=recommendations
            )

            logger.info(f"Completed skill gap analysis: {len(skill_gaps)} gaps found, "
                        f"overall_score={overall_gap_score:.3f}")

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze skill gaps: {str(e)}")
            return None

    def detect_bottlenecks(self, learner_id: Optional[str] = None) -> Optional[BottleneckDetection]:
        """
        Detect learning bottlenecks in the system.

        Args:
            learner_id: Specific learner to analyze (None for system-wide)

        Returns:
            Bottleneck detection analysis with identified issues
        """
        try:
            timestamp = time.time()

            # Check if there are any interactions - if not, return empty analysis for empty state
            has_interactions = False
            if self.history:
                interactions = self.history.get_all_interactions_chronological()
                has_interactions = len(interactions) > 0
            elif self.prompt_controller:
                metrics = self.prompt_controller.get_learning_metrics()
                has_interactions = metrics.get('total_interactions', 0) > 0

            # For empty state (no interactions), return empty bottleneck analysis
            if not has_interactions:
                analysis = BottleneckDetection(
                    timestamp=timestamp,
                    bottlenecks=[],  # No bottlenecks when no learning data exists
                    overall_bottleneck_score=0.0,  # No bottlenecks = perfect health
                    critical_bottlenecks_count=0,
                    system_health_score=1.0,  # Perfect health when no data
                    recommendations=["No learning data available yet. Start interacting with the system to generate bottleneck analysis."]
                )
                logger.info("Empty state: No bottlenecks to detect (no learning data available)")
                return analysis

            bottlenecks = []

            # Check various bottleneck types
            bottlenecks.extend(self._detect_concept_bottlenecks(learner_id))
            bottlenecks.extend(self._detect_practical_bottlenecks(learner_id))
            bottlenecks.extend(self._detect_motivational_bottlenecks(learner_id))
            bottlenecks.extend(self._detect_resource_bottlenecks())
            bottlenecks.extend(self._detect_technical_bottlenecks())

            # Calculate overall scores
            overall_bottleneck_score = sum(b.severity for b in bottlenecks) / max(1, len(bottlenecks))
            critical_bottlenecks = sum(1 for b in bottlenecks if b.severity > 0.7)

            # System health score (inverse of bottleneck score)
            system_health_score = 1.0 - overall_bottleneck_score

            # Generate recommendations
            recommendations = self._generate_bottleneck_recommendations(bottlenecks)

            analysis = BottleneckDetection(
                timestamp=timestamp,
                bottlenecks=bottlenecks,
                overall_bottleneck_score=overall_bottleneck_score,
                critical_bottlenecks_count=critical_bottlenecks,
                system_health_score=system_health_score,
                recommendations=recommendations
            )

            logger.info(f"Completed bottleneck detection: {len(bottlenecks)} bottlenecks found, "
                        f"health_score={system_health_score:.3f}")

            return analysis

        except Exception as e:
            logger.error(f"Failed to detect bottlenecks: {str(e)}")
            return None

    def predict_performance(self, learner_id: str, horizon_days: int = 30) -> Optional[PerformancePrediction]:
        """
        Predict future learning performance using historical data.

        Args:
            learner_id: Learner to predict for
            horizon_days: Prediction horizon in days

        Returns:
            Performance prediction with confidence and factors
        """
        try:
            timestamp = time.time()

            # Get historical performance data
            historical_data = self._get_historical_performance(learner_id)

            if not historical_data:
                return None

            # Simple linear regression for prediction
            predicted_performance = self._predict_future_performance(historical_data, horizon_days)

            # Calculate confidence based on data quality and consistency
            confidence_score = self._calculate_prediction_confidence(historical_data)

            confidence_level = self._get_confidence_level(confidence_score)

            # Identify influencing factors
            factors = self._identify_performance_factors(learner_id, historical_data)

            # Risk assessment
            risk_factors = self._assess_performance_risks(historical_data, predicted_performance)

            # Generate improvement trajectory
            trajectory = self._generate_improvement_trajectory(historical_data, horizon_days)

            # Recommendations
            recommendations = self._generate_performance_recommendations(
                predicted_performance, factors, risk_factors
            )

            prediction = PerformancePrediction(
                learner_id=learner_id,
                prediction_horizon=horizon_days,
                timestamp=timestamp,
                predicted_performance=predicted_performance,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                factors_influencing=factors,
                risk_factors=risk_factors,
                improvement_trajectory=trajectory,
                recommendations=recommendations
            )

            logger.info(f"Generated performance prediction for {learner_id}: "
                       f"predicted={predicted_performance:.3f}, confidence={confidence_level}")

            return prediction

        except Exception as e:
            logger.error(f"Failed to predict performance for {learner_id}: {str(e)}")
            return None

    def generate_learning_insights(self, learner_id: Optional[str] = None) -> Optional[LearningInsights]:
        """
        Generate actionable learning insights and recommendations.

        Args:
            learner_id: Specific learner to analyze (None for system-wide)

        Returns:
            Learning insights with prioritized recommendations
        """
        try:
            timestamp = time.time()

            # Check if there are any interactions - if not, return empty insights for empty state
            has_interactions = False
            if self.history:
                interactions = self.history.get_all_interactions_chronological()
                has_interactions = len(interactions) > 0
            elif self.prompt_controller:
                metrics = self.prompt_controller.get_learning_metrics()
                has_interactions = metrics.get('total_interactions', 0) > 0

            # For empty state (no interactions), return empty insights
            if not has_interactions:
                learning_insights = LearningInsights(
                    timestamp=timestamp,
                    insights=[],  # No insights when no learning data exists
                    overall_insight_score=0.0,  # No insights = neutral score
                    high_priority_count=0,
                    categories_covered=[],
                    implementation_roadmap=[],
                    insights_metadata={'empty_state': True, 'message': 'No learning data available yet. Start interacting with the system to generate insights.'}
                )
                logger.info("Empty state: No learning insights available (no learning data)")
                return learning_insights

            insights = []

            # Generate insights from different analysis areas
            insights.extend(self._generate_skill_based_insights(learner_id))
            insights.extend(self._generate_performance_insights(learner_id))
            insights.extend(self._generate_system_insights())
            insights.extend(self._generate_personalized_insights(learner_id))

            # Sort by priority and impact
            insights.sort(key=lambda x: (self._priority_score(x.priority), x.impact_score), reverse=True)

            # Calculate overall insight score
            overall_score = sum(i.impact_score * i.confidence for i in insights) / max(1, len(insights))

            # Count high priority insights
            high_priority_count = sum(1 for i in insights if i.priority in [InsightPriority.HIGH, InsightPriority.URGENT])

            # Get categories covered
            categories = list(set(i.category for i in insights))

            # Generate implementation roadmap
            roadmap = self._generate_implementation_roadmap(insights)

            learning_insights = LearningInsights(
                timestamp=timestamp,
                insights=insights,
                overall_insight_score=overall_score,
                high_priority_count=high_priority_count,
                categories_covered=categories,
                implementation_roadmap=roadmap
            )

            logger.info(f"Generated {len(insights)} learning insights, "
                        f"{high_priority_count} high priority")

            return learning_insights

        except Exception as e:
            logger.error(f"Failed to generate learning insights: {str(e)}")
            return None

    def monitor_system_health(self) -> Optional[SystemHealthMetrics]:
        """
        Monitor overall system health and performance.

        Returns:
            System health metrics with component status
        """
        try:
            timestamp = time.time()

            # Component health checks
            component_health = {}

            if self.reward_service:
                component_health['reward_service'] = self._check_reward_service_health()

            if self.hallucination_service:
                component_health['hallucination_service'] = self._check_hallucination_service_health()

            if self.meta_learning_service:
                component_health['meta_learning_service'] = self._check_meta_learning_health()

            if self.external_llm_service:
                component_health['external_llm_service'] = self._check_external_llm_health()

            # Overall health score
            overall_health = sum(component_health.values()) / max(1, len(component_health))

            # Performance metrics
            performance_metrics = self._gather_performance_metrics()

            # Error rates
            error_rates = self._calculate_error_rates()

            # Resource utilization
            resource_utilization = self._monitor_resource_utilization()

            # Generate alerts
            alerts = self._generate_health_alerts(component_health, error_rates)

            health = SystemHealthMetrics(
                timestamp=timestamp,
                overall_health_score=overall_health,
                component_health=component_health,
                performance_metrics=performance_metrics,
                error_rates=error_rates,
                resource_utilization=resource_utilization,
                alerts=alerts
            )

            # Store in history for trend analysis
            self.system_health_history.append(health)
            if len(self.system_health_history) > 1000:
                self.system_health_history = self.system_health_history[-1000:]

            logger.info(f"System health check completed: overall_score={overall_health:.3f}")

            return health

        except Exception as e:
            logger.error(f"Failed to monitor system health: {str(e)}")
            return None

    def analyze_trends(self, timeframe: str = "day") -> List[TrendAnalysis]:
        """
        Analyze trends in learning metrics over time.

        Args:
            timeframe: Analysis timeframe ("hour", "day", "week", "month")

        Returns:
            List of trend analyses for different metrics
        """
        try:
            trends = []

            # Analyze different metrics
            metrics_to_analyze = [
                'learning_performance', 'system_health', 'error_rate',
                'skill_improvement', 'bottleneck_severity'
            ]

            for metric in metrics_to_analyze:
                trend = self._analyze_metric_trend(metric, timeframe)
                if trend:
                    trends.append(trend)

            logger.info(f"Completed trend analysis for {len(trends)} metrics")

            return trends

        except Exception as e:
            logger.error(f"Failed to analyze trends: {str(e)}")
            return []

    def detect_anomalies(self) -> List[AnomalyDetection]:
        """
        Detect anomalies in learning patterns and system behavior.

        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []

            # Check various metrics for anomalies
            anomalies.extend(self._detect_performance_anomalies())
            anomalies.extend(self._detect_system_anomalies())
            anomalies.extend(self._detect_learning_anomalies())

            # Store anomalies for historical tracking
            self.anomaly_history.extend(anomalies)
            if len(self.anomaly_history) > 500:
                self.anomaly_history = self.anomaly_history[-500:]

            logger.info(f"Detected {len(anomalies)} anomalies")

            return anomalies

        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            return []

    # Helper methods for skill gap analysis
    def _get_skill_requirements(self) -> Dict[str, float]:
        """Get required skill levels from curriculum."""
        requirements = {}

        if self.curriculum_service:
            try:
                # Get curriculum skills and their required levels
                curriculum_tree = getattr(self.curriculum_service, 'curriculum_tree', None)
                if curriculum_tree and hasattr(curriculum_tree, 'skills'):
                    for skill_id, skill in curriculum_tree.skills.items():
                        requirements[skill_id] = getattr(skill, 'required_level', 0.8)
            except Exception as e:
                logger.warning(f"Failed to get skill requirements from curriculum: {str(e)}")

        # Default requirements if curriculum not available
        if not requirements:
            requirements = {
                'ml_fundamentals': 0.8,
                'prompt_engineering': 0.7,
                'data_analysis': 0.6,
                'model_evaluation': 0.7,
                'problem_solving': 0.8
            }

        return requirements

    def _get_current_skill_levels(self, learner_id: Optional[str]) -> Dict[str, float]:
        """Get current skill levels for a learner."""
        current_skills = {}

        # Try to get from meta-learning service
        if self.meta_learning_service:
            try:
                skill_levels = self.meta_learning_service.meta_learner.get_current_skill_levels(learner_id)
                if skill_levels:
                    current_skills.update(skill_levels)
            except Exception as e:
                logger.warning(f"Failed to get skill levels from meta-learning service: {str(e)}")

        # Estimate from reward history and interactions
        if self.history:
            try:
                interactions = self.history.get_all_interactions_chronological()
                if interactions:
                    # Simple heuristic: higher rewards indicate better skills
                    avg_reward = sum(i.get('reward', 0) for i in interactions) / len(interactions)
                    # Distribute across skills based on interaction patterns
                    current_skills = {
                        'ml_fundamentals': min(1.0, avg_reward + 0.2),
                        'prompt_engineering': min(1.0, avg_reward + 0.1),
                        'data_analysis': min(1.0, avg_reward),
                        'model_evaluation': min(1.0, avg_reward + 0.15),
                        'problem_solving': min(1.0, avg_reward + 0.05)
                    }
            except Exception as e:
                logger.warning(f"Failed to estimate skill levels from history: {str(e)}")

        return current_skills

    def _calculate_gap_severity(self, gap_size: float, skill_id: str) -> SkillGapSeverity:
        """Calculate severity of a skill gap."""
        if gap_size > 0.5:
            return SkillGapSeverity.CRITICAL
        elif gap_size > 0.3:
            return SkillGapSeverity.HIGH
        elif gap_size > 0.15:
            return SkillGapSeverity.MEDIUM
        else:
            return SkillGapSeverity.LOW

    # Helper methods for bottleneck detection
    def _detect_concept_bottlenecks(self, learner_id: Optional[str]) -> List[LearningBottleneck]:
        """Detect conceptual learning bottlenecks."""
        bottlenecks = []

        try:
            # Check for repeated failures on similar concepts
            if self.history:
                interactions = self.history.get_all_interactions_chronological()
                recent_failures = [i for i in interactions[-50:] if i.get('reward', 1.0) < 0.3]

                if len(recent_failures) > 10:
                    bottlenecks.append(LearningBottleneck(
                        bottleneck_id=f"concept_stagnation_{int(time.time())}",
                        type=BottleneckType.CONCEPTUAL,
                        description="Persistent difficulty with core concepts",
                        severity=min(1.0, len(recent_failures) / 20.0),
                        affected_area="Core ML Concepts",
                        root_causes=["Insufficient foundational knowledge", "Complex explanations"],
                        impact_score=0.8,
                        resolution_suggestions=[
                            "Review fundamental ML concepts",
                            "Use simpler learning materials",
                            "Practice with basic examples first"
                        ]
                    ))
        except Exception as e:
            logger.warning(f"Failed to detect concept bottlenecks: {str(e)}")

        return bottlenecks

    def _detect_practical_bottlenecks(self, learner_id: Optional[str]) -> List[LearningBottleneck]:
        """Detect practical application bottlenecks."""
        bottlenecks = []

        try:
            # Check for low success rates in practical tasks
            if self.reward_service:
                avg_metrics = self.reward_service.get_average_metrics(window=50)
                success_rate = avg_metrics.get('total_reward', 0.5)

                if success_rate < 0.4:
                    bottlenecks.append(LearningBottleneck(
                        bottleneck_id=f"practical_application_{int(time.time())}",
                        type=BottleneckType.PRACTICAL,
                        description="Difficulty applying concepts practically",
                        severity=1.0 - success_rate,
                        affected_area="Practical Implementation",
                        root_causes=["Gap between theory and practice", "Insufficient hands-on experience"],
                        impact_score=0.7,
                        resolution_suggestions=[
                            "Increase hands-on practice sessions",
                            "Work on real-world projects",
                            "Get mentorship for practical guidance"
                        ]
                    ))
        except Exception as e:
            logger.warning(f"Failed to detect practical bottlenecks: {str(e)}")

        return bottlenecks

    def _detect_motivational_bottlenecks(self, learner_id: Optional[str]) -> List[LearningBottleneck]:
        """Detect motivational learning bottlenecks."""
        bottlenecks = []

        try:
            # Check for declining engagement patterns
            if self.history:
                interactions = self.history.get_all_interactions_chronological()
                if len(interactions) > 20:
                    recent_rewards = [i.get('reward', 0) for i in interactions[-20:]]
                    older_rewards = [i.get('reward', 0) for i in interactions[-40:-20]]

                    if recent_rewards and older_rewards:
                        recent_avg = sum(recent_rewards) / len(recent_rewards)
                        older_avg = sum(older_rewards) / len(older_rewards)

                        if older_avg - recent_avg > 0.2:
                            bottlenecks.append(LearningBottleneck(
                                bottleneck_id=f"motivation_decline_{int(time.time())}",
                                type=BottleneckType.MOTIVATIONAL,
                                description="Declining motivation and engagement",
                                severity=min(1.0, (older_avg - recent_avg) * 2),
                                affected_area="Learning Motivation",
                                root_causes=["Loss of interest", "Frustration with difficult concepts", "Lack of progress feedback"],
                                impact_score=0.6,
                                resolution_suggestions=[
                                    "Set smaller, achievable goals",
                                    "Celebrate small wins",
                                    "Find more engaging learning materials",
                                    "Take short breaks to maintain motivation"
                                ]
                            ))
        except Exception as e:
            logger.warning(f"Failed to detect motivational bottlenecks: {str(e)}")

        return bottlenecks

    def _detect_resource_bottlenecks(self) -> List[LearningBottleneck]:
        """Detect resource-related bottlenecks."""
        bottlenecks = []

        try:
            # Check system resource utilization
            if hasattr(self, 'system_health_history') and self.system_health_history:
                recent_health = self.system_health_history[-1]
                resource_usage = recent_health.resource_utilization

                for resource, usage in resource_usage.items():
                    if usage > 0.9:  # Over 90% utilization
                        bottlenecks.append(LearningBottleneck(
                            bottleneck_id=f"resource_{resource}_{int(time.time())}",
                            type=BottleneckType.RESOURCE,
                            description=f"High {resource} utilization impacting performance",
                            severity=min(1.0, usage),
                            affected_area=f"System Resources ({resource})",
                            root_causes=[f"Insufficient {resource} capacity", "Resource-intensive operations"],
                            impact_score=0.5,
                            resolution_suggestions=[
                                f"Optimize {resource} usage",
                                f"Consider scaling {resource} resources",
                                "Monitor resource usage patterns"
                            ]
                        ))
        except Exception as e:
            logger.warning(f"Failed to detect resource bottlenecks: {str(e)}")

        return bottlenecks

    def _detect_technical_bottlenecks(self) -> List[LearningBottleneck]:
        """Detect technical system bottlenecks."""
        bottlenecks = []

        try:
            # Check for high error rates
            if hasattr(self, 'system_health_history') and self.system_health_history:
                recent_health = self.system_health_history[-1]
                error_rates = recent_health.error_rates

                for component, error_rate in error_rates.items():
                    if error_rate > 0.1:  # Over 10% error rate
                        bottlenecks.append(LearningBottleneck(
                            bottleneck_id=f"technical_{component}_{int(time.time())}",
                            type=BottleneckType.TECHNICAL,
                            description=f"High error rate in {component}",
                            severity=min(1.0, error_rate * 2),
                            affected_area=f"Technical Components ({component})",
                            root_causes=["Software bugs", "Integration issues", "Configuration problems"],
                            impact_score=0.9,
                            resolution_suggestions=[
                                f"Debug {component} errors",
                                "Check system logs",
                                "Review recent configuration changes"
                            ]
                        ))
        except Exception as e:
            logger.warning(f"Failed to detect technical bottlenecks: {str(e)}")

        return bottlenecks

    # Helper methods for performance prediction
    def _get_historical_performance(self, learner_id: str) -> List[Dict[str, Any]]:
        """Get historical performance data for a learner."""
        historical_data = []

        try:
            if self.history:
                interactions = self.history.get_all_interactions_chronological()
                for interaction in interactions[-100:]:  # Last 100 interactions
                    historical_data.append({
                        'timestamp': interaction.get('timestamp', time.time()),
                        'reward': interaction.get('reward', 0),
                        'action': interaction.get('action', 0),
                        'success': interaction.get('reward', 0) > 0.5
                    })
        except Exception as e:
            logger.warning(f"Failed to get historical performance for {learner_id}: {str(e)}")

        return historical_data

    def _predict_future_performance(self, historical_data: List[Dict], horizon_days: int) -> float:
        """Predict future performance using simple trend analysis."""
        if not historical_data:
            return 0.5

        # Simple linear trend
        rewards = [d['reward'] for d in historical_data]
        if len(rewards) < 2:
            return rewards[0] if rewards else 0.5

        # Calculate trend slope
        x = list(range(len(rewards)))
        slope = np.polyfit(x, rewards, 1)[0]

        # Predict future value
        future_steps = min(horizon_days, len(rewards))  # Simple heuristic
        predicted = rewards[-1] + slope * future_steps

        return max(0.0, min(1.0, predicted))

    def _calculate_prediction_confidence(self, historical_data: List[Dict]) -> float:
        """Calculate confidence in performance prediction."""
        if len(historical_data) < 5:
            return 0.3

        rewards = [d['reward'] for d in historical_data]

        # Confidence based on data consistency and amount
        consistency = 1.0 - (np.std(rewards) / max(0.1, np.mean(rewards)))
        data_amount = min(1.0, len(rewards) / 50.0)

        return min(1.0, (consistency * 0.7 + data_amount * 0.3))

    # Helper methods for insights generation
    def _generate_skill_based_insights(self, learner_id: Optional[str]) -> List[LearningInsight]:
        """Generate insights based on skill analysis."""
        insights = []

        try:
            skill_gaps = self.analyze_skill_gaps(learner_id)
            if skill_gaps and skill_gaps.skill_gaps:
                # Focus on critical gaps
                critical_gaps = [g for g in skill_gaps.skill_gaps if g.severity == SkillGapSeverity.CRITICAL]

                if critical_gaps:
                    insights.append(LearningInsight(
                        insight_id=f"critical_skill_gaps_{int(time.time())}",
                        title="Critical Skill Gaps Identified",
                        description=f"Found {len(critical_gaps)} critical skill gaps that need immediate attention",
                        priority=InsightPriority.URGENT,
                        category="Skills",
                        impact_score=0.9,
                        confidence=0.8,
                        actionable_steps=[
                            "Prioritize learning the identified critical skills",
                            "Allocate dedicated study time for gap closure",
                            "Seek expert guidance for complex topics"
                        ],
                        expected_benefits=[
                            "Rapid improvement in overall performance",
                            "Reduced frustration with difficult concepts",
                            "Better learning outcomes"
                        ],
                        implementation_complexity="medium",
                        timeframe="immediate"
                    ))
        except Exception as e:
            logger.warning(f"Failed to generate skill-based insights: {str(e)}")

        return insights

    def _generate_performance_insights(self, learner_id: Optional[str]) -> List[LearningInsight]:
        """Generate insights based on performance analysis."""
        insights = []

        try:
            if learner_id:
                prediction = self.predict_performance(learner_id)
                if prediction and prediction.predicted_performance < 0.5:
                    insights.append(LearningInsight(
                        insight_id=f"performance_improvement_{int(time.time())}",
                        title="Performance Improvement Needed",
                        description="Predicted performance indicates need for intervention",
                        priority=InsightPriority.HIGH,
                        category="Performance",
                        impact_score=0.8,
                        confidence=prediction.confidence_score,
                        actionable_steps=[
                            "Review recent learning activities",
                            "Identify and address knowledge gaps",
                            "Consider changing learning strategies"
                        ],
                        expected_benefits=[
                            "Improved learning outcomes",
                            "Better skill acquisition",
                            "Increased motivation"
                        ],
                        implementation_complexity="medium",
                        timeframe="short_term"
                    ))
        except Exception as e:
            logger.warning(f"Failed to generate performance insights: {str(e)}")

        return insights

    def _generate_system_insights(self) -> List[LearningInsight]:
        """Generate system-wide insights."""
        insights = []

        try:
            health = self.monitor_system_health()
            if health and health.overall_health_score < 0.7:
                insights.append(LearningInsight(
                    insight_id=f"system_health_{int(time.time())}",
                    title="System Health Issues Detected",
                    description="System performance and reliability need attention",
                    priority=InsightPriority.HIGH,
                    category="System",
                    impact_score=0.7,
                    confidence=0.9,
                    actionable_steps=[
                        "Review system error logs",
                        "Check resource utilization",
                        "Optimize system configuration"
                    ],
                    expected_benefits=[
                        "Improved system reliability",
                        "Better user experience",
                        "Reduced downtime"
                    ],
                    implementation_complexity="high",
                    timeframe="short_term"
                ))
        except Exception as e:
            logger.warning(f"Failed to generate system insights: {str(e)}")

        return insights

    def _generate_personalized_insights(self, learner_id: Optional[str]) -> List[LearningInsight]:
        """Generate personalized insights for specific learners."""
        insights = []

        try:
            bottlenecks = self.detect_bottlenecks(learner_id)
            if bottlenecks and bottlenecks.bottlenecks:
                # Create insights from bottlenecks
                for bottleneck in bottlenecks.bottlenecks[:3]:  # Top 3 bottlenecks
                    insights.append(LearningInsight(
                        insight_id=f"bottleneck_{bottleneck.bottleneck_id}",
                        title=f"Address {bottleneck.type.value.title()} Bottleneck",
                        description=bottleneck.description,
                        priority=InsightPriority.HIGH if bottleneck.severity > 0.7 else InsightPriority.MEDIUM,
                        category="Personalized",
                        impact_score=bottleneck.impact_score,
                        confidence=0.8,
                        actionable_steps=bottleneck.resolution_suggestions,
                        expected_benefits=["Improved learning progress", "Reduced frustration"],
                        implementation_complexity="medium",
                        timeframe="short_term"
                    ))
        except Exception as e:
            logger.warning(f"Failed to generate personalized insights: {str(e)}")

        return insights

    # Helper methods for system health monitoring
    def _check_reward_service_health(self) -> float:
        """Check reward service health."""
        try:
            if not self.reward_service:
                return 0.0

            # Check if service is responsive and has data
            metrics = self.reward_service.get_current_metrics()
            history_length = metrics.get('history_length', 0)

            # Health based on data availability and recency
            health = min(1.0, history_length / 100.0)
            return health
        except Exception:
            return 0.0

    def _check_hallucination_service_health(self) -> float:
        """Check hallucination service health."""
        try:
            if not self.hallucination_service:
                return 0.0

            metrics = self.hallucination_service.get_metrics()
            total_checks = metrics.total_checks

            health = min(1.0, total_checks / 50.0)
            return health
        except Exception:
            return 0.0

    def _check_meta_learning_health(self) -> float:
        """Check meta-learning service health."""
        try:
            if not self.meta_learning_service:
                return 0.0

            status = self.meta_learning_service.meta_learner.get_meta_learning_status()
            # Simple health check based on having strategies
            strategy_count = len(status.get('strategy_history', []))
            health = min(1.0, strategy_count / 5.0)
            return health
        except Exception:
            return 0.0

    def _check_external_llm_health(self) -> float:
        """Check external LLM service health."""
        try:
            if not self.external_llm_service:
                return 0.0

            # Check if service can provide models
            models = self.external_llm_service.get_available_models()
            health = 1.0 if models else 0.0
            return health
        except Exception:
            return 0.0

    def _gather_performance_metrics(self) -> Dict[str, Any]:
        """Gather system performance metrics."""
        metrics = {}

        try:
            # Response times, throughput, etc.
            if self.prompt_controller:
                learning_metrics = self.prompt_controller.get_learning_metrics()
                metrics.update({
                    'success_rate': learning_metrics.get('success_rate', 0),
                    'total_interactions': learning_metrics.get('total_interactions', 0)
                })
        except Exception as e:
            logger.warning(f"Failed to gather performance metrics: {str(e)}")

        return metrics

    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates for different components."""
        error_rates = {}

        try:
            if self.system_health_history:
                recent_health = self.system_health_history[-10:]  # Last 10 health checks
                component_rates = defaultdict(list)

                for health in recent_health:
                    for component, rate in health.error_rates.items():
                        component_rates[component].append(rate)

                error_rates = {}
                for component, rates in component_rates.items():
                    error_rates[component] = sum(rates) / len(rates) if rates else 0.0
            else:
                # Default values if no history available
                error_rates = {
                    'api_errors': 0.02,
                    'service_errors': 0.01,
                    'integration_errors': 0.005
                }
        except Exception as e:
            logger.warning(f"Failed to calculate error rates: {str(e)}")
            error_rates = {}

        return error_rates

    def _monitor_resource_utilization(self) -> Dict[str, float]:
        """Monitor system resource utilization."""
        utilization = {}

        try:
            utilization = {
                'cpu': psutil.cpu_percent(interval=0.1) / 100.0,
                'memory': psutil.virtual_memory().percent / 100.0,
                'disk': psutil.disk_usage('/').percent / 100.0,
                'network': 0.0  # Network utilization calculation requires more complex monitoring
            }
        except Exception as e:
            logger.warning(f"Failed to monitor resource utilization: {str(e)}")
            utilization = {}

        return utilization

    # Placeholder methods for remaining functionality
    def _get_affected_tasks(self, skill_id: str) -> List[str]:
        """Get tasks affected by a skill gap."""
        return ["Task 1", "Task 2", "Task 3"]

    def _generate_gap_recommendations(self, skill_id: str, gap_size: float) -> List[str]:
        """Generate recommendations for closing skill gaps."""
        return ["Practice more", "Study documentation", "Take courses"]

    def _estimate_gap_closure_time(self, gap_size: float, skill_id: str) -> Optional[int]:
        """Estimate time to close a skill gap."""
        return int(gap_size * 40)  # Rough estimate

    def _get_skill_name(self, skill_id: str) -> str:
        """Get human-readable skill name."""
        return skill_id.replace('_', ' ').title()

    def _generate_overall_gap_recommendations(self, skill_gaps: List[SkillGap]) -> List[str]:
        """Generate overall recommendations for skill gaps."""
        return ["Focus on critical gaps first", "Create study plan", "Seek mentorship"]

    def _generate_bottleneck_recommendations(self, bottlenecks: List[LearningBottleneck]) -> List[str]:
        """Generate recommendations for bottlenecks."""
        return ["Address high-severity bottlenecks first", "Monitor progress", "Adjust learning strategy"]

    def _identify_performance_factors(self, learner_id: str, historical_data: List[Dict]) -> Dict[str, float]:
        """Identify factors influencing performance."""
        return {"consistency": 0.8, "practice": 0.7, "difficulty": 0.6}

    def _assess_performance_risks(self, historical_data: List[Dict], predicted: float) -> List[str]:
        """Assess risks to performance prediction."""
        return ["Inconsistent learning patterns", "High variability in results"]

    def _generate_improvement_trajectory(self, historical_data: List[Dict], horizon: int) -> List[Dict[str, Any]]:
        """Generate improvement trajectory."""
        return [{"day": i, "predicted_performance": 0.5 + i * 0.01} for i in range(horizon)]

    def _generate_performance_recommendations(self, predicted: float, factors: Dict[str, float], risks: List[str]) -> List[str]:
        """Generate performance recommendations."""
        return ["Increase practice frequency", "Focus on weak areas", "Monitor progress regularly"]

    def _generate_implementation_roadmap(self, insights: List[LearningInsight]) -> List[Dict[str, Any]]:
        """Generate implementation roadmap for insights."""
        return [{"phase": "immediate", "insights": [i.insight_id for i in insights if i.timeframe == "immediate"]}]

    def _priority_score(self, priority: InsightPriority) -> int:
        """Convert priority to numeric score."""
        scores = {
            InsightPriority.LOW: 1,
            InsightPriority.MEDIUM: 2,
            InsightPriority.HIGH: 3,
            InsightPriority.URGENT: 4
        }
        return scores.get(priority, 1)

    def _calculate_overall_learning_score(self, skill_gaps: Optional[SkillGapAnalysis],
                                        bottlenecks: Optional[BottleneckDetection],
                                        health: Optional[SystemHealthMetrics]) -> float:
        """Calculate overall learning system score."""
        scores = []

        if skill_gaps:
            scores.append(1.0 - skill_gaps.overall_gap_score)

        if bottlenecks:
            scores.append(bottlenecks.system_health_score)

        if health:
            scores.append(health.overall_health_score)

        return sum(scores) / max(1, len(scores)) if scores else 0.5

    def _generate_key_indicators(self, skill_gaps: Optional[SkillGapAnalysis],
                               bottlenecks: Optional[BottleneckDetection],
                               health: Optional[SystemHealthMetrics]) -> Dict[str, Any]:
        """Generate key performance indicators."""
        indicators = {}

        if skill_gaps:
            indicators['skill_gaps'] = len(skill_gaps.skill_gaps)
            indicators['critical_gaps'] = skill_gaps.critical_gaps_count

        if bottlenecks:
            indicators['bottlenecks'] = len(bottlenecks.bottlenecks)
            indicators['system_health'] = bottlenecks.system_health_score

        if health:
            indicators['overall_health'] = health.overall_health_score

        return indicators

    def _generate_alerts(self, anomalies: List[AnomalyDetection],
                        health: Optional[SystemHealthMetrics]) -> List[Dict[str, Any]]:
        """Generate system alerts."""
        alerts = []

        # Anomaly alerts
        for anomaly in anomalies:
            if anomaly.severity > 0.7:
                alerts.append({
                    'type': 'anomaly',
                    'message': anomaly.description,
                    'severity': 'high',
                    'component': anomaly.affected_metric
                })

        # Health alerts
        if health and health.overall_health_score < 0.6:
            alerts.append({
                'type': 'health',
                'message': 'System health is degraded',
                'severity': 'high',
                'component': 'system'
            })

        return alerts

    def _generate_health_alerts(self, component_health: Dict[str, float],
                              error_rates: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate health-specific alerts."""
        alerts = []

        for component, health in component_health.items():
            if health < 0.5:
                alerts.append({
                    'type': 'component_health',
                    'component': component,
                    'message': f'{component} health is degraded ({health:.2f})',
                    'severity': 'medium'
                })

        for component, error_rate in error_rates.items():
            if error_rate > 0.05:
                alerts.append({
                    'type': 'error_rate',
                    'component': component,
                    'message': f'High error rate in {component} ({error_rate:.2%})',
                    'severity': 'high'
                })

        return alerts

    def _analyze_metric_trend(self, metric: str, timeframe: str) -> Optional[TrendAnalysis]:
        """Analyze trend for a specific metric."""
        # Placeholder implementation
        return TrendAnalysis(
            metric_name=metric,
            timeframe=timeframe,
            trend_direction="stable",
            trend_strength=0.5,
            data_points=[],
            statistical_significance=0.8,
            insights=["Trend analysis not fully implemented"]
        )

    def _detect_performance_anomalies(self) -> List[AnomalyDetection]:
        """Detect performance-related anomalies."""
        # Placeholder implementation
        return []

    def _detect_system_anomalies(self) -> List[AnomalyDetection]:
        """Detect system-related anomalies."""
        # Placeholder implementation
        return []

    def _detect_learning_anomalies(self) -> List[AnomalyDetection]:
        """Detect learning-related anomalies."""
        anomalies = []

        try:
            # Check for feedback-based anomalies
            if self.feedback_service:
                feedback_anomalies = self._detect_feedback_anomalies()
                anomalies.extend(feedback_anomalies)
        except Exception as e:
            logger.warning(f"Failed to detect feedback anomalies: {str(e)}")

        return anomalies

    def _detect_feedback_anomalies(self) -> List[AnomalyDetection]:
        """Detect anomalies in user feedback patterns."""
        anomalies = []

        try:
            # Get feedback metrics
            feedback_metrics = self.feedback_service.get_metrics()

            # Check for unusually low engagement
            if feedback_metrics.user_engagement_rate < 0.1:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"low_engagement_{int(time.time())}",
                    description="Unusually low user engagement with feedback system",
                    severity=0.7,
                    affected_metric="user_engagement",
                    detected_value=feedback_metrics.user_engagement_rate,
                    expected_range=(0.2, 1.0),
                    potential_cause="System issues or user dissatisfaction",
                    recommended_actions=[
                        "Check system for technical issues",
                        "Survey users about satisfaction",
                        "Review recent changes that might affect usability"
                    ],
                    timestamp=time.time(),
                    confidence=0.8
                ))

            # Check for sudden drops in feedback quality
            if feedback_metrics.feedback_quality_score < 0.5:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"feedback_quality_drop_{int(time.time())}",
                    description="Feedback quality has dropped significantly",
                    severity=0.6,
                    affected_metric="feedback_quality",
                    detected_value=feedback_metrics.feedback_quality_score,
                    expected_range=(0.7, 1.0),
                    potential_cause="Changes in user experience or system behavior",
                    recommended_actions=[
                        "Analyze recent feedback content",
                        "Check for UI/UX issues",
                        "Review system changes that might affect user interaction"
                    ],
                    timestamp=time.time(),
                    confidence=0.75
                ))

            # Check for collaborative learning stagnation
            if feedback_metrics.collaborative_patterns_learned < 1:
                anomalies.append(AnomalyDetection(
                    anomaly_id=f"collaborative_stagnation_{int(time.time())}",
                    description="No new collaborative learning patterns detected recently",
                    severity=0.4,
                    affected_metric="collaborative_learning",
                    detected_value=feedback_metrics.collaborative_patterns_learned,
                    expected_range=(1, float('inf')),
                    potential_cause="Insufficient diverse feedback or system issues",
                    recommended_actions=[
                        "Encourage more diverse user feedback",
                        "Check feedback processing pipeline",
                        "Review collaborative learning algorithms"
                    ],
                    timestamp=time.time(),
                    confidence=0.6
                ))

        except Exception as e:
            logger.warning(f"Error detecting feedback anomalies: {str(e)}")

        return anomalies

    def analyze_feedback_impact(self, learner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the impact of user feedback on learning outcomes.

        Args:
            learner_id: Specific learner to analyze (None for system-wide)

        Returns:
            Analysis of feedback impact on performance and learning
        """
        try:
            impact_analysis = {
                'feedback_driven_improvements': [],
                'correction_effectiveness': {},
                'preference_alignment_score': 0.0,
                'collaborative_learning_benefits': {},
                'feedback_engagement_metrics': {},
                'recommendations': []
            }

            if not self.feedback_service:
                return impact_analysis

            # Analyze correction effectiveness
            corrections = self._analyze_correction_effectiveness(learner_id)
            impact_analysis['correction_effectiveness'] = corrections

            # Analyze preference alignment
            if learner_id:
                preferences = self.feedback_service.get_user_preferences(
                    type('Request', (), {'user_id': learner_id, 'include_history': False})()
                ).preferences

                alignment_score = self._calculate_preference_alignment_impact(preferences)
                impact_analysis['preference_alignment_score'] = alignment_score

            # Analyze collaborative learning benefits
            collaborative_benefits = self._analyze_collaborative_learning_benefits()
            impact_analysis['collaborative_learning_benefits'] = collaborative_benefits

            # Get feedback engagement metrics
            feedback_metrics = self.feedback_service.get_metrics()
            impact_analysis['feedback_engagement_metrics'] = {
                'total_feedbacks': feedback_metrics.total_feedbacks,
                'user_engagement_rate': feedback_metrics.user_engagement_rate,
                'correction_adoption_rate': feedback_metrics.correction_adoption_rate,
                'average_rating': feedback_metrics.average_rating
            }

            # Generate recommendations based on analysis
            recommendations = self._generate_feedback_based_recommendations(impact_analysis)
            impact_analysis['recommendations'] = recommendations

            logger.info(f"Completed feedback impact analysis for learner {learner_id or 'system'}")

            return impact_analysis

        except Exception as e:
            logger.error(f"Failed to analyze feedback impact: {str(e)}")
            return {'error': str(e)}

    def _analyze_correction_effectiveness(self, learner_id: Optional[str]) -> Dict[str, Any]:
        """Analyze how effective user corrections have been."""
        effectiveness = {
            'total_corrections': 0,
            'adoption_rate': 0.0,
            'improvement_rate': 0.0,
            'correction_types_effectiveness': {}
        }

        try:
            corrections = self.feedback_service.get_corrections(learner_id)
            if corrections:
                effectiveness['total_corrections'] = len(corrections)

                adopted_corrections = [c for c in corrections if c.get('adopted', False)]
                effectiveness['adoption_rate'] = len(adopted_corrections) / len(corrections) if corrections else 0.0

                improvement_scores = [c.get('improvement_score', 0) for c in adopted_corrections]
                effectiveness['improvement_rate'] = sum(improvement_scores) / len(improvement_scores) if improvement_scores else 0.0

                correction_types = defaultdict(list)
                for c in corrections:
                    ctype = c.get('correction_type', 'unknown')
                    correction_types[ctype].append(c.get('effectiveness', 0))

                for ctype, effs in correction_types.items():
                    effectiveness['correction_types_effectiveness'][ctype] = sum(effs) / len(effs) if effs else 0.0
            else:
                # No corrections available
                pass
        except Exception as e:
            logger.warning(f"Failed to analyze correction effectiveness: {str(e)}")

        return effectiveness

    def _calculate_preference_alignment_impact(self, preferences) -> float:
        """Calculate the impact of preference alignment on learning."""
        try:
            # Analyze how well preferences are being honored
            alignment_score = 0.0

            if hasattr(preferences, 'feedback_history') and preferences.feedback_history:
                # Higher alignment if user has provided feedback consistently
                consistency_score = min(len(preferences.feedback_history) / 10.0, 1.0)
                alignment_score += consistency_score * 0.6

            if hasattr(preferences, 'preference_weights') and preferences.preference_weights:
                # Alignment based on how well preferences are learned
                weight_diversity = len(preferences.preference_weights) / 5.0  # Max 5 categories
                alignment_score += min(weight_diversity, 1.0) * 0.4

            return alignment_score

        except Exception as e:
            logger.warning(f"Failed to calculate preference alignment impact: {str(e)}")
            return 0.0

    def _analyze_collaborative_learning_benefits(self) -> Dict[str, Any]:
        """Analyze benefits from collaborative learning patterns."""
        benefits = {
            'patterns_discovered': 0,
            'cross_user_learning': 0.0,
            'knowledge_sharing_efficiency': 0.0
        }

        try:
            collaborative_insights = self.feedback_service.get_collaborative_insights()
            benefits['patterns_discovered'] = len(collaborative_insights)

            if collaborative_insights:
                avg_confidence = sum(ci.confidence_score for ci in collaborative_insights) / len(collaborative_insights)
                benefits['cross_user_learning'] = avg_confidence
                benefits['knowledge_sharing_efficiency'] = min(avg_confidence * 1.2, 1.0)

        except Exception as e:
            logger.warning(f"Failed to analyze collaborative learning benefits: {str(e)}")

        return benefits

    def _generate_feedback_based_recommendations(self, impact_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on feedback impact analysis."""
        recommendations = []

        try:
            # Analyze correction effectiveness
            correction_effectiveness = impact_analysis.get('correction_effectiveness', {})
            if correction_effectiveness.get('adoption_rate', 0) < 0.5:
                recommendations.append("Improve correction adoption by making corrections more visible and actionable")

            # Analyze preference alignment
            alignment_score = impact_analysis.get('preference_alignment_score', 0)
            if alignment_score < 0.6:
                recommendations.append("Enhance preference learning to better align with user needs")

            # Analyze engagement
            engagement = impact_analysis.get('feedback_engagement_metrics', {})
            if engagement.get('user_engagement_rate', 0) < 0.3:
                recommendations.append("Increase user engagement with feedback collection mechanisms")

            # Analyze collaborative benefits
            collaborative = impact_analysis.get('collaborative_learning_benefits', {})
            if collaborative.get('patterns_discovered', 0) < 3:
                recommendations.append("Encourage more diverse feedback to improve collaborative learning")

        except Exception as e:
            logger.warning(f"Failed to generate feedback recommendations: {str(e)}")

        return recommendations

    def _get_confidence_level(self, confidence_score: float) -> PredictionConfidence:
        """Convert confidence score to level."""
        if confidence_score > 0.8:
            return PredictionConfidence.HIGH
        elif confidence_score > 0.6:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
