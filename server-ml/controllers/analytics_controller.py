"""
Analytics Controller for LLM Learning System

Handles API endpoints for advanced analytics and monitoring including:
- Analytics dashboard
- Skill gap analysis
- Bottleneck detection
- Performance prediction
- Learning insights
- System health monitoring
"""

import time
import logging
from typing import Optional

from models.analytics import (
    AnalyticsDashboard, AnalyticsDashboardRequest, AnalyticsDashboardResponse,
    SkillGapAnalysis, SkillGapAnalysisRequest, SkillGapAnalysisResponse,
    BottleneckDetection, BottleneckDetectionRequest, BottleneckDetectionResponse,
    PerformancePrediction, PerformancePredictionRequest, PerformancePredictionResponse,
    LearningInsights, LearningInsightsRequest, LearningInsightsResponse,
    SystemHealthMetrics, SystemHealthRequest, SystemHealthResponse
)
from services.analytics_service import AnalyticsService
from utils.logging_config import get_logger, log_system_health

logger = get_logger(__name__)


class AnalyticsController:
    """
    Controller for analytics and monitoring endpoints.

    Provides comprehensive analytics capabilities for the learning system,
    including real-time dashboards, predictive analytics, and system monitoring.
    """

    def __init__(self, analytics_service: Optional[AnalyticsService] = None):
        """
        Initialize analytics controller.

        Args:
            analytics_service: Analytics service instance (will create if None)
        """
        self.analytics_service = analytics_service or AnalyticsService()

        # Set service dependencies if available
        self._initialize_service_dependencies()

        logger.info("AnalyticsController initialized")

    def _initialize_service_dependencies(self):
        """Initialize service dependencies for analytics."""
        try:
            # Import services dynamically to avoid circular imports
            from services.reward_service import RewardService
            from services.hallucination_service import HallucinationService
            from services.meta_learning_service import MetaLearningService
            from services.external_llm_service import ExternalLLMService
            from services.feedback_service import FeedbackService
            from controllers.prompt_controller import PromptController

            # Try to get service instances from the global app context
            # This is a simplified approach - in production, use dependency injection
            try:
                # These would typically be injected or retrieved from a service container
                reward_service = RewardService()
                hallucination_service = HallucinationService()
                feedback_service = FeedbackService()
                meta_learning_service = MetaLearningService(reward_service, hallucination_service, None, feedback_service)
                external_llm_service = ExternalLLMService()

                # Set dependencies in analytics service
                self.analytics_service.set_dependencies(
                    reward_service=reward_service,
                    hallucination_service=hallucination_service,
                    meta_learning_service=meta_learning_service,
                    external_llm_service=external_llm_service,
                    feedback_service=feedback_service
                )

                logger.info("Analytics service dependencies initialized")

            except Exception as e:
                logger.warning(f"Could not initialize all service dependencies: {str(e)}")

        except ImportError as e:
            logger.warning(f"Could not import services: {str(e)}")

    def get_analytics_dashboard(self, request: AnalyticsDashboardRequest) -> AnalyticsDashboardResponse:
        """
        Get comprehensive analytics dashboard data.

        Args:
            request: Dashboard request parameters

        Returns:
            Analytics dashboard response with comprehensive data
        """
        start_time = time.time()

        try:
            logger.info(f"[Analytics] Generating dashboard for learner: {request.learner_id}")

            # Get dashboard data from analytics service
            dashboard = self.analytics_service.get_analytics_dashboard(
                learner_id=request.learner_id,
                include_historical=request.include_historical
            )

            processing_time = time.time() - start_time

            response = AnalyticsDashboardResponse(
                dashboard=dashboard,
                processing_time=processing_time,
                data_quality_score=self._calculate_data_quality_score(dashboard)
            )

            logger.info(f"[Analytics] Dashboard generated in {processing_time:.2f}s with quality score {response.data_quality_score:.2f}")

            return response

        except Exception as e:
            logger.error(f"[Analytics] Failed to generate dashboard: {str(e)}")
            # Return minimal response on error
            processing_time = time.time() - start_time
            empty_dashboard = AnalyticsDashboard(
                timestamp=time.time(),
                learner_id=request.learner_id,
                overall_learning_score=0.0,
                key_indicators={'error': 'Dashboard generation failed'},
                alerts=[{'type': 'error', 'message': str(e), 'severity': 'high'}]
            )

            return AnalyticsDashboardResponse(
                dashboard=empty_dashboard,
                processing_time=processing_time,
                data_quality_score=0.0
            )

    def analyze_skill_gaps(self, request: SkillGapAnalysisRequest) -> SkillGapAnalysisResponse:
        """
        Analyze skill gaps for a learner.

        Args:
            request: Skill gap analysis request

        Returns:
            Skill gap analysis response
        """
        start_time = time.time()

        try:
            logger.info(f"[Analytics] Analyzing skill gaps for learner: {request.learner_id}")

            # Perform skill gap analysis
            analysis = self.analytics_service.analyze_skill_gaps(request.learner_id)

            processing_time = time.time() - start_time

            response = SkillGapAnalysisResponse(
                analysis=analysis,
                processing_time=processing_time
            )

            logger.info(f"[Analytics] Skill gap analysis completed in {processing_time:.2f}s, "
                       f"found {len(analysis.skill_gaps) if analysis else 0} gaps")

            return response

        except Exception as e:
            logger.error(f"[Analytics] Failed to analyze skill gaps: {str(e)}")
            processing_time = time.time() - start_time

            # Return empty analysis on error
            empty_analysis = SkillGapAnalysis(
                learner_id=request.learner_id,
                timestamp=time.time(),
                skill_gaps=[],
                overall_gap_score=0.0,
                critical_gaps_count=0,
                recommendations=["Analysis failed - please try again"]
            )

            return SkillGapAnalysisResponse(
                analysis=empty_analysis,
                processing_time=processing_time
            )

    def detect_bottlenecks(self, request: BottleneckDetectionRequest) -> BottleneckDetectionResponse:
        """
        Detect learning bottlenecks in the system.

        Args:
            request: Bottleneck detection request

        Returns:
            Bottleneck detection response
        """
        start_time = time.time()

        try:
            logger.info("[Analytics] Detecting learning bottlenecks")

            # Perform bottleneck detection
            analysis = self.analytics_service.detect_bottlenecks(
                learner_id=request.learner_id if hasattr(request, 'learner_id') else None
            )

            processing_time = time.time() - start_time

            response = BottleneckDetectionResponse(
                analysis=analysis,
                processing_time=processing_time
            )

            logger.info(f"[Analytics] Bottleneck detection completed in {processing_time:.2f}s, "
                       f"found {len(analysis.bottlenecks) if analysis else 0} bottlenecks")

            return response

        except Exception as e:
            logger.error(f"[Analytics] Failed to detect bottlenecks: {str(e)}")
            processing_time = time.time() - start_time

            # Return empty analysis on error
            empty_analysis = BottleneckDetection(
                timestamp=time.time(),
                bottlenecks=[],
                overall_bottleneck_score=0.0,
                critical_bottlenecks_count=0,
                system_health_score=1.0,
                recommendations=["Detection failed - please try again"]
            )

            return BottleneckDetectionResponse(
                analysis=empty_analysis,
                processing_time=processing_time
            )

    def predict_performance(self, request: PerformancePredictionRequest) -> PerformancePredictionResponse:
        """
        Predict future learning performance.

        Args:
            request: Performance prediction request

        Returns:
            Performance prediction response
        """
        start_time = time.time()

        try:
            logger.info(f"[Analytics] Predicting performance for learner: {request.learner_id}, "
                       f"horizon: {request.prediction_horizon_days} days")

            # Generate performance prediction
            prediction = self.analytics_service.predict_performance(
                learner_id=request.learner_id,
                horizon_days=request.prediction_horizon_days
            )

            processing_time = time.time() - start_time

            response = PerformancePredictionResponse(
                prediction=prediction,
                processing_time=processing_time
            )

            confidence_level = prediction.confidence_level if prediction else "unknown"
            logger.info(f"[Analytics] Performance prediction completed in {processing_time:.2f}s "
                       f"with {confidence_level} confidence")

            return response

        except Exception as e:
            logger.error(f"[Analytics] Failed to predict performance: {str(e)}")
            processing_time = time.time() - start_time

            # Return null prediction on error
            return PerformancePredictionResponse(
                prediction=None,
                processing_time=processing_time
            )

    def generate_learning_insights(self, request: LearningInsightsRequest) -> LearningInsightsResponse:
        """
        Generate actionable learning insights.

        Args:
            request: Learning insights request

        Returns:
            Learning insights response
        """
        start_time = time.time()

        try:
            logger.info(f"[Analytics] Generating learning insights for learner: {request.learner_id}")

            # Generate learning insights
            insights = self.analytics_service.generate_learning_insights(request.learner_id)

            processing_time = time.time() - start_time

            response = LearningInsightsResponse(
                insights=insights,
                processing_time=processing_time
            )

            insight_count = len(insights.insights) if insights else 0
            logger.info(f"[Analytics] Learning insights generated in {processing_time:.2f}s, "
                       f"found {insight_count} insights")

            return response

        except Exception as e:
            logger.error(f"[Analytics] Failed to generate learning insights: {str(e)}")
            processing_time = time.time() - start_time

            # Return empty insights on error
            empty_insights = LearningInsights(
                timestamp=time.time(),
                insights=[],
                overall_insight_score=0.0,
                high_priority_count=0,
                categories_covered=[],
                implementation_roadmap=[],
                insights_metadata={'error': str(e)}
            )

            return LearningInsightsResponse(
                insights=empty_insights,
                processing_time=processing_time
            )

    def monitor_system_health(self, request: SystemHealthRequest) -> SystemHealthResponse:
        """
        Monitor system health and performance.

        Args:
            request: System health request

        Returns:
            System health response
        """
        start_time = time.time()

        try:
            logger.info("🏥 Monitoring system health and performance")

            # Get system health metrics
            health = self.analytics_service.monitor_system_health()

            processing_time = time.time() - start_time

            response = SystemHealthResponse(
                health=health,
                processing_time=processing_time
            )

            health_score = health.overall_health_score if health else 0.0

            # Determine health status for logging
            if health_score >= 0.8:
                health_status = "healthy"
            elif health_score >= 0.6:
                health_status = "degraded"
            else:
                health_status = "critical"

            # Log system health with detailed metrics
            health_metrics = {
                'overall_score': health_score,
                'processing_time': processing_time,
                'component_count': len(health.component_health) if health else 0,
                'alert_count': len(health.alerts) if health else 0,
                'error_rate': health.error_rates.get('system', 0.0) if health else 1.0
            }

            log_system_health("analytics_system", health_status, health_metrics)

            logger.info(f"✅ System health check completed in {processing_time:.2f}s, "
                       f"overall health: {health_score:.2f} ({health_status})")

            return response

        except Exception as e:
            logger.error(f"❌ Failed to monitor system health: {str(e)}")
            processing_time = time.time() - start_time

            # Log critical health status
            log_system_health("analytics_system", "critical", {
                'error': str(e),
                'processing_time': processing_time
            })

            # Return minimal health data on error
            minimal_health = SystemHealthMetrics(
                timestamp=time.time(),
                overall_health_score=0.0,
                component_health={},
                performance_metrics={},
                error_rates={'system': 1.0},
                resource_utilization={},
                alerts=[{'type': 'error', 'message': 'Health monitoring failed', 'severity': 'high'}]
            )

            return SystemHealthResponse(
                health=minimal_health,
                processing_time=processing_time
            )

    def get_analytics_status(self) -> dict:
        """
        Get analytics system status and configuration.

        Returns:
            Status information about the analytics system
        """
        try:
            status = {
                'service_status': 'operational',
                'timestamp': time.time(),
                'config': {
                    'enabled': self.analytics_service.config.enabled,
                    'update_interval': self.analytics_service.config.update_interval_seconds,
                    'monitoring_components': self.analytics_service.config.monitoring_components
                },
                'capabilities': [
                    'skill_gap_analysis',
                    'bottleneck_detection',
                    'performance_prediction',
                    'learning_insights',
                    'system_health_monitoring',
                    'trend_analysis',
                    'anomaly_detection'
                ],
                'data_quality': self._assess_data_quality()
            }

            return status

        except Exception as e:
            logger.error(f"[Analytics] Failed to get analytics status: {str(e)}")
            return {
                'service_status': 'error',
                'timestamp': time.time(),
                'error': str(e)
            }

    def _calculate_data_quality_score(self, dashboard: AnalyticsDashboard) -> float:
        """
        Calculate data quality score for dashboard.

        Args:
            dashboard: Analytics dashboard

        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            quality_factors = []

            # Check if core components are present
            if dashboard.skill_gap_analysis:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)

            if dashboard.bottleneck_detection:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)

            if dashboard.learning_insights:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)

            if dashboard.system_health:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.0)

            # Check data freshness (within last hour)
            current_time = time.time()
            freshness_scores = []

            for component_time in dashboard.data_freshness.values():
                if component_time:
                    hours_old = (current_time - component_time) / 3600
                    freshness = max(0.0, 1.0 - (hours_old / 24))  # Degrade over 24 hours
                    freshness_scores.append(freshness)

            if freshness_scores:
                avg_freshness = sum(freshness_scores) / len(freshness_scores)
                quality_factors.append(avg_freshness)

            # Overall quality
            if quality_factors:
                return sum(quality_factors) / len(quality_factors)
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate data quality score: {str(e)}")
            return 0.5

    def _assess_data_quality(self) -> dict:
        """
        Assess overall data quality for analytics.

        Returns:
            Data quality assessment
        """
        try:
            # Check data availability and recency
            quality_metrics = {
                'data_availability': 0.8,  # Placeholder
                'data_recency': 0.9,       # Placeholder
                'data_completeness': 0.7,  # Placeholder
                'data_accuracy': 0.85,     # Placeholder
                'overall_quality': 0.81
            }

            return quality_metrics

        except Exception as e:
            logger.error(f"Failed to assess data quality: {str(e)}")
            return {'overall_quality': 0.0, 'error': str(e)}