"""
FastAPI Application for LLM Learning System - Educational Implementation

This is the main application file that demonstrates how to build a professional
REST API using FastAPI with proper MVC architecture. Think of this as the
"glue" that connects all your components together.

What you'll learn from this file:
1. How to structure a FastAPI application with MVC pattern
2. Proper dependency injection and component initialization
3. Request/response modeling with Pydantic
4. Error handling and logging in APIs
5. API documentation generation
6. Middleware setup for CORS, logging, etc.

MVC Architecture in this API:
- Models: Data structures and business logic (LLM, History, etc.)
- Views: Response formatting and visualization (JSON responses)
- Controllers: API endpoints that orchestrate operations

Design Patterns Used:
- Dependency Injection: Components are created and injected
- Repository Pattern: Data access is abstracted
- Service Layer: Business logic is encapsulated
- Factory Pattern: Object creation is centralized

Best Practices Demonstrated:
- Type hints for better code documentation
- Comprehensive error handling
- Logging for debugging and monitoring
- Configuration management
- API versioning
- Input validation
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import time
from dotenv import load_dotenv

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import our custom utilities first (educational order)
from utils.logging_config import (
    get_logger, log_api_request, log_performance, log_service_initialization,
    log_system_health, log_curriculum_status, log_external_llm_interaction,
    log_feedback_processing, log_persistence_operation, start_progress_tracker,
    update_progress, complete_progress, log_learning_event
)
from utils.config import get_api_config, get_llm_config, get_external_llm_config, config_manager

# Import MVC components
from models.learner import LLM
from models.prompt import Prompt
from models.history import History
from models.task_generator import TaskGenerator
from models.hallucination import (
    HallucinationCheckRequest,
    HallucinationCheckResponse,
    HallucinationConfig,
    HallucinationMetrics,
    HallucinationMetricsResponse
)
from models.persistence import (
    LearningStateType,
    SaveStateRequest,
    SaveStateResponse,
    LoadStateRequest,
    LoadStateResponse,
    ListVersionsRequest,
    ListVersionsResponse,
    ExportStateRequest,
    ExportStateResponse,
    ImportStateRequest,
    ImportStateResponse,
    RollbackRequest,
    RollbackResponse,
    BackupRequest,
    BackupResponse,
    RestoreRequest,
    RestoreResponse,
    DistributedStatusResponse,
    SyncRequest,
    SyncResponse
)
from models.feedback import (
    SubmitFeedbackRequest,
    SubmitCorrectionRequest,
    RateResponseRequest,
    SubmitErrorReportRequest
)
from controllers.evaluator import Evaluator
from controllers.scheduler import Scheduler
from controllers.prompt_controller import PromptController
from controllers.analytics_controller import AnalyticsController
from controllers.feedback_controller import FeedbackController
from services.visualization_service import VisualizationService
from services.learning_loop_service import LearningLoopService
from services.reward_service import RewardService
from services.hallucination_service import HallucinationService
from services.meta_learning_service import MetaLearningService
from services.external_llm_service import ExternalLLMService
from services.analytics_service import AnalyticsService
from services.persistence_service import PersistenceService
from services.feedback_service import FeedbackService
from controllers.persistence_controller import PersistenceController

# Set up logging for this module
logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# Load configuration
api_config = get_api_config()
llm_config = get_llm_config()

# Create FastAPI application with configuration
app = FastAPI(
    title=api_config.title,
    description=api_config.description,
    version=api_config.version,
    debug=api_config.debug
)

# Add CORS middleware for web frontend integration
# Why CORS? Web browsers block cross-origin requests for security
# This allows your React/Vue frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Initialize services first (required for MVC components)
# Why initialize services first? MVC components depend on services
logger.info("🔧 Initializing core services...")

# Basic services with no dependencies
log_service_initialization("HallucinationService", "starting")
hallucination_service = HallucinationService()
log_service_initialization("HallucinationService", "success")

log_service_initialization("FeedbackService", "starting")
feedback_service = FeedbackService()
log_service_initialization("FeedbackService", "success")

log_service_initialization("AnalyticsService", "starting")
analytics_service = AnalyticsService()
log_service_initialization("AnalyticsService", "success")

log_service_initialization("PersistenceService", "starting")
persistence_service = PersistenceService()
log_service_initialization("PersistenceService", "success")

log_service_initialization("VisualizationService", "starting")
visualization_service = VisualizationService()
log_service_initialization("VisualizationService", "success")

# Services with dependencies
log_service_initialization("RewardService", "starting", {"dependencies": ["HallucinationService", "FeedbackService"]})
reward_service = RewardService(hallucination_service, feedback_service)
log_service_initialization("RewardService", "success")

# External LLM service (conditional)
external_llm_service = None
if get_external_llm_config().enabled:
    log_service_initialization("ExternalLLMService", "starting")
    try:
        external_llm_service = ExternalLLMService()
        log_service_initialization("ExternalLLMService", "success", {"config": get_external_llm_config().enabled_providers})
    except Exception as e:
        log_service_initialization("ExternalLLMService", "failed", {"error": str(e)})
        external_llm_service = None
else:
    log_service_initialization("ExternalLLMService", "warning", {"reason": "External LLM service disabled in config"})

# Initialize MVC components with configuration
# Why initialize here? This demonstrates dependency injection at the application level
# In production, you'd use a DI container, but this shows the concept
logger.info("🏗️  Initializing MVC components...")

# Model layer - Data and business logic
log_service_initialization("History", "starting")
history = History()
log_service_initialization("History", "success")

log_service_initialization("TaskGenerator", "starting", {"num_actions": llm_config.num_actions})
task_generator = TaskGenerator(num_actions=llm_config.num_actions)
log_service_initialization("TaskGenerator", "success")

log_service_initialization("Evaluator", "starting", {"dependencies": ["History"]})
evaluator = Evaluator(history)
log_service_initialization("Evaluator", "success")

# Scheduler depends on feedback_service
log_service_initialization("Scheduler", "starting", {"dependencies": ["TaskGenerator", "FeedbackService"]})
scheduler = Scheduler(task_generator, feedback_service=feedback_service)
log_service_initialization("Scheduler", "success")

# Meta-learning service depends on reward_service, hallucination_service, scheduler, feedback_service
log_service_initialization("MetaLearningService", "starting", {
    "dependencies": ["RewardService", "HallucinationService", "Scheduler", "FeedbackService"]
})
meta_learning_service = MetaLearningService(reward_service, hallucination_service, scheduler, feedback_service)
log_service_initialization("MetaLearningService", "success")

# LLM depends on meta_learning_service
log_service_initialization("LLM", "starting", {
    "config": {
        "num_actions": llm_config.num_actions,
        "alpha": llm_config.alpha,
        "epsilon": llm_config.epsilon
    },
    "dependencies": ["MetaLearningService"]
})
llm = LLM(
    num_actions=llm_config.num_actions,
    alpha=llm_config.alpha,
    epsilon=llm_config.epsilon,
    meta_learning_service=meta_learning_service
)
log_service_initialization("LLM", "success")

# Learning loop service depends on prompt_controller, evaluator, scheduler, task_generator, feedback_service, reward_service, meta_learning_service
# Initialize learning loop service first (before prompt_controller)
log_service_initialization("LearningLoopService", "starting", {
    "dependencies": ["Evaluator", "Scheduler", "TaskGenerator", "FeedbackService", "RewardService", "MetaLearningService"]
})
learning_loop_service = LearningLoopService(None, evaluator, scheduler, task_generator, feedback_service, reward_service, meta_learning_service)  # prompt_controller will be set later
log_service_initialization("LearningLoopService", "success")

# Controller layer - Handle interactions between models and views
log_service_initialization("PromptController", "starting", {
    "dependencies": ["LLM", "History", "RewardService", "HallucinationService", "ExternalLLMService", "LearningLoopService"]
})
prompt_controller = PromptController(llm, history, reward_service, hallucination_service, external_llm_service, learning_loop_service)
# Add meta-learning service reference to prompt controller for integration
prompt_controller.meta_learning_service = meta_learning_service
log_service_initialization("PromptController", "success")

# Now set the prompt_controller reference in learning_loop_service
learning_loop_service.prompt_controller = prompt_controller

log_service_initialization("AnalyticsController", "starting")
analytics_controller = AnalyticsController()
log_service_initialization("AnalyticsController", "success")

log_service_initialization("FeedbackController", "starting", {"dependencies": ["FeedbackService"]})
feedback_controller = FeedbackController(feedback_service)
log_service_initialization("FeedbackController", "success")

# Component registry for persistence operations
component_registry = {
    'llm': llm,
    'history': history,
    'scheduler': scheduler,
    'reward_service': reward_service,
    'meta_learning_service': meta_learning_service,
    'analytics_service': analytics_service,
    'feedback_service': feedback_service
}

log_service_initialization("PersistenceController", "starting", {
    "dependencies": ["PersistenceService"],
    "components": list(component_registry.keys())
})
persistence_controller = PersistenceController(persistence_service, component_registry)
log_service_initialization("PersistenceController", "success")

# DO NOT start autonomous learning loop automatically
# Wait for initial user prompt from UI
log_service_initialization("AutonomousLearningLoop", "info", {
    "message": "⏳ Autonomous learning loop will start after receiving initial user prompt from UI",
    "status": "waiting_for_user_input"
})
logger.info("💡 System is ready. Please submit an initial prompt through the UI to begin autonomous learning.")

# Optionally uncomment below to start immediately (not recommended for user-driven mode):
# try:
#     learning_loop_service.start_learning_loop()
#     log_service_initialization("AutonomousLearningLoop", "success", {"message": "Autonomous learning loop started automatically"})
# except Exception as e:
#     log_service_initialization("AutonomousLearningLoop", "failed", {"error": str(e)})

logger.info("✅ All MVC components initialized successfully")


# Middleware for logging API requests
# This demonstrates how to add custom middleware in FastAPI
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Enhanced middleware to log all API requests with detailed performance metrics.

    This teaches you:
    1. How middleware works in FastAPI
    2. Request/response logging for monitoring
    3. Performance measurement and analysis
    4. Error tracking in APIs
    5. Structured logging for API operations

    Why middleware? It runs for every request without changing endpoint code
    """
    start_time = time.time()
    request_id = f"{request.method}_{request.url.path}_{start_time}"

    # Log the incoming request with enhanced details
    logger.info(f"🌐 API Request: {request.method} {request.url.path} | "
               f"Client: {request.client.host if request.client else 'unknown'} | "
               f"ID: {request_id}")

    # Log request headers for debugging (excluding sensitive ones)
    safe_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ['authorization', 'cookie', 'x-api-key']}
    if safe_headers:
        logger.debug(f"📋 Request Headers: {safe_headers}")

    try:
        # Process the request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Enhanced performance logging
        performance_data = {
            'request_id': request_id,
            'method': request.method,
            'endpoint': request.url.path,
            'status_code': response.status_code,
            'duration': process_time,
            'response_size': response.headers.get('content-length', 'unknown')
        }

        # Log performance metrics
        log_performance(f"api_{request.method.lower()}_{request.url.path.replace('/', '_')}",
                       process_time, performance_data)

        # Enhanced API logging with performance indicators
        if process_time > 1.0:  # Slow request warning
            logger.warning(f"🐌 Slow API Response: {request.method} {request.url.path} -> "
                          f"{response.status_code} ({process_time:.3f}s)")
        elif response.status_code >= 400:
            logger.warning(f"⚠️  API Error: {request.method} {request.url.path} -> "
                          f"{response.status_code} ({process_time:.3f}s)")
        else:
            logger.info(f"✅ API Response: {request.method} {request.url.path} -> "
                       f"{response.status_code} ({process_time:.3f}s)")

        # Add processing time to response headers for debugging
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        # Log failed requests with enhanced error details
        process_time = time.time() - start_time
        error_details = {
            'request_id': request_id,
            'method': request.method,
            'endpoint': request.url.path,
            'error_type': type(e).__name__,
            'error_message': str(e),
            'duration': process_time
        }

        logger.error(f"❌ API Request Failed: {request.method} {request.url.path} | "
                    f"Error: {type(e).__name__}: {str(e)} | Duration: {process_time:.3f}s")

        # Log performance for failed requests too
        log_performance(f"api_failed_{request.method.lower()}_{request.url.path.replace('/', '_')}",
                       process_time, error_details)

        # Re-raise the exception
        raise

# Pydantic models for request/response
class PromptRequest(BaseModel):
    prompt_text: str
    use_external_llm: Optional[bool] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class PromptResponse(BaseModel):
    response: str
    action: int
    metrics: Dict[str, Any]
    external_llm_used: Optional[bool] = None
    external_llm_info: Optional[Dict[str, Any]] = None

class MetricsResponse(BaseModel):
    success_rate: float
    pattern_frequency: Dict[str, int]
    total_interactions: int

class EvaluateRequest(BaseModel):
    prompt_text: Optional[str] = None
    num_episodes: int = 100

class EvaluateResponse(BaseModel):
    average_reward: float

class ScheduleRequest(BaseModel):
    num_tasks: int
    prompt_text: Optional[str] = None

class ScheduleResponse(BaseModel):
    tasks: List[List[float]]

class ChartDataResponse(BaseModel):
    chart_data: List[Dict[str, Any]]  # Array of data points for Recharts

class PromptHistoryItem(BaseModel):
    prompt_text: str
    response: str
    reward: Optional[float]
    action: Optional[int]
    timestamp: float
    source: str

class PromptHistoryResponse(BaseModel):
    interactions: List[PromptHistoryItem]
    total_count: int

class RewardWeightsConfig(BaseModel):
    accuracy: float
    coherence: float
    factuality: float
    creativity: float

class RewardWeightsResponse(BaseModel):
    success: bool
    message: str
    current_weights: Optional[RewardWeightsConfig] = None

class MultiObjectiveRewards(BaseModel):
    accuracy: float
    coherence: float
    factuality: float
    creativity: float

class RewardMetrics(BaseModel):
    individual_rewards: MultiObjectiveRewards
    total_reward: float
    weighted_reward: float
    weights_used: RewardWeightsConfig

class RewardMetricsResponse(BaseModel):
    current_metrics: RewardMetrics
    reward_weights: RewardWeightsConfig
    history_length: int
    average_metrics: Optional[Dict[str, float]] = None

class EvaluateMultiRequest(BaseModel):
    prompt_text: Optional[str] = None
    num_episodes: int = 100

class EvaluateMultiResponse(BaseModel):
    average_weighted_reward: float
    average_individual_rewards: MultiObjectiveRewards
    total_episodes: int

class CurriculumSkillsResponse(BaseModel):
    skills: Dict[str, Any]
    categories: Dict[str, List[str]]
    difficulty_levels: Dict[str, List[str]]

class CurriculumProgressResponse(BaseModel):
    learner_id: str
    progress: Dict[str, Any]
    available_skills: List[str]
    recommendations: Dict[str, Any]
    curriculum_summary: Dict[str, Any]

class CurriculumAdjustRequest(BaseModel):
    learner_id: str
    target_difficulty: Optional[str] = None  # "easy", "medium", "hard", "expert"
    skill_focus: Optional[Dict[str, float]] = None  # skill_id -> weight

class CurriculumResetRequest(BaseModel):
    learner_id: str
    confirm_reset: bool = False

# Meta-learning API models
class MetaLearningStatusResponse(BaseModel):
    current_strategy: str
    current_params: Dict[str, Any]
    strategy_history: List[str]
    active_rules: List[Dict[str, Any]]
    performance_summary: Dict[str, Any]
    recommendations: List[Dict[str, Any]]

class MetaPerformanceMetricsResponse(BaseModel):
    current_strategy: str
    current_params: Dict[str, Any]
    strategy_performance: Dict[str, Dict[str, Any]]
    recent_performance: List[Dict[str, Any]]
    adaptation_history: List[Dict[str, Any]]
    context_awareness: Dict[str, Any]
    strategy_effectiveness: Dict[str, float]
    learning_transfer_memory: Dict[str, Any]

class MetaAdaptationTriggerResponse(BaseModel):
    success: bool
    message: str
    new_strategy: Optional[str] = None
    parameter_changes: Optional[Dict[str, Any]] = None

class MetaStrategiesResponse(BaseModel):
    strategies: List[Dict[str, Any]]

class MetaStrategySwitchRequest(BaseModel):
    strategy_name: str

class MetaStrategySwitchResponse(BaseModel):
    success: bool
    message: str
    old_strategy: Optional[str] = None
    new_strategy: Optional[str] = None
    available_strategies: Optional[List[str]] = None

# External LLM API models
class LLMSwitchRequest(BaseModel):
    use_external: bool

class LLMSwitchResponse(BaseModel):
    success: bool
    message: str
    current_mode: str

class LLMModelsResponse(BaseModel):
    models: Dict[str, Any]
    total_count: int

class LLMCostsResponse(BaseModel):
    costs: Dict[str, Any]
    total_providers: int

class LLMCompareRequest(BaseModel):
    prompt_text: str
    providers_models: List[Dict[str, str]]

class LLMCompareResponse(BaseModel):
    comparisons: Dict[str, Any]
    prompt_text: str
    total_models: int

class LLMStatusResponse(BaseModel):
    status: Dict[str, Any]

# Routes

@app.post("/prompt/handle", response_model=PromptResponse)
async def handle_prompt(request: PromptRequest, user_id: Optional[str] = None):
    start_time = time.time()
    prompt_id = f"prompt_{start_time}_{hash(request.prompt_text) % 10000}"

    try:
        logger.info(f"🤖 Handling prompt request: {request.prompt_text[:50]}... | ID: {prompt_id}")

        # Log learning progress before processing
        current_metrics = prompt_controller.get_learning_metrics()
        log_learning_event("prompt_processing_started", {
            'prompt_id': prompt_id,
            'user_id': user_id,
            'prompt_length': len(request.prompt_text),
            'use_external_llm': request.use_external_llm,
            'current_interactions': current_metrics.get('total_interactions', 0),
            'success_rate': current_metrics.get('success_rate', 0.0)
        })

        result = prompt_controller.handle_prompt(
            request.prompt_text,
            use_external_llm=request.use_external_llm,
            provider=request.provider,
            model=request.model,
            user_id=user_id
        )

        # 🚀 IMPORTANT: Trigger autonomous learning loop with initial prompt
        if learning_loop_service.waiting_for_initial_prompt and not learning_loop_service.initial_prompt_received:
            logger.info("🎯 First user prompt detected! Triggering autonomous learning loop...")
            learning_loop_service.receive_initial_prompt(request.prompt_text, result, user_id)

        # result is now a dict with response, action, hallucination_analysis, reward_result
        metrics = prompt_controller.get_learning_metrics()

        # Include hallucination analysis in metrics if available
        hallucinated = False
        if result.get('hallucination_analysis'):
            metrics['hallucination_analysis'] = result['hallucination_analysis'].dict()
            hallucinated = result['hallucination_analysis'].is_hallucinated
            # Flag potentially hallucinated responses
            if hallucinated:
                metrics['hallucination_warning'] = f"Response may contain hallucinations (confidence: {result['hallucination_analysis'].overall_confidence:.2f})"

        # Include reward details if available
        if result.get('reward_result'):
            metrics['reward_details'] = result['reward_result']

        # Extract external LLM information and log interaction
        external_llm_info = None
        if result.get('external_llm_response') and not result['external_llm_response'].error:
            external_llm_info = {
                'provider': result['external_llm_response'].provider,
                'model': result['external_llm_response'].model,
                'cost': result['external_llm_response'].cost,
                'latency': result['external_llm_response'].latency,
                'input_tokens': result['external_llm_response'].input_tokens,
                'output_tokens': result['external_llm_response'].output_tokens
            }

            # Log external LLM interaction
            log_external_llm_interaction(
                provider=result['external_llm_response'].provider,
                model=result['external_llm_response'].model,
                operation='generate',
                cost=result['external_llm_response'].cost,
                latency=result['external_llm_response'].latency,
                tokens={
                    'input': result['external_llm_response'].input_tokens,
                    'output': result['external_llm_response'].output_tokens,
                    'total': result['external_llm_response'].input_tokens + result['external_llm_response'].output_tokens
                }
            )

        # Log learning progress update
        processing_time = time.time() - start_time
        log_learning_event("prompt_processing_completed", {
            'prompt_id': prompt_id,
            'user_id': user_id,
            'action': result['action'],
            'hallucinated': hallucinated,
            'processing_time': processing_time,
            'reward': result.get('reward_result', {}).get('total_reward') if result.get('reward_result') else None,
            'external_llm_used': result.get('external_llm_used', False),
            'new_interactions': metrics.get('total_interactions', 0)
        })

        # Enhanced success logging
        status_icon = "⚠️" if hallucinated else "✅"
        logger.info(f"{status_icon} Prompt handled successfully | Action: {result['action']} | "
                   f"Hallucinated: {hallucinated} | Time: {processing_time:.3f}s | "
                   f"Interactions: {metrics['total_interactions']}")

        return PromptResponse(
            response=result['response'],
            action=result['action'],
            metrics=metrics,
            external_llm_used=result.get('external_llm_used'),
            external_llm_info=external_llm_info
        )
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Failed to handle prompt: {str(e)} | ID: {prompt_id} | Time: {processing_time:.3f}s")

        # Log failed learning event
        log_learning_event("prompt_processing_failed", {
            'prompt_id': prompt_id,
            'user_id': user_id,
            'error': str(e),
            'processing_time': processing_time
        })

        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompt/metrics", response_model=MetricsResponse)
async def get_learning_metrics():
    try:
        logger.info("[ML API] Fetching learning metrics")
        metrics = prompt_controller.get_learning_metrics()
        logger.info(f"[ML API] Metrics fetched: success_rate={metrics['success_rate']:.3f}, total_interactions={metrics['total_interactions']}")
        return MetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"[ML API] Failed to fetch metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    try:
        prompt = Prompt(request.prompt_text) if request.prompt_text else None
        # For evaluation, we need a task. Using a dummy task or the first scheduled task
        tasks = scheduler.schedule(1, prompt)
        if not tasks:
            raise HTTPException(status_code=400, detail="No tasks available for evaluation")
        task = tasks[0]
        avg_reward = evaluator.evaluate(llm, task, prompt, request.num_episodes)
        return EvaluateResponse(average_reward=avg_reward)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/schedule", response_model=ScheduleResponse)
async def schedule_tasks(request: ScheduleRequest):
    try:
        prompt = Prompt(request.prompt_text) if request.prompt_text else None
        tasks = scheduler.schedule(request.num_tasks, prompt)
        return ScheduleResponse(tasks=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize/chartjs", response_model=ChartDataResponse)
async def get_chartjs_data():
    """
    Get Chart.js compatible data for web visualizations.

    This endpoint demonstrates:
    1. How to prepare data for frontend consumption
    2. JSON serialization of complex data structures
    3. Error handling in data processing
    4. Performance logging for expensive operations

    Educational Notes:
    - Web dashboards need JSON, not matplotlib plots
    - Data transformation is a key API responsibility
    - Consider caching for expensive computations
    """
    start_time = time.time()

    try:
        logger.info("Generating Chart.js data for visualization")

        # Get learning metrics from the controller
        metrics = prompt_controller.get_learning_metrics()

        # Get learning loop progress data
        loop_progress = learning_loop_service.get_learning_progress()

        # Check if there are any interactions - if not, return empty data for empty state
        has_interactions = metrics.get('total_interactions', 0) > 0
        if not has_interactions:
            logger.info("Empty state: No interactions available, returning empty chart data")
            return ChartDataResponse(chart_data=[])

        # Get historical rewards from history if available
        rewards = []
        if hasattr(history, 'get_all_interactions_chronological'):
            interactions = history.get_all_interactions_chronological()
            if interactions:
                rewards = [i.get('reward', 0) for i in interactions[-50:]]  # Last 50 rewards

        # If no historical rewards, check learning loop data
        if not rewards and loop_progress and 'recent_rewards' in loop_progress and loop_progress['recent_rewards']:
            rewards = loop_progress['recent_rewards']

        # Use the visualization service to prepare Chart.js data
        chart_data = visualization_service.get_chartjs_data(rewards, metrics, loop_progress)

        process_time = time.time() - start_time
        log_performance("chartjs_data_generation", process_time)

        logger.info("Chart.js data generated successfully")
        return ChartDataResponse(chart_data=chart_data)

    except Exception as e:
        logger.error(f"Failed to generate Chart.js data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization data generation failed: {str(e)}")

@app.get("/visualize/learning_progress")
async def get_learning_progress_data():
    """
    Get comprehensive learning progress data for analysis.

    This endpoint shows how to:
    1. Aggregate data from multiple sources
    2. Structure data for different frontend needs
    3. Handle missing or incomplete data gracefully
    4. Provide insights alongside raw data

    Educational Notes:
    - APIs should provide data, not just visualizations
    - Allow frontends flexibility in how they present data
    - Include metadata and insights when possible
    """
    start_time = time.time()

    try:
        logger.info("[ML API] Retrieving learning progress data")

        # Get current metrics from prompt controller
        metrics = prompt_controller.get_learning_metrics()

        # Get learning loop progress data
        loop_progress = learning_loop_service.get_learning_progress()

        # Generate insights using the visualization service
        # This shows how services can enhance controller data
        insights = visualization_service._generate_insights(loop_progress.get('recent_rewards', []), metrics)

        # Structure the response
        response_data = {
            "metrics": metrics,
            "learning_loop": loop_progress,
            "insights": insights,
            "timestamp": time.time(),
            "data_quality": "good" if metrics["total_interactions"] > 0 else "insufficient_data"
        }

        process_time = time.time() - start_time
        log_performance("learning_progress_retrieval", process_time)

        logger.info(f"[ML API] Learning progress data retrieved with {metrics['total_interactions']} interactions")
        return response_data

    except Exception as e:
        logger.error(f"[ML API] Failed to retrieve learning progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Learning progress retrieval failed: {str(e)}")

@app.post("/visualize/generate_report")
async def generate_learning_report():
    """
    Generate a comprehensive learning report with visualizations.

    This endpoint demonstrates:
    1. Complex data aggregation across the system
    2. Report generation as a service
    3. Structured data for comprehensive analysis
    4. Performance considerations for heavy computations

    Educational Notes:
    - Some operations are expensive and should be cached
    - Reports combine multiple data sources
    - Consider async processing for heavy reports
    """
    start_time = time.time()

    try:
        logger.info("Generating comprehensive learning report")

        # Get all available data
        metrics = prompt_controller.get_learning_metrics()

        # Sample reward data (in real app, this would be historical)
        sample_rewards = [0.1 * i for i in range(10)]

        # Generate comprehensive report
        report = visualization_service.create_comprehensive_report(sample_rewards, metrics)

        process_time = time.time() - start_time
        log_performance("comprehensive_report_generation", process_time)

        logger.info("Comprehensive learning report generated")
        return report

    except Exception as e:
        logger.error(f"Failed to generate learning report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# User prompt processing endpoint
@app.post("/prompt/process")
async def process_user_prompt(request: PromptRequest, user_id: Optional[str] = None):
    """
    Process a user-submitted prompt individually.

    This endpoint allows users to submit prompts directly for processing,
    rather than relying on autonomous prompt generation. Each prompt is
    processed immediately and the system learns from the interaction.
    """
    start_time = time.time()
    prompt_id = f"user_prompt_{start_time}_{hash(request.prompt_text) % 10000}"

    try:
        logger.info(f"🤖 Processing user prompt: {request.prompt_text[:50]}... | ID: {prompt_id}")

        # Log user prompt processing start
        log_learning_event("user_prompt_processing_started", {
            'prompt_id': prompt_id,
            'user_id': user_id,
            'prompt_length': len(request.prompt_text),
            'use_external_llm': request.use_external_llm,
            'current_interactions': prompt_controller.get_learning_metrics().get('total_interactions', 0)
        })

        # Process the user prompt
        result = prompt_controller.handle_prompt(
            request.prompt_text,
            source="user",
            use_external_llm=request.use_external_llm,
            provider=request.provider,
            model=request.model,
            user_id=user_id
        )

        # 🚀 IMPORTANT: Trigger autonomous learning loop with initial prompt
        if learning_loop_service.waiting_for_initial_prompt and not learning_loop_service.initial_prompt_received:
            logger.info("🎯 First user prompt detected! Triggering autonomous learning loop...")
            learning_loop_service.receive_initial_prompt(request.prompt_text, result, user_id)

        # Get updated metrics
        metrics = prompt_controller.get_learning_metrics()

        # Include hallucination analysis in metrics if available
        hallucinated = False
        if result.get('hallucination_analysis'):
            metrics['hallucination_analysis'] = result['hallucination_analysis'].dict()
            hallucinated = result['hallucination_analysis'].is_hallucinated

        # Include reward details if available
        if result.get('reward_result'):
            metrics['reward_details'] = result['reward_result']

        # Extract external LLM information
        external_llm_info = None
        if result.get('external_llm_response') and not result['external_llm_response'].error:
            external_llm_info = {
                'provider': result['external_llm_response'].provider,
                'model': result['external_llm_response'].model,
                'cost': result['external_llm_response'].cost,
                'latency': result['external_llm_response'].latency,
                'input_tokens': result['external_llm_response'].input_tokens,
                'output_tokens': result['external_llm_response'].output_tokens
            }

        # Log user prompt processing completion
        processing_time = time.time() - start_time
        log_learning_event("user_prompt_processing_completed", {
            'prompt_id': prompt_id,
            'user_id': user_id,
            'action': result['action'],
            'hallucinated': hallucinated,
            'processing_time': processing_time,
            'reward': result.get('reward_result', {}).get('weighted_reward') if result.get('reward_result') else None,
            'external_llm_used': result.get('external_llm_used', False),
            'new_interactions': metrics.get('total_interactions', 0)
        })

        # Enhanced success logging
        status_icon = "⚠️" if hallucinated else "✅"
        logger.info(f"{status_icon} User prompt processed successfully | Action: {result['action']} | "
                   f"Hallucinated: {hallucinated} | Time: {processing_time:.3f}s | "
                   f"Interactions: {metrics['total_interactions']}")

        return PromptResponse(
            response=result['response'],
            action=result['action'],
            metrics=metrics,
            external_llm_used=result.get('external_llm_used'),
            external_llm_info=external_llm_info
        )
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Failed to process user prompt: {str(e)} | ID: {prompt_id} | Time: {processing_time:.3f}s")

        # Log failed user prompt processing
        log_learning_event("user_prompt_processing_failed", {
            'prompt_id': prompt_id,
            'user_id': user_id,
            'error': str(e),
            'processing_time': processing_time
        })

        raise HTTPException(status_code=500, detail=str(e))

# Legacy learning loop endpoints (deprecated but kept for compatibility)
@app.post("/learning/start")
async def start_learning_loop():
    """
    DEPRECATED: This endpoint is no longer used in user-driven learning mode.

    The system now processes individual user prompts instead of running
    autonomous learning loops.
    """
    logger.warning("⚠️  Learning loop start requested but system is now user-driven")
    return {
        "message": "Learning loop is deprecated. Use /prompt/process for individual prompt processing.",
        "status": "deprecated",
        "note": "The system now operates in user-driven mode where prompts are submitted individually."
    }

@app.post("/learning/stop")
async def stop_learning_loop():
    """
    DEPRECATED: This endpoint is no longer used in user-driven learning mode.
    """
    logger.warning("⚠️  Learning loop stop requested but system is now user-driven")
    return {
        "message": "Learning loop is deprecated. No autonomous loop is running.",
        "status": "deprecated"
    }

@app.get("/learning/progress")
async def get_learning_progress():
    """
    Get learning progress data from user interactions and autonomous learning loop.

    Returns metrics based on both user-submitted prompts and autonomous learning loop iterations.
    """
    try:
        # Get metrics from user interactions
        metrics = prompt_controller.get_learning_metrics()

        # Get learning loop progress data
        loop_progress = learning_loop_service.get_learning_progress()

        progress_data = {
            'iterations': loop_progress.get('iterations', metrics.get('total_interactions', 0)),
            'current_avg_reward': loop_progress.get('current_avg_reward', metrics.get('success_rate', 0.0)),
            'recent_rewards': loop_progress.get('recent_rewards', []),
            'success_rate': loop_progress.get('success_rate', metrics.get('success_rate', 0.0)),
            'generated_prompts_count': loop_progress.get('generated_prompts_count', 0),
            'is_running': learning_loop_service.is_running,  # Check if autonomous loop is running
            'data_available': metrics.get('total_interactions', 0) > 0 or loop_progress.get('iterations', 0) > 0,
            'message': "Progress includes both user-submitted prompts and autonomous learning loop iterations."
        }

        return progress_data
    except Exception as e:
        logger.error(f"Failed to get learning progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning progress: {str(e)}")

@app.get("/learning/config")
async def get_learning_config():
    """
    Get LLM and strategy configuration.

    Returns current configuration including:
    - LLM parameters (alpha, epsilon, num_actions)
    - Learning strategy settings
    - Meta-learning configuration
    - Reward system weights
    - External LLM settings
    """
    try:
        logger.info("[ML API] Retrieving learning configuration")

        # Get various configuration components
        config_data = {
            'llm_config': {
                'num_actions': llm_config.num_actions,
                'alpha': llm_config.alpha,
                'epsilon': llm_config.epsilon,
                'learning_rate': getattr(llm_config, 'learning_rate', None),
                'discount_factor': getattr(llm_config, 'discount_factor', None)
            },
            'reward_weights': reward_service.reward_weights.copy(),
            'meta_learning': {
                'current_strategy': meta_learning_service.meta_learner.current_strategy if hasattr(meta_learning_service, 'meta_learner') else None,
                'adaptation_enabled': True,  # Assuming meta-learning is enabled
                'performance_tracking': True
            },
            'external_llm': {
                'enabled': get_external_llm_config().enabled,
                'providers': list(get_external_llm_config().enabled_providers) if get_external_llm_config().enabled else [],
                'default_provider': getattr(get_external_llm_config(), 'default_provider', None)
            },
            'curriculum': {
                'difficulty_levels': ['beginner', 'intermediate', 'advanced', 'expert'],
                'adaptive_pacing': True,
                'skill_tracking': True
            },
            'timestamp': time.time()
        }

        logger.info("[ML API] Learning configuration retrieved")
        return config_data

    except Exception as e:
        logger.error(f"[ML API] Failed to get learning config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning config: {str(e)}")

# Autonomous learning control endpoints
@app.post("/learning/autonomous/start")
async def start_autonomous_learning():
    """
    Start the autonomous learning loop.

    Begins the background autonomous learning process that continuously
    generates prompts, processes responses, and learns from interactions.
    """
    try:
        logger.info("[ML API] Starting autonomous learning loop")

        if learning_loop_service.is_running:
            logger.warning("[ML API] Autonomous learning loop is already running")
            return {
                "success": False,
                "message": "Autonomous learning loop is already running",
                "status": "already_running"
            }

        learning_loop_service.start_learning_loop()

        logger.info("[ML API] Autonomous learning loop started successfully")
        return {
            "success": True,
            "message": "Autonomous learning loop started successfully",
            "status": "started"
        }

    except Exception as e:
        logger.error(f"[ML API] Failed to start autonomous learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start autonomous learning: {str(e)}")

@app.post("/learning/autonomous/stop")
async def stop_autonomous_learning():
    """
    Stop the autonomous learning loop.

    Ends the background autonomous learning process.
    """
    try:
        logger.info("[ML API] Stopping autonomous learning loop")

        if not learning_loop_service.is_running:
            logger.warning("[ML API] Autonomous learning loop is not running")
            return {
                "success": False,
                "message": "Autonomous learning loop is not running",
                "status": "not_running"
            }

        learning_loop_service.stop_learning_loop()

        logger.info("[ML API] Autonomous learning loop stopped successfully")
        return {
            "success": True,
            "message": "Autonomous learning loop stopped successfully",
            "status": "stopped"
        }

    except Exception as e:
        logger.error(f"[ML API] Failed to stop autonomous learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop autonomous learning: {str(e)}")

@app.get("/learning/autonomous/status")
async def get_autonomous_learning_status():
    """
    Get the status of the autonomous learning loop.

    Returns comprehensive status including:
    - Whether loop is running
    - Whether waiting for initial user prompt
    - Current iteration count
    - Learning statistics
    - Thread health status
    - Initial prompt information
    """
    try:
        logger.info("[ML API] Getting autonomous learning loop status")

        # Get loop status using the new method
        loop_status = learning_loop_service.get_loop_status()

        # Get learning progress data (replaces get_stats)
        progress = learning_loop_service.get_learning_progress()

        status_data = {
            'is_running': loop_status['is_running'],
            'waiting_for_initial_prompt': loop_status['waiting_for_initial_prompt'],
            'initial_prompt_received': loop_status['initial_prompt_received'],
            'thread_alive': loop_status['thread_alive'],
            'current_iteration': progress.get('iterations', 0),
            'total_rewards': len(progress.get('recent_rewards', [])),
            'average_reward': progress.get('current_avg_reward', 0.0),
            'curriculum_skills_count': len(progress.get('curriculum_skill_stats', {})),
            'ppo_training_stats': progress.get('ppo_stats', {}),
            'initial_prompt_data': loop_status.get('initial_prompt_data'),
            'message': (
                "⏳ Waiting for initial user prompt from UI to start autonomous learning." 
                if loop_status['waiting_for_initial_prompt'] 
                else "✅ Autonomous learning loop is active." if loop_status['is_running']
                else "⏸️  Autonomous learning loop is stopped."
            )
        }

        logger.info(f"[ML API] Loop status: {status_data['message']}")
        return status_data

    except Exception as e:
        logger.error(f"Failed to get autonomous learning status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.get("/prompt/history", response_model=PromptHistoryResponse)
async def get_prompt_history():
    """
    Get all prompt interactions (both user and AI-generated) in chronological order.

    This endpoint provides:
    1. Complete history of all prompts processed by the system
    2. Chronological ordering by timestamp
    3. Source identification (user vs AI-generated)
    4. Response details and metadata
    """
    try:
        logger.info("[ML API] Retrieving prompt history")

        # Get all interactions from history in chronological order
        interactions = history.get_all_interactions_chronological()

        # Convert to response format
        prompt_items = []
        for interaction in interactions:
            prompt_items.append(PromptHistoryItem(
                prompt_text=interaction['prompt'].text,
                response=interaction['response'],
                reward=interaction.get('reward'),
                action=interaction.get('action'),
                timestamp=interaction.get('timestamp', 0),
                source=interaction.get('source', 'unknown')
            ))

        response_data = PromptHistoryResponse(
            interactions=prompt_items,
            total_count=len(prompt_items)
        )

        logger.info(f"[ML API] Prompt history retrieved with {len(prompt_items)} interactions")
        return response_data

    except Exception as e:
        logger.error(f"[ML API] Failed to retrieve prompt history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prompt history: {str(e)}")

@app.get("/learning/history", response_model=PromptHistoryResponse)
async def get_learning_history(limit: int = 100, offset: int = 0):
    """
    Get learning history data showing LLM prompts, actions, and interactions.

    This endpoint provides:
    1. Learning interactions with prompts, responses, and rewards
    2. Action taken by the LLM for each interaction
    3. Reward received and other learning metrics
    4. Chronological ordering by timestamp
    5. Pagination support for large datasets

    Query Parameters:
    - limit: Maximum number of interactions to return (default: 100)
    - offset: Number of interactions to skip (default: 0)
    """
    try:
        logger.info(f"[ML API] Retrieving learning history (limit: {limit}, offset: {offset})")

        # Get all interactions from history in chronological order
        all_interactions = history.get_all_interactions_chronological()

        # Apply pagination
        total_count = len(all_interactions)
        start_idx = min(offset, total_count)
        end_idx = min(start_idx + limit, total_count)
        paginated_interactions = all_interactions[start_idx:end_idx]

        # Convert to response format
        history_items = []
        for interaction in paginated_interactions:
            history_items.append(PromptHistoryItem(
                prompt_text=interaction['prompt'].text if hasattr(interaction['prompt'], 'text') else str(interaction['prompt']),
                response=interaction['response'],
                reward=interaction.get('reward'),
                action=interaction.get('action'),
                timestamp=interaction.get('timestamp', 0),
                source=interaction.get('source', 'unknown')
            ))

        response_data = PromptHistoryResponse(
            interactions=history_items,
            total_count=total_count
        )

        logger.info(f"[ML API] Learning history retrieved with {len(history_items)} interactions (total: {total_count})")
        return response_data

    except Exception as e:
        logger.error(f"[ML API] Failed to retrieve learning history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve learning history: {str(e)}")

# Reward system endpoints
@app.post("/rewards/configure", response_model=RewardWeightsResponse)
async def configure_reward_weights(config: RewardWeightsConfig):
    """
    Configure reward weights for multi-objective learning.

    Allows setting custom weights for different reward objectives:
    - accuracy: Weight for response correctness
    - coherence: Weight for logical consistency
    - factuality: Weight for factual accuracy
    - creativity: Weight for response diversity/novelty

    Weights must sum to 1.0.
    """
    try:
        weights_dict = config.dict()
        success = reward_service.configure_weights(weights_dict)

        if success:
            return RewardWeightsResponse(
                success=True,
                message="Reward weights configured successfully",
                current_weights=config
            )
        else:
            return RewardWeightsResponse(
                success=False,
                message="Failed to configure reward weights. Ensure weights sum to 1.0 and are between 0.0-1.0",
                current_weights=None
            )
    except Exception as e:
        logger.error(f"Failed to configure reward weights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure reward weights: {str(e)}")

@app.get("/rewards/metrics", response_model=RewardMetricsResponse)
async def get_reward_metrics():
    """
    Get current multi-objective reward metrics.

    Returns current reward values, configured weights, and historical averages.
    """
    try:
        current_metrics = reward_service.get_current_metrics()
        average_metrics = reward_service.get_average_metrics(window=50)

        # Convert to response format with proper structure
        metrics_response = RewardMetricsResponse(
            current_metrics=RewardMetrics(
                individual_rewards=MultiObjectiveRewards(
                    accuracy=current_metrics['current_metrics']['accuracy'],
                    coherence=current_metrics['current_metrics']['coherence'],
                    factuality=current_metrics['current_metrics']['factuality'],
                    creativity=current_metrics['current_metrics']['creativity']
                ),
                total_reward=current_metrics['current_metrics']['total_reward'],
                weighted_reward=current_metrics['current_metrics']['weighted_reward'],
                weights_used=RewardWeightsConfig(**current_metrics['reward_weights'])
            ),
            reward_weights=RewardWeightsConfig(**current_metrics['reward_weights']),
            history_length=current_metrics['history_length'],
            average_metrics=average_metrics
        )

        # Add data availability indicator
        if hasattr(metrics_response, '__dict__'):
            metrics_response.__dict__['data_available'] = current_metrics['history_length'] > 0
            if not metrics_response.data_available:
                metrics_response.__dict__['message'] = "No reward data available yet. Start interacting with the system to generate reward metrics."

        return metrics_response
    except Exception as e:
        logger.error(f"Failed to get reward metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get reward metrics: {str(e)}")

@app.get("/rewards/breakdown")
async def get_reward_breakdown():
    """
    Get multi-objective reward breakdown.

    Returns detailed breakdown of current reward components including:
    - Individual reward scores for accuracy, coherence, factuality, creativity
    - Weighted contributions to total reward
    - Performance analysis and trends
    """
    try:
        logger.info("[ML API] Retrieving reward breakdown")

        # Get current metrics with detailed breakdown
        current_metrics = reward_service.get_current_metrics()
        average_metrics = reward_service.get_average_metrics(window=20)

        breakdown = {
            'current_breakdown': {
                'individual_rewards': current_metrics['current_metrics'],
                'weights_used': current_metrics['reward_weights'],
                'total_reward': current_metrics['current_metrics']['total_reward'],
                'weighted_reward': current_metrics['current_metrics']['weighted_reward']
            },
            'historical_averages': average_metrics,
            'performance_analysis': {
                'best_performing_objective': max(current_metrics['current_metrics'], key=lambda k: current_metrics['current_metrics'][k] if k in ['accuracy', 'coherence', 'factuality', 'creativity'] else 0),
                'needs_improvement': [obj for obj in ['accuracy', 'coherence', 'factuality', 'creativity'] if current_metrics['current_metrics'].get(obj, 0) < 0.5],
                'consistency_score': sum(average_metrics.values()) / len(average_metrics) if average_metrics else 0.0
            },
            'timestamp': time.time()
        }

        logger.info("[ML API] Reward breakdown retrieved")
        return breakdown

    except Exception as e:
        logger.error(f"[ML API] Failed to get reward breakdown: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get reward breakdown: {str(e)}")

@app.get("/rewards/history")
async def get_reward_history(limit: int = 100):
    """
    Get time-series reward data.

    Returns historical reward data for trend analysis.

    Query Parameters:
    - limit: Maximum number of history entries to return (default: 100)
    """
    try:
        logger.info(f"[ML API] Retrieving reward history (limit: {limit})")

        history = reward_service.get_reward_history(limit=limit)

        # Format for time-series analysis
        formatted_history = []
        for entry in history:
            formatted_entry = {
                'timestamp': entry.get('timestamp', time.time()),
                'individual_rewards': entry.get('rewards', {}),
                'adjusted_rewards': entry.get('adjusted_rewards', {}),
                'total_reward': entry.get('total_reward', 0.0),
                'weighted_reward': entry.get('weighted_reward', 0.0),
                'weights_used': entry.get('weights', reward_service.reward_weights.copy()),
                'user_id': entry.get('user_id'),
                'response_id': entry.get('response_id')
            }
            formatted_history.append(formatted_entry)

        response = {
            'history': formatted_history,
            'total_entries': len(formatted_history),
            'time_range': f"Last {len(formatted_history)} reward calculations",
            'summary': {
                'avg_weighted_reward': sum(h['weighted_reward'] for h in formatted_history) / max(1, len(formatted_history)),
                'avg_total_reward': sum(h['total_reward'] for h in formatted_history) / max(1, len(formatted_history)),
                'most_recent_weights': formatted_history[0]['weights_used'] if formatted_history else reward_service.reward_weights.copy()
            },
            'data_available': len(formatted_history) > 0,
            'message': "No reward history available yet" if len(formatted_history) == 0 else None
        }

        logger.info(f"[ML API] Reward history retrieved: {len(formatted_history)} entries")
        return response

    except Exception as e:
        logger.error(f"[ML API] Failed to get reward history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get reward history: {str(e)}")

@app.get("/rewards/weights")
async def get_reward_weights():
    """
    Get current reward weights configuration.

    Returns the current reward weights for multi-objective optimization.
    """
    try:
        logger.info("[ML API] Retrieving current reward weights")

        weights = reward_service.reward_weights.copy()

        response = {
            'weights': weights,
            'description': {
                'accuracy': 'Weight for response correctness and relevance',
                'coherence': 'Weight for logical consistency and flow',
                'factuality': 'Weight for factual accuracy and hallucination avoidance',
                'creativity': 'Weight for response diversity and novelty'
            },
            'constraints': 'Weights must sum to 1.0',
            'current_sum': sum(weights.values()),
            'timestamp': time.time()
        }

        logger.info("[ML API] Reward weights retrieved")
        return response

    except Exception as e:
        logger.error(f"[ML API] Failed to get reward weights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get reward weights: {str(e)}")

@app.post("/rewards/weights")
async def update_reward_weights(config: RewardWeightsConfig):
    """
    Update reward weights configuration.

    Allows updating the weights for multi-objective reward optimization.
    """
    try:
        logger.info("[ML API] Updating reward weights")

        success = reward_service.configure_weights(config.dict())

        if success:
            logger.info("Reward weights updated successfully")
            return {
                "success": True,
                "message": "Reward weights updated successfully",
                "new_weights": config.dict(),
                "timestamp": time.time()
            }
        else:
            logger.error("Failed to update reward weights")
            return {
                "success": False,
                "message": "Failed to update reward weights. Ensure weights sum to 1.0 and are between 0.0-1.0",
                "current_weights": reward_service.reward_weights.copy()
            }

    except Exception as e:
        logger.error(f"[ML API] Failed to update reward weights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update reward weights: {str(e)}")

# Metrics endpoints
@app.get("/metrics/interactions")
async def get_interaction_metrics(hours: int = 24):
    """
    Get interaction volume data.

    Returns time-series data about system interactions and usage patterns.

    Query Parameters:
    - hours: Number of hours to look back (default: 24)
    """
    try:
        logger.info(f"[ML API] Retrieving interaction metrics for last {hours} hours")

        # Get current metrics from prompt controller
        current_metrics = prompt_controller.get_learning_metrics()

        # Get recent history from history service
        recent_interactions = history.get_all_interactions_chronological()

        # Filter by time window
        cutoff_time = time.time() - (hours * 3600)
        filtered_interactions = [i for i in recent_interactions if i.get('timestamp', 0) >= cutoff_time]

        # Group by hour for time-series analysis
        hourly_stats = {}
        for interaction in filtered_interactions:
            hour_key = int(interaction.get('timestamp', time.time()) // 3600) * 3600
            if hour_key not in hourly_stats:
                hourly_stats[hour_key] = {
                    'total_interactions': 0,
                    'successful_interactions': 0,
                    'avg_reward': 0.0,
                    'unique_users': set(),
                    'action_distribution': {}
                }

            stats = hourly_stats[hour_key]
            stats['total_interactions'] += 1

            # Count successful interactions (those with rewards)
            if interaction.get('reward') is not None:
                stats['successful_interactions'] += 1
                stats['avg_reward'] += interaction['reward']

            # Track unique users
            if 'user_id' in interaction:
                stats['unique_users'].add(interaction['user_id'])

            # Track action distribution
            action = interaction.get('action')
            if action is not None:
                stats['action_distribution'][action] = stats['action_distribution'].get(action, 0) + 1

        # Calculate averages and finalize stats
        for hour_key, stats in hourly_stats.items():
            if stats['successful_interactions'] > 0:
                stats['avg_reward'] /= stats['successful_interactions']
            stats['success_rate'] = stats['successful_interactions'] / stats['total_interactions'] if stats['total_interactions'] > 0 else 0.0
            stats['unique_users'] = len(stats['unique_users'])

        # Convert to response format
        interaction_data = {
            'time_range_hours': hours,
            'total_interactions': len(filtered_interactions),
            'current_metrics': current_metrics,
            'hourly_breakdown': [
                {
                    'timestamp': hour_key,
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(hour_key)),
                    **stats
                }
                for hour_key, stats in sorted(hourly_stats.items())
            ],
            'summary': {
                'avg_interactions_per_hour': len(filtered_interactions) / max(1, hours),
                'overall_success_rate': current_metrics.get('success_rate', 0.0),
                'total_unique_users': len(set(i.get('user_id') for i in filtered_interactions if i.get('user_id'))),
                'peak_hour_interactions': max((s['total_interactions'] for s in hourly_stats.values()), default=0)
            },
            'timestamp': time.time(),
            'data_available': len(filtered_interactions) > 0,
            'message': "No interaction data available yet" if len(filtered_interactions) == 0 else None
        }

        logger.info(f"[ML API] Interaction metrics retrieved: {len(filtered_interactions)} interactions analyzed")
        return interaction_data

    except Exception as e:
        logger.error(f"[ML API] Failed to get interaction metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get interaction metrics: {str(e)}")

@app.post("/evaluate/multi", response_model=EvaluateMultiResponse)
async def evaluate_multi_objective(request: EvaluateMultiRequest):
    """
    Evaluate the LLM with multi-objective reward system.

    Runs multiple episodes and returns average rewards across all objectives.
    """
    try:
        # For multi-objective evaluation, we need to run episodes with the reward service
        total_weighted_reward = 0.0
        total_individual_rewards = {'accuracy': 0.0, 'coherence': 0.0, 'factuality': 0.0, 'creativity': 0.0}
        episodes_completed = 0

        for episode in range(request.num_episodes):
            try:
                # Generate a prompt (use provided or generate random)
                if request.prompt_text:
                    prompt = Prompt(request.prompt_text)
                else:
                    # Generate a simple test prompt
                    test_prompts = [
                        "Create a machine learning model for image classification",
                        "Explain how neural networks work",
                        "Build a recommendation system",
                        "Analyze time series data",
                        "Implement natural language processing"
                    ]
                    prompt_text = test_prompts[episode % len(test_prompts)]
                    prompt = Prompt(prompt_text)

                # Process prompt
                response, action = llm.process_prompt(prompt.text)

                # Calculate multi-objective reward
                reward_result = reward_service.calculate_multi_objective_reward(
                    prompt.text, response, action
                )

                # Learn from the reward
                llm.learn(action, reward_result)

                # Accumulate results
                total_weighted_reward += reward_result['weighted_reward']
                for obj in total_individual_rewards:
                    total_individual_rewards[obj] += reward_result['individual_rewards'][obj]

                episodes_completed += 1

            except Exception as e:
                logger.warning(f"Episode {episode} failed: {str(e)}")
                continue

        if episodes_completed == 0:
            raise HTTPException(status_code=400, detail="No episodes completed successfully")

        # Calculate averages
        avg_weighted_reward = total_weighted_reward / episodes_completed
        avg_individual_rewards = {
            obj: total / episodes_completed
            for obj, total in total_individual_rewards.items()
        }

        return EvaluateMultiResponse(
            average_weighted_reward=avg_weighted_reward,
            average_individual_rewards=MultiObjectiveRewards(**avg_individual_rewards),
            total_episodes=episodes_completed
        )

    except Exception as e:
        logger.error(f"Failed to run multi-objective evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run multi-objective evaluation: {str(e)}")

# Hallucination detection endpoints
@app.post("/hallucination/check", response_model=HallucinationCheckResponse)
async def check_hallucination(request: HallucinationCheckRequest):
    """
    Check a response for potential hallucinations.

    Analyzes the provided prompt and response for:
    - Pattern-based hallucination detection
    - Confidence scoring and uncertainty metrics
    - Fact-checking against known knowledge patterns
    - Risk level assessment

    Returns detailed analysis including indicators, confidence scores, and recommendations.
    """
    try:
        logger.info(f"[ML API] Checking hallucination for response: {request.response_text[:50]}...")
        response = hallucination_service.check_hallucination(request)
        logger.info(f"[ML API] Hallucination check completed: confidence={response.analysis.overall_confidence:.3f}, "
                   f"hallucinated={response.analysis.is_hallucinated}")
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to check hallucination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check hallucination: {str(e)}")

@app.get("/hallucination/metrics", response_model=HallucinationMetricsResponse)
async def get_hallucination_metrics():
    """
    Get hallucination detection metrics and statistics.

    Returns comprehensive metrics including:
    - Total checks performed
    - Detection rate and false positive rate
    - Average confidence and uncertainty scores
    - Common hallucination types distribution
    - Recent analysis history
    """
    try:
        logger.info("[ML API] Retrieving hallucination metrics")
        metrics = hallucination_service.get_metrics()
        recent_analyses = hallucination_service.get_recent_analyses(limit=10)

        response = HallucinationMetricsResponse(
            metrics=metrics,
            recent_analyses=recent_analyses,
            timestamp=time.time()
        )

        logger.info(f"[ML API] Hallucination metrics retrieved: {metrics.total_checks} total checks, "
                   f"detection_rate={metrics.detection_rate:.3f}")
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to get hallucination metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get hallucination metrics: {str(e)}")

@app.post("/hallucination/configure")
async def configure_hallucination_detection(config: HallucinationConfig):
    """
    Configure hallucination detection parameters.

    Allows updating:
    - Detection thresholds and weights
    - Pattern detection phrases
    - Fact-checking knowledge patterns
    - Confidence and uncertainty thresholds

    All changes take effect immediately for subsequent checks.
    """
    try:
        logger.info("[ML API] Configuring hallucination detection parameters")
        success = hallucination_service.update_config(config)

        if success:
            logger.info("Hallucination detection configuration updated successfully")
            return {
                "success": True,
                "message": "Hallucination detection configuration updated successfully",
                "config": config.dict()
            }
        else:
            logger.error("Failed to update hallucination detection configuration")
            return {
                "success": False,
                "message": "Failed to update hallucination detection configuration",
                "config": None
            }
    except Exception as e:
        logger.error(f"[ML API] Failed to configure hallucination detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure hallucination detection: {str(e)}")

@app.get("/hallucination/trends")
async def get_hallucination_trends(hours: int = 24):
    """
    Get 24-hour hallucination trend data.

    Returns time-series data showing hallucination patterns over the specified time period.

    Query Parameters:
    - hours: Number of hours to look back (default: 24)
    """
    try:
        logger.info(f"[ML API] Retrieving hallucination trends for last {hours} hours")

        # Get recent analyses
        recent_analyses = hallucination_service.get_recent_analyses(limit=1000)

        # Filter by time window
        cutoff_time = time.time() - (hours * 3600)
        filtered_analyses = [a for a in recent_analyses if hasattr(a, 'timestamp') and a.timestamp >= cutoff_time]

        # Group by hour
        hourly_stats = {}
        for analysis in filtered_analyses:
            hour_key = int(analysis.timestamp // 3600) * 3600  # Round to hour
            if hour_key not in hourly_stats:
                hourly_stats[hour_key] = {
                    'total_checks': 0,
                    'hallucinated': 0,
                    'avg_confidence': 0.0,
                    'avg_uncertainty': 0.0,
                    'risk_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
                }

            stats = hourly_stats[hour_key]
            stats['total_checks'] += 1
            if analysis.is_hallucinated:
                stats['hallucinated'] += 1
            stats['avg_confidence'] += analysis.overall_confidence
            stats['avg_uncertainty'] += analysis.uncertainty_score
            risk_key = analysis.risk_level.value if hasattr(analysis.risk_level, 'value') else str(analysis.risk_level)
            if risk_key in stats['risk_distribution']:
                stats['risk_distribution'][risk_key] += 1

        # Calculate averages
        for hour_key, stats in hourly_stats.items():
            if stats['total_checks'] > 0:
                stats['avg_confidence'] /= stats['total_checks']
                stats['avg_uncertainty'] /= stats['total_checks']
                stats['detection_rate'] = stats['hallucinated'] / stats['total_checks']

        # Convert to time-series format
        trends = {
            'time_range_hours': hours,
            'total_analyses': len(filtered_analyses),
            'hourly_data': [
                {
                    'timestamp': hour_key,
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(hour_key)),
                    **stats
                }
                for hour_key, stats in sorted(hourly_stats.items())
            ],
            'summary': {
                'overall_detection_rate': sum(s['hallucinated'] for s in hourly_stats.values()) / max(1, sum(s['total_checks'] for s in hourly_stats.values())),
                'peak_hour_detection_rate': max((s['detection_rate'] for s in hourly_stats.values() if 'detection_rate' in s), default=0.0),
                'avg_confidence': sum(s['avg_confidence'] for s in hourly_stats.values()) / max(1, len(hourly_stats))
            },
            'data_available': len(filtered_analyses) > 0,
            'message': "No hallucination data available yet" if len(filtered_analyses) == 0 else None
        }

        logger.info(f"[ML API] Hallucination trends retrieved: {len(trends['hourly_data'])} data points")
        return trends

    except Exception as e:
        logger.error(f"[ML API] Failed to get hallucination trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get hallucination trends: {str(e)}")

@app.get("/hallucination/detections")
async def get_recent_hallucination_detections(limit: int = 50):
    """
    Get recent hallucination detections.

    Returns the most recent hallucination detections with details.

    Query Parameters:
    - limit: Maximum number of detections to return (default: 50)
    """
    try:
        logger.info(f"[ML API] Retrieving recent hallucination detections (limit: {limit})")

        # Get recent analyses and filter for hallucinations
        recent_analyses = hallucination_service.get_recent_analyses(limit=limit * 2)  # Get more to filter
        hallucinated_analyses = [a for a in recent_analyses if a.is_hallucinated][:limit]

        detections = []
        for analysis in hallucinated_analyses:
            detection = {
                'timestamp': analysis.timestamp if hasattr(analysis, 'timestamp') else time.time(),
                'confidence': analysis.overall_confidence,
                'risk_level': analysis.risk_level.value if hasattr(analysis.risk_level, 'value') else str(analysis.risk_level),
                'indicators': [
                    {
                        'type': ind.type.value if hasattr(ind.type, 'value') else str(ind.type),
                        'severity': ind.severity.value if hasattr(ind.severity, 'value') else str(ind.severity),
                        'confidence': ind.confidence,
                        'text_snippet': ind.text_snippet,
                        'explanation': ind.explanation
                    }
                    for ind in analysis.indicators
                ],
                'analysis_metadata': analysis.analysis_metadata if hasattr(analysis, 'analysis_metadata') else {}
            }
            detections.append(detection)

        response = {
            'total_detections': len(detections),
            'detections': detections,
            'time_range': f"Last {len(recent_analyses)} analyses checked",
            'data_available': len(detections) > 0,
            'message': "No hallucination detections available yet" if len(detections) == 0 else None
        }

        logger.info(f"[ML API] Recent hallucination detections retrieved: {len(detections)} detections")
        return response

    except Exception as e:
        logger.error(f"[ML API] Failed to get recent hallucination detections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent hallucination detections: {str(e)}")

@app.get("/hallucination/risk-distribution")
async def get_hallucination_risk_distribution(hours: int = 24):
    """
    Get risk level distribution for hallucinations.

    Returns the distribution of hallucination risk levels over the specified time period.

    Query Parameters:
    - hours: Number of hours to look back (default: 24)
    """
    try:
        logger.info(f"[ML API] Retrieving hallucination risk distribution for last {hours} hours")

        # Get recent analyses
        recent_analyses = hallucination_service.get_recent_analyses(limit=1000)

        # Filter by time window
        cutoff_time = time.time() - (hours * 3600)
        filtered_analyses = [a for a in recent_analyses if hasattr(a, 'timestamp') and a.timestamp >= cutoff_time]

        # Calculate risk distribution
        risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        confidence_ranges = {'LOW': [], 'MEDIUM': [], 'HIGH': [], 'CRITICAL': []}

        for analysis in filtered_analyses:
            risk_key = analysis.risk_level.value if hasattr(analysis.risk_level, 'value') else str(analysis.risk_level)
            if risk_key in risk_distribution:
                risk_distribution[risk_key] += 1
                confidence_ranges[risk_key].append(analysis.overall_confidence)

        # Calculate statistics for each risk level
        distribution_stats = {}
        for risk_level in risk_distribution:
            count = risk_distribution[risk_level]
            confidences = confidence_ranges[risk_level]

            distribution_stats[risk_level] = {
                'count': count,
                'percentage': count / max(1, len(filtered_analyses)) * 100,
                'avg_confidence': sum(confidences) / max(1, len(confidences)),
                'min_confidence': min(confidences) if confidences else 0.0,
                'max_confidence': max(confidences) if confidences else 0.0
            }

        response = {
            'time_range_hours': hours,
            'total_analyses': len(filtered_analyses),
            'risk_distribution': distribution_stats,
            'summary': {
                'most_common_risk': max(risk_distribution, key=risk_distribution.get) if risk_distribution else 'NONE',
                'highest_risk_count': max(risk_distribution.values()) if risk_distribution else 0,
                'overall_risk_trend': 'high' if distribution_stats.get('CRITICAL', {}).get('count', 0) > distribution_stats.get('LOW', {}).get('count', 0) else 'low'
            },
            'data_available': len(filtered_analyses) > 0,
            'message': "No hallucination risk data available yet" if len(filtered_analyses) == 0 else None
        }

        logger.info(f"[ML API] Hallucination risk distribution retrieved: {len(filtered_analyses)} analyses analyzed")
        return response

    except Exception as e:
        logger.error(f"[ML API] Failed to get hallucination risk distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get hallucination risk distribution: {str(e)}")

# Curriculum management endpoints
@app.get("/curriculum/skills", response_model=CurriculumSkillsResponse)
async def get_curriculum_skills():
    """
    Get the current skill progression tree grouped by categories.

    Returns skills grouped by categories with frontend-compatible structure:
    - Skills grouped by category names
    - Each skill has name, level (mapped from difficulty), and completed (default false)
    """
    try:
        logger.info("[ML API] Retrieving curriculum skills")
        curriculum_tree = scheduler.curriculum_tree

        if not curriculum_tree or not hasattr(curriculum_tree, 'skills') or not curriculum_tree.skills:
            logger.info("[ML API] No curriculum skills available, returning empty state")
            return CurriculumSkillsResponse(
                skills={},
                categories={},
                difficulty_levels={}
            )

        # Group skills by categories for frontend compatibility
        skills_by_category = {}
        for category, skill_ids in curriculum_tree.categories.items():
            skills_by_category[category] = []
            for skill_id in skill_ids:
                skill = curriculum_tree.skills[skill_id]
                # Map to frontend expected structure
                level_map = {
                    "easy": "Beginner",
                    "medium": "Intermediate",
                    "hard": "Advanced",
                    "expert": "Expert"
                }
                frontend_skill = {
                    "name": skill.name,
                    "level": level_map.get(skill.difficulty.value, "Beginner"),
                    "completed": False  # Will be updated by progress data
                }
                skills_by_category[category].append(frontend_skill)

        response = CurriculumSkillsResponse(
            skills=skills_by_category,
            categories=curriculum_tree.categories,
            difficulty_levels={diff.value: skills for diff, skills in curriculum_tree.difficulty_levels.items()}
        )

        logger.info(f"[ML API] Curriculum skills retrieved: {len(curriculum_tree.skills)} skills in {len(skills_by_category)} categories")
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to retrieve curriculum skills: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve curriculum skills: {str(e)}")

@app.post("/curriculum/adjust")
async def adjust_curriculum(request: CurriculumAdjustRequest):
    """
    Adjust curriculum difficulty and focus dynamically.

    Allows adjusting:
    - Target difficulty level for task generation
    - Skill focus weights for personalized learning
    - Dynamic curriculum adaptation based on learner needs
    """
    try:
        logger.info(f"[ML API] Adjusting curriculum for learner {request.learner_id}")

        # Update learner progress if they exist
        progress = scheduler.get_learner_progress(request.learner_id)
        if not progress:
            scheduler.register_learner(request.learner_id)
            progress = scheduler.get_learner_progress(request.learner_id)

        # Adjust difficulty if specified
        if request.target_difficulty:
            from models.curriculum import DifficultyLevel
            try:
                new_difficulty = DifficultyLevel(request.target_difficulty.lower())
                progress.current_difficulty = new_difficulty
                logger.info(f"Updated difficulty to {new_difficulty.value} for learner {request.learner_id}")
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid difficulty level: {request.target_difficulty}")

        # Adjust skill focus if specified
        if request.skill_focus:
            for skill_id, weight in request.skill_focus.items():
                scheduler.task_generator.update_skill_focus(skill_id, weight)
            logger.info(f"Updated skill focus for learner {request.learner_id}")

        return {
            "success": True,
            "message": "Curriculum adjusted successfully",
            "learner_id": request.learner_id,
            "current_difficulty": progress.current_difficulty.value if progress else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ML API] Failed to adjust curriculum: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to adjust curriculum: {str(e)}")

@app.get("/curriculum/progress", response_model=CurriculumProgressResponse)
async def get_curriculum_progress(learner_id: str):
    """
    Get curriculum completion progress for a learner.

    Returns comprehensive progress data including:
    - Skill mastery levels
    - Available skills to learn next
    - Curriculum completion percentage
    - Learning recommendations
    """
    try:
        logger.info(f"[ML API] Retrieving curriculum progress for learner {learner_id}")

        status = scheduler.get_curriculum_status(learner_id)
        if "error" in status:
            # Return empty state data instead of error for better UX
            logger.info(f"[ML API] Learner {learner_id} not found, returning empty state data")
            empty_response = CurriculumProgressResponse(
                learner_id=learner_id,
                progress={
                    "skill_progress": {},
                    "completed_skills": [],
                    "current_difficulty": "beginner",
                    "total_time_spent": 0,
                    "last_activity": None
                },
                available_skills=[],
                recommendations={
                    "next_skills": [],
                    "difficulty_adjustment": None,
                    "focus_areas": []
                },
                curriculum_summary={
                    "total_skills": len(scheduler.curriculum_tree.skills),
                    "completed_skills": 0,
                    "completion_percentage": 0.0,
                    "current_difficulty": "beginner",
                    "recommended_difficulty": "beginner"
                }
            )
            return empty_response

        response = CurriculumProgressResponse(**status)
        logger.info(f"[ML API] Curriculum progress retrieved for learner {learner_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ML API] Failed to retrieve curriculum progress: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve curriculum progress: {str(e)}")

@app.post("/curriculum/reset")
async def reset_curriculum(request: CurriculumResetRequest):
    """
    Reset curriculum progress to beginning.

    WARNING: This will reset all learner progress and cannot be undone.
    Requires explicit confirmation via confirm_reset flag.
    """
    try:
        logger.info(f"[ML API] Reset request for learner {request.learner_id}")

        if not request.confirm_reset:
            raise HTTPException(
                status_code=400,
                detail="Reset not confirmed. Set confirm_reset=true to proceed with curriculum reset."
            )

        # Reset learner progress
        if request.learner_id in scheduler.learner_progress:
            scheduler.learner_progress[request.learner_id] = scheduler.register_learner(request.learner_id)
            logger.info(f"Curriculum reset for learner {request.learner_id}")
        else:
            # Create new progress if learner doesn't exist
            scheduler.register_learner(request.learner_id)
            logger.info(f"Created new curriculum progress for learner {request.learner_id}")

        return {
            "success": True,
            "message": "Curriculum reset successfully",
            "learner_id": request.learner_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ML API] Failed to reset curriculum: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset curriculum: {str(e)}")

@app.get("/curriculum/gaps")
async def get_curriculum_gaps(learner_id: str):
    """
    Analyze skill gaps and recommendations for a learner.

    Returns comprehensive skill gap analysis including:
    - Identified skill gaps with severity levels
    - Recommendations for closing gaps
    - Estimated time and resources needed
    - Learning path suggestions
    """
    try:
        logger.info(f"[ML API] Analyzing skill gaps for learner {learner_id}")

        from models.analytics import SkillGapAnalysisRequest
        request = SkillGapAnalysisRequest(learner_id=learner_id, include_recommendations=True)
        response = analytics_controller.analyze_skill_gaps(request)

        # Check if response is None (no data available)
        if response is None or response.analysis is None:
            logger.info(f"[ML API] No skill gap data available for learner {learner_id}, returning empty state")
            return {
                "analysis": {
                    "learner_id": learner_id,
                    "timestamp": time.time(),
                    "skill_gaps": [],
                    "overall_gap_score": 0.0,
                    "critical_gaps_count": 0,
                    "recommendations": ["No learning data available yet. Start by completing some tasks to generate skill gap analysis."]
                },
                "processing_time": 0.0,
                "data_available": False,
                "message": "No learning data available yet for skill gap analysis"
            }

        logger.info(f"[ML API] Skill gap analysis completed for learner {learner_id}")
        return response

    except Exception as e:
        logger.error(f"[ML API] Failed to analyze skill gaps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze skill gaps: {str(e)}")

@app.get("/curriculum/recommendations")
async def get_curriculum_recommendations(learner_id: str):
    """
    Get personalized learning recommendations for a learner.

    Returns tailored recommendations including:
    - Next skills to learn based on current progress
    - Difficulty level adjustments
    - Learning pace suggestions
    - Focus area priorities
    """
    try:
        logger.info(f"[ML API] Generating curriculum recommendations for learner {learner_id}")

        # Get current progress and generate recommendations
        progress = scheduler.get_curriculum_status(learner_id)
        if "error" in progress:
            # Return empty state recommendations
            logger.info(f"[ML API] Learner {learner_id} not found, returning empty state recommendations")
            return {
                "learner_id": learner_id,
                "current_progress": None,
                "recommendations": [
                    {
                        "type": "getting_started",
                        "priority": "high",
                        "description": "No learning data available yet. Start by completing some tasks to get personalized recommendations.",
                        "suggested_actions": ["complete_first_task", "explore_available_skills"]
                    }
                ],
                "timestamp": time.time(),
                "data_available": False,
                "message": "No learning data available yet for recommendations"
            }

        # Generate recommendations based on progress
        recommendations = {
            "learner_id": learner_id,
            "current_progress": progress,
            "recommendations": [],
            "timestamp": time.time(),
            "data_available": True
        }

        # Add skill-based recommendations
        available_skills = progress.get("available_skills", [])
        if available_skills:
            recommendations["recommendations"].append({
                "type": "skill_progression",
                "priority": "high",
                "description": f"Continue with available skills: {', '.join(available_skills[:3])}",
                "skills": available_skills[:5]
            })

        # Add difficulty recommendations
        current_difficulty = progress.get("curriculum_summary", {}).get("current_difficulty", "beginner")
        if current_difficulty == "beginner":
            recommendations["recommendations"].append({
                "type": "difficulty_adjustment",
                "priority": "medium",
                "description": "Consider progressing to intermediate level once basic skills are mastered",
                "suggested_difficulty": "intermediate"
            })

        # Add pace recommendations
        completion_rate = progress.get("curriculum_summary", {}).get("completion_percentage", 0)
        if completion_rate > 80:
            recommendations["recommendations"].append({
                "type": "pace_adjustment",
                "priority": "low",
                "description": "Excellent progress! Consider increasing learning pace or exploring advanced topics",
                "suggested_actions": ["increase_frequency", "explore_advanced"]
            })

        logger.info(f"[ML API] Curriculum recommendations generated for learner {learner_id}")
        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ML API] Failed to generate curriculum recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate curriculum recommendations: {str(e)}")

# Meta-learning endpoints
@app.get("/meta/status", response_model=MetaLearningStatusResponse)
async def get_meta_learning_status():
    """
    Get comprehensive meta-learning system status.

    Returns current strategy, parameters, active rules, performance summary,
    and adaptation recommendations.
    """
    try:
        logger.info("[ML API] Retrieving meta-learning status")
        status = meta_learning_service.meta_learner.get_meta_learning_status()
        logger.info(f"[ML API] Meta-learning status retrieved: strategy={status['current_strategy']}")
        return MetaLearningStatusResponse(**status)
    except Exception as e:
        logger.error(f"[ML API] Failed to get meta-learning status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get meta-learning status: {str(e)}")

@app.get("/meta/performance", response_model=MetaPerformanceMetricsResponse)
async def get_meta_performance_metrics():
    """
    Get comprehensive meta-learning performance metrics.

    Returns strategy performance, recent performance data, adaptation history,
    context awareness, and transfer learning memory.
    """
    try:
        logger.info("[ML API] Retrieving meta-learning performance metrics")
        metrics = meta_learning_service.get_meta_performance_metrics()
        logger.info(f"[ML API] Meta-learning performance metrics retrieved: {len(metrics['recent_performance'])} recent entries")
        return MetaPerformanceMetricsResponse(**metrics)
    except Exception as e:
        logger.error(f"[ML API] Failed to get meta-learning performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get meta-learning performance metrics: {str(e)}")

@app.post("/meta/adapt", response_model=MetaAdaptationTriggerResponse)
async def trigger_meta_adaptation():
    """
    Manually trigger meta-learning parameter adaptation.

    This forces the system to evaluate current performance and adapt
    learning parameters and strategies as needed.
    """
    try:
        logger.info("[ML API] Triggering meta-learning adaptation")
        result = meta_learning_service.trigger_adaptation()
        logger.info(f"[ML API] Meta-learning adaptation triggered: success={result['success']}")
        return MetaAdaptationTriggerResponse(**result)
    except Exception as e:
        logger.error(f"[ML API] Failed to trigger meta-learning adaptation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger meta-learning adaptation: {str(e)}")

@app.get("/meta/strategies", response_model=MetaStrategiesResponse)
async def get_available_strategies():
    """
    Get information about all available learning strategies.

    Returns strategy names, descriptions, performance history,
    and current status for each strategy.
    """
    try:
        logger.info("[ML API] Retrieving available learning strategies")
        strategies = meta_learning_service.get_available_strategies()
        logger.info(f"[ML API] Available strategies retrieved: {len(strategies)} strategies")
        return MetaStrategiesResponse(strategies=strategies)
    except Exception as e:
        logger.error(f"[ML API] Failed to get available strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available strategies: {str(e)}")

@app.post("/meta/strategy", response_model=MetaStrategySwitchResponse)
async def switch_learning_strategy(request: MetaStrategySwitchRequest):
    """
    Switch to a different learning strategy.

    Allows manual override of the current learning strategy.
    The system will continue using this strategy until adaptation triggers a change.
    """
    try:
        logger.info(f"[ML API] Switching learning strategy to: {request.strategy_name}")
        result = meta_learning_service.switch_strategy(request.strategy_name)
        logger.info(f"[ML API] Strategy switch result: success={result['success']}")
        return MetaStrategySwitchResponse(**result)
    except Exception as e:
        logger.error(f"[ML API] Failed to switch learning strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to switch learning strategy: {str(e)}")

# External LLM endpoints
@app.post("/llm/switch", response_model=LLMSwitchResponse)
async def switch_llm_mode(request: LLMSwitchRequest):
    """
    Switch between internal and external LLM modes.

    Allows dynamic switching between the internal PPO-based LLM and external
    commercial LLMs (OpenAI, Anthropic, etc.) for response generation.
    """
    try:
        logger.info(f"🔄 Switching LLM mode to {'external' if request.use_external else 'internal'}")

        success = prompt_controller.switch_llm_mode(request.use_external)
        current_mode = "external" if request.use_external else "internal"

        if success:
            # Log external LLM interaction change
            log_external_llm_interaction(
                provider="system",
                model="mode_switch",
                operation="switch_mode",
                error=None if success else "Switch failed"
            )

            logger.info(f"✅ Successfully switched to {current_mode} LLM mode")
            return LLMSwitchResponse(
                success=True,
                message=f"Successfully switched to {current_mode} LLM mode",
                current_mode=current_mode
            )
        else:
            logger.error("❌ Failed to switch LLM mode")
            return LLMSwitchResponse(
                success=False,
                message="Failed to switch LLM mode",
                current_mode="unknown"
            )
    except Exception as e:
        logger.error(f"❌ Failed to switch LLM mode: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to switch LLM mode: {str(e)}")

@app.get("/llm/models", response_model=LLMModelsResponse)
async def get_llm_models():
    """
    Get available external LLM models and their capabilities.

    Returns information about all configured external LLM models including
    their providers, context windows, costs, and capabilities.
    """
    try:
        models = prompt_controller.get_external_llm_models()
        logger.info(f"[ML API] Retrieved {len(models)} external LLM models")
        return LLMModelsResponse(
            models=models,
            total_count=len(models)
        )
    except Exception as e:
        logger.error(f"[ML API] Failed to get LLM models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get LLM models: {str(e)}")

@app.get("/llm/costs", response_model=LLMCostsResponse)
async def get_llm_costs(provider: Optional[str] = None):
    """
    Get external LLM usage costs and limits.

    Returns cost tracking information for external LLM usage, including
    total costs, monthly budgets, and usage limits.
    """
    try:
        costs = prompt_controller.get_external_llm_costs(provider)
        total_providers = len(costs) if isinstance(costs, dict) else 1
        logger.info(f"[ML API] Retrieved cost metrics for {total_providers} providers")
        return LLMCostsResponse(
            costs=costs,
            total_providers=total_providers
        )
    except Exception as e:
        logger.error(f"[ML API] Failed to get LLM costs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get LLM costs: {str(e)}")

@app.post("/llm/compare", response_model=LLMCompareResponse)
async def compare_llm_models(request: LLMCompareRequest):
    """
    Compare responses from different external LLM models.

    Generates responses to the same prompt using multiple models and
    returns a comparison of their outputs, costs, and performance metrics.
    """
    try:
        comparisons = await prompt_controller.compare_external_llm_models(
            request.prompt_text, request.providers_models
        )
        logger.info(f"[ML API] Compared {len(comparisons)} LLM models")
        return LLMCompareResponse(
            comparisons=comparisons,
            prompt_text=request.prompt_text,
            total_models=len(comparisons)
        )
    except Exception as e:
        logger.error(f"[ML API] Failed to compare LLM models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to compare LLM models: {str(e)}")

@app.get("/llm/status", response_model=LLMStatusResponse)
async def get_llm_status():
    """
    Get current LLM system status and configuration.

    Returns comprehensive information about the current LLM setup,
    including which mode is active, available providers, and system health.
    """
    try:
        status = prompt_controller.get_llm_status()
        logger.info("[ML API] Retrieved LLM system status")
        return LLMStatusResponse(status=status)
    except Exception as e:
        logger.error(f"[ML API] Failed to get LLM status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get LLM status: {str(e)}")

# Analytics endpoints
@app.get("/analytics/dashboard")
async def get_analytics_dashboard(learner_id: Optional[str] = None, include_historical: bool = False):
    """
    Get comprehensive analytics dashboard data.

    Provides real-time learning progress dashboards with:
    - Skill gap analysis and recommendations
    - Learning bottleneck detection
    - Performance prediction models
    - Learning insights and actionable recommendations
    - System health monitoring

    Query Parameters:
    - learner_id: Optional specific learner to analyze (system-wide if not provided)
    - include_historical: Whether to include historical trend data
    """
    try:
        from models.analytics import AnalyticsDashboardRequest
        request = AnalyticsDashboardRequest(
            learner_id=learner_id,
            include_historical=include_historical
        )
        response = analytics_controller.get_analytics_dashboard(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to get analytics dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics dashboard: {str(e)}")

@app.get("/analytics/skill-gaps")
async def get_skill_gaps(learner_id: str):
    """
    Get skill gap analysis and recommendations.

    Analyzes current skill levels against required competencies and provides:
    - Identification of skill gaps
    - Severity assessment
    - Improvement recommendations
    - Estimated time to close gaps
    """
    try:
        from models.analytics import SkillGapAnalysisRequest
        request = SkillGapAnalysisRequest(learner_id=learner_id, include_recommendations=True)
        response = analytics_controller.analyze_skill_gaps(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to get skill gaps: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get skill gaps: {str(e)}")

@app.get("/analytics/bottlenecks")
async def get_bottlenecks(learner_id: Optional[str] = None):
    """
    Detect and report learning bottlenecks.

    Identifies obstacles in the learning process including:
    - Conceptual difficulties
    - Practical application issues
    - Motivational barriers
    - Resource constraints
    - Technical problems
    """
    try:
        from models.analytics import BottleneckDetectionRequest
        request = BottleneckDetectionRequest(include_system_wide=True, learner_id=learner_id)
        response = analytics_controller.detect_bottlenecks(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to detect bottlenecks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to detect bottlenecks: {str(e)}")

@app.get("/analytics/predictions")
async def get_performance_predictions(learner_id: str, horizon_days: int = 30):
    """
    Get performance predictions for future learning outcomes.

    Uses historical data to forecast:
    - Future performance levels
    - Learning trajectory
    - Risk factors
    - Improvement recommendations
    """
    try:
        from models.analytics import PerformancePredictionRequest
        request = PerformancePredictionRequest(
            learner_id=learner_id,
            prediction_horizon_days=horizon_days,
            include_factors=True
        )
        response = analytics_controller.predict_performance(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to get performance predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance predictions: {str(e)}")

@app.get("/analytics/insights")
async def get_learning_insights(learner_id: Optional[str] = None):
    """
    Get learning insights and actionable recommendations.

    Provides prioritized insights including:
    - Skill-based recommendations
    - Performance improvement suggestions
    - System optimization insights
    - Personalized learning strategies
    """
    try:
        from models.analytics import LearningInsightsRequest
        request = LearningInsightsRequest(learner_id=learner_id)
        response = analytics_controller.generate_learning_insights(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to get learning insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning insights: {str(e)}")

@app.get("/analytics/health")
async def get_system_health():
    """
    Monitor system health and performance.

    Returns comprehensive health metrics including:
    - Component health scores
    - Performance metrics
    - Error rates
    - Resource utilization
    - System alerts
    """
    try:
        from models.analytics import SystemHealthRequest
        request = SystemHealthRequest(include_detailed_metrics=True)
        response = analytics_controller.monitor_system_health(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to get system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@app.get("/analytics/status")
async def get_analytics_status():
    """
    Get analytics system status and configuration.

    Returns information about:
    - Analytics service status
    - Available capabilities
    - Data quality metrics
    - System configuration
    """
    try:
        status = analytics_controller.get_analytics_status()
        return status
    except Exception as e:
        logger.error(f"[ML API] Failed to get analytics status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics status: {str(e)}")

# Persistence endpoints
@app.post("/persistence/save", response_model=SaveStateResponse)
async def save_learning_state(request: SaveStateRequest):
    """
    Save current learning state to persistent storage.

    Allows saving different types of learning states:
    - llm_model: PyTorch model checkpoints and training state
    - meta_learning: Meta-learning parameters and strategies
    - history: Interaction history and patterns
    - curriculum: Curriculum progress and learner state
    - reward_system: Reward weights and metrics
    - analytics: Analytics data and insights
    - feedback: User feedback and preferences
    - complete_system: All components combined

    The state is versioned and can be rolled back to previous versions.
    """
    start_time = time.time()
    try:
        logger.info(f"💾 Starting save operation for state type: {request.state_type}")

        # Start progress tracking for large saves
        tracker_id = start_progress_tracker(f"save_{request.state_type}")

        # Log persistence operation start
        log_persistence_operation("save", request.state_type, "started", {
            'description': request.description,
            'include_metadata': request.include_metadata
        })

        result = await persistence_controller.save_learning_state(request)

        # Complete progress tracking
        duration = time.time() - start_time
        complete_progress(tracker_id, success=True, message=f"Saved {request.state_type}")

        # Log successful persistence operation
        log_persistence_operation("save", request.state_type, "success", {
            'version': result.version_id,
            'size_mb': getattr(result, 'size_mb', None),
            'duration': duration
        })

        logger.info(f"✅ Successfully saved {request.state_type} state | Version: {result.version_id} | "
                   f"Duration: {duration:.3f}s")
        return result
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Failed to save learning state: {str(e)} | Duration: {duration:.3f}s")

        # Log failed persistence operation
        log_persistence_operation("save", request.state_type, "failed", {
            'error': str(e),
            'duration': duration
        })

        raise HTTPException(status_code=500, detail=f"Failed to save learning state: {str(e)}")

@app.post("/persistence/load", response_model=LoadStateResponse)
async def load_learning_state(request: LoadStateRequest):
    """
    Load learning state from persistent storage.

    Loads a specific version of a learning state (or latest if not specified).
    The loaded state is applied to the current running system.
    """
    try:
        return await persistence_controller.load_learning_state(request)
    except Exception as e:
        logger.error(f"[Persistence API] Failed to load learning state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load learning state: {str(e)}")

@app.get("/persistence/versions", response_model=ListVersionsResponse)
async def list_state_versions(state_type: LearningStateType, instance_id: Optional[str] = None, limit: int = 10):
    """
    List available versions for a learning state type.

    Returns chronological list of versions with metadata including:
    - Version ID and timestamp
    - Description (if provided)
    - Performance metrics
    - Data size information
    """
    try:
        request = ListVersionsRequest(state_type=state_type, instance_id=instance_id, limit=limit)
        return await persistence_controller.list_state_versions(request)
    except Exception as e:
        logger.error(f"[Persistence API] Failed to list state versions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list state versions: {str(e)}")

@app.post("/persistence/export", response_model=ExportStateResponse)
async def export_learning_state(request: ExportStateRequest):
    """
    Export learning state for backup or migration.

    Exports selected components to JSON format. Can include:
    - Model checkpoints
    - Training history
    - Meta-learning state
    - Curriculum progress
    - System configuration

    Supports compression for large exports.
    """
    try:
        return await persistence_controller.export_learning_state(request)
    except Exception as e:
        logger.error(f"[Persistence API] Failed to export learning state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export learning state: {str(e)}")

@app.post("/persistence/import", response_model=ImportStateResponse)
async def import_learning_state(request: ImportStateRequest):
    """
    Import learning state from export.

    Imports previously exported learning state data.
    Supports validation and selective component import.
    """
    try:
        return await persistence_controller.import_learning_state(request)
    except Exception as e:
        logger.error(f"[Persistence API] Failed to import learning state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to import learning state: {str(e)}")

@app.post("/persistence/rollback", response_model=RollbackResponse)
async def rollback_state(request: RollbackRequest):
    """
    Rollback to a previous version of learning state.

    WARNING: This will replace the current state with an older version.
    Requires explicit confirmation via confirm_rollback flag.
    """
    try:
        return await persistence_controller.rollback_state(request)
    except Exception as e:
        logger.error(f"[Persistence API] Failed to rollback state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rollback state: {str(e)}")

@app.post("/persistence/backup", response_model=BackupResponse)
async def create_backup(request: BackupRequest):
    """
    Create a backup of current learning state.

    Creates a compressed backup of selected components for disaster recovery.
    Backups are stored in the database and can be restored later.
    """
    try:
        return await persistence_controller.create_backup(request)
    except Exception as e:
        logger.error(f"[Persistence API] Failed to create backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")

@app.post("/persistence/restore", response_model=RestoreResponse)
async def restore_backup(request: RestoreRequest):
    """
    Restore from a backup.

    WARNING: This will overwrite current state with backup data.
    Requires explicit confirmation via confirm_restore flag.
    """
    try:
        return await persistence_controller.restore_backup(request)
    except Exception as e:
        logger.error(f"[Persistence API] Failed to restore backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to restore backup: {str(e)}")

@app.get("/persistence/distributed/status", response_model=DistributedStatusResponse)
async def get_distributed_status():
    """
    Get distributed learning system status.

    Returns information about:
    - Current instance role (coordinator/follower)
    - Active instances in the cluster
    - Shared state synchronization status
    - Held locks and coordination state
    """
    try:
        return await persistence_controller.get_distributed_status()
    except Exception as e:
        logger.error(f"[Persistence API] Failed to get distributed status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get distributed status: {str(e)}")

@app.post("/persistence/distributed/sync", response_model=SyncResponse)
async def sync_shared_state(request: SyncRequest):
    """
    Manually trigger shared state synchronization.

    Forces synchronization of critical learning state across distributed instances.
    Can be used to ensure consistency after network issues or manual state changes.
    """
    try:
        return await persistence_controller.sync_shared_state(request)
    except Exception as e:
        logger.error(f"[Persistence API] Failed to sync shared state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to sync shared state: {str(e)}")

@app.post("/persistence/reset")
async def reset_all_learning_data(confirm_reset: bool = False):
    """
    Reset all learning data, history, analytics, and state.

    WARNING: This will permanently delete all learning progress, history, analytics data,
    and reset all components to their initial state. This action cannot be undone.

    Query Parameters:
    - confirm_reset: Must be set to true to confirm the reset operation
    """
    try:
        logger.info(f"[Persistence API] Reset request received, confirm_reset={confirm_reset}")

        result = await persistence_controller.reset_all_data(confirm_reset)

        if result["success"]:
            logger.info("[Persistence API] All learning data reset successfully")
            return {
                "success": True,
                "message": "All learning data has been reset successfully",
                "reset_results": result["reset_results"],
                "timestamp": result["timestamp"]
            }
        else:
            logger.warning(f"[Persistence API] Partial reset completed: {result['message']}")
            return {
                "success": False,
                "message": result["message"],
                "reset_results": result["reset_results"],
                "timestamp": result["timestamp"]
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Persistence API] Failed to reset learning data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset learning data: {str(e)}")

# Feedback integration endpoints
@app.post("/feedback/submit")
async def submit_feedback(request: SubmitFeedbackRequest):
    """
    Submit user feedback on an AI response.

    Allows users to provide ratings, comments, and corrections on AI-generated responses.
    Feedback is used to improve future responses and personalize the learning experience.
    """
    try:
        logger.info(f"💬 Processing feedback submission from user {request.user_id}")

        # Log feedback processing start
        log_feedback_processing(request.user_id, "feedback_submission", {
            'response_id': request.response_id,
            'feedback_type': request.feedback_type,
            'category': getattr(request, 'category', None),
            'rating': getattr(request, 'rating', None)
        })

        response = feedback_controller.submit_feedback(request)

        # Log successful feedback processing
        log_feedback_processing(request.user_id, "feedback_processed", {
            'response_id': request.response_id,
            'feedback_type': request.feedback_type,
            'processed': True
        })

        logger.info(f"✅ Feedback submitted successfully for user {request.user_id}")
        return response
    except Exception as e:
        logger.error(f"❌ Failed to submit feedback: {str(e)}")

        # Log failed feedback processing
        log_feedback_processing(request.user_id, "feedback_failed", {
            'response_id': request.response_id,
            'error': str(e)
        })

        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@app.get("/feedback/history")
async def get_feedback_history(user_id: str, limit: int = 50, offset: int = 0,
                              category: Optional[str] = None, feedback_type: Optional[str] = None):
    """
    Get user's feedback history.

    Returns chronological list of user's feedback submissions with filtering options.

    Query Parameters:
    - user_id: User identifier (required)
    - limit: Maximum number of items to return (default: 50)
    - offset: Number of items to skip (default: 0)
    - category: Filter by feedback category
    - feedback_type: Filter by feedback type
    """
    try:
        from models.feedback import GetFeedbackHistoryRequest, FeedbackCategory, FeedbackType

        # Build request object
        request = GetFeedbackHistoryRequest(
            user_id=user_id,
            limit=limit,
            offset=offset
        )

        # Add optional filters
        if category:
            try:
                request.category_filter = FeedbackCategory(category.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

        if feedback_type:
            try:
                request.feedback_type_filter = FeedbackType(feedback_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid feedback type: {feedback_type}")

        response = feedback_controller.get_feedback_history(request)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ML API] Failed to get feedback history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback history: {str(e)}")

@app.post("/feedback/correct")
async def submit_correction(request: SubmitCorrectionRequest):
    """
    Submit a correction for a wrong AI response.

    Allows users to provide corrected versions of AI responses that were inaccurate.
    Corrections are used to improve the system's learning and response quality.
    """
    try:
        response = feedback_controller.submit_correction(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to submit correction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit correction: {str(e)}")

@app.get("/feedback/preferences")
async def get_user_preferences(user_id: str, include_history: bool = False):
    """
    Get learned user preferences.

    Returns the system's understanding of user preferences based on their feedback history.

    Query Parameters:
    - user_id: User identifier (required)
    - include_history: Whether to include feedback history in response (default: false)
    """
    try:
        from models.feedback import GetPreferencesRequest
        request = GetPreferencesRequest(user_id=user_id, include_history=include_history)
        response = feedback_controller.get_user_preferences(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to get user preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get user preferences: {str(e)}")

@app.post("/feedback/rate")
async def rate_response(request: RateResponseRequest):
    """
    Rate a response and update preferences.

    Allows users to quickly rate responses on a 1-5 scale, which updates their preference profile.
    """
    try:
        response = feedback_controller.rate_response(request)
        return response
    except Exception as e:
        logger.error(f"[ML API] Failed to rate response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rate response: {str(e)}")

@app.get("/feedback/analytics")
async def get_feedback_analytics(user_id: Optional[str] = None, response_id: Optional[str] = None):
    """
    Get feedback analytics and insights.

    Provides comprehensive analytics about feedback patterns, user engagement,
    and collaborative learning insights.

    Query Parameters:
    - user_id: Optional user-specific analytics
    - response_id: Optional response-specific analysis
    """
    try:
        analytics = feedback_controller.get_feedback_analytics(user_id, response_id)
        return analytics
    except Exception as e:
        logger.error(f"[ML API] Failed to get feedback analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback analytics: {str(e)}")

@app.get("/feedback/insights")
async def get_feedback_insights(category_filter: Optional[str] = None):
    """
    Get feedback insights and recommendations.

    Provides actionable insights based on feedback patterns and collaborative learning data.

    Query Parameters:
    - category_filter: Optional category to filter insights
    """
    try:
        insights = feedback_controller.get_feedback_insights(category_filter)
        return insights
    except Exception as e:
        logger.error(f"[ML API] Failed to get feedback insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feedback insights: {str(e)}")

@app.post("/feedback/error-report")
async def submit_error_report(request: SubmitErrorReportRequest):
    """
    Submit an error report from the frontend.

    Allows users to report system errors, bugs, and issues encountered during use.
    Error reports are stored as feedback and can be used for system improvement.

    Request Body:
    - user_id: User identifier
    - session_id: Optional session identifier
    - error_message: Error message text
    - stack_trace: Optional stack trace information
    - context: Optional context information about when the error occurred
    - component: Optional component where the error occurred
    - severity: Error severity level (low, medium, high, critical)
    - user_agent: Optional browser/client information
    - url: Optional URL where the error occurred
    """
    try:
        logger.info(f"🐛 Processing error report submission from user {request.user_id}")

        response = feedback_controller.submit_error_report(request)

        logger.info(f"✅ Error report submitted successfully for user {request.user_id}")
        return response
    except Exception as e:
        logger.error(f"❌ Failed to submit error report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit error report: {str(e)}")

# Curriculum-driven learning endpoints
class CurriculumStartRequest(BaseModel):
    user_prompt: str
    num_tasks: int = 5

class CurriculumStartResponse(BaseModel):
    curriculum_info: Dict[str, Any]
    message: str

class CurriculumTaskResponse(BaseModel):
    task_info: Dict[str, Any]
    task_sequence_position: int
    total_tasks: int

class CurriculumProcessRequest(BaseModel):
    response: str
    task_info: Dict[str, Any]
    user_id: Optional[str] = None

class CurriculumProcessResponse(BaseModel):
    result: Dict[str, Any]
    curriculum_progress: Dict[str, Any]
    next_task_available: bool

class CurriculumStatusResponse(BaseModel):
    active: bool
    current_task: Optional[int] = None
    total_tasks: Optional[int] = None
    completion_percentage: Optional[float] = None
    remaining_tasks: Optional[int] = None
    skills_covered: Optional[List[str]] = None
    next_task: Optional[Dict[str, Any]] = None

class RLTrainingStatusResponse(BaseModel):
    is_training: bool
    episodes: int
    total_steps: int
    policy_updates: int
    current_avg_reward: float
    current_success_rate: float
    curriculum_difficulty: str
    completed_skills: int
    available_skills: int
    recent_rewards: List[float]
    recent_success_rates: List[float]

@app.post("/curriculum/start", response_model=CurriculumStartResponse)
async def start_curriculum_learning(request: CurriculumStartRequest, user_id: Optional[str] = None):
    """
    Start curriculum-driven learning from a user prompt.

    Analyzes the user's prompt to create a personalized learning curriculum
    with progressive difficulty tasks.
    """
    try:
        logger.info(f"📚 Starting curriculum learning for user {user_id} with prompt: {request.user_prompt[:50]}...")

        curriculum_info = prompt_controller.start_curriculum_learning(
            request.user_prompt, request.num_tasks
        )

        logger.info(f"✅ Curriculum learning started: {len(curriculum_info.get('skills_covered', []))} skills, "
                   f"{curriculum_info.get('total_tasks', 0)} tasks")
        return CurriculumStartResponse(
            curriculum_info=curriculum_info,
            message="Curriculum learning started successfully"
        )
    except Exception as e:
        logger.error(f"❌ Failed to start curriculum learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start curriculum learning: {str(e)}")

@app.get("/curriculum/task", response_model=CurriculumTaskResponse)
async def get_next_curriculum_task(user_id: Optional[str] = None):
    """
    Get the next task in the curriculum sequence.

    Returns the next task to work on in the curriculum, or None if curriculum is complete.
    """
    try:
        task = prompt_controller.get_next_curriculum_task()

        if task is None:
            return CurriculumTaskResponse(
                task_info={},
                task_sequence_position=0,
                total_tasks=0
            )

        status = prompt_controller.get_curriculum_status()

        return CurriculumTaskResponse(
            task_info=task,
            task_sequence_position=status.get('current_task', 0),
            total_tasks=status.get('total_tasks', 0)
        )
    except Exception as e:
        logger.error(f"❌ Failed to get next curriculum task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get next curriculum task: {str(e)}")

@app.post("/curriculum/process", response_model=CurriculumProcessResponse)
async def process_curriculum_response(request: CurriculumProcessRequest):
    """
    Process a response to a curriculum task.

    Evaluates the response, updates progress, and provides feedback.
    """
    try:
        logger.info(f"🔄 Processing curriculum response for task: {request.task_info.get('skill_id', 'unknown')}")

        result = prompt_controller.process_curriculum_response(
            request.response, request.task_info, request.user_id
        )

        status = prompt_controller.get_curriculum_status()
        next_task_available = status.get('next_task') is not None

        logger.info(f"✅ Curriculum response processed: performance={result.get('curriculum_progress', {}).get('performance_score', 0):.3f}")
        return CurriculumProcessResponse(
            result=result,
            curriculum_progress=result.get('curriculum_progress', {}),
            next_task_available=next_task_available
        )
    except Exception as e:
        logger.error(f"❌ Failed to process curriculum response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process curriculum response: {str(e)}")

@app.get("/curriculum/status", response_model=CurriculumStatusResponse)
async def get_curriculum_status():
    """
    Get current curriculum learning status.

    Returns information about the current state of curriculum learning.
    """
    try:
        status = prompt_controller.get_curriculum_status()
        return CurriculumStatusResponse(**status)
    except Exception as e:
        logger.error(f"❌ Failed to get curriculum status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get curriculum status: {str(e)}")

@app.post("/curriculum/stop")
async def stop_curriculum_learning():
    """
    Stop curriculum-driven learning.

    Ends the current curriculum session.
    """
    try:
        prompt_controller.stop_curriculum_learning()
        logger.info("🛑 Curriculum learning stopped")
        return {"message": "Curriculum learning stopped successfully"}
    except Exception as e:
        logger.error(f"❌ Failed to stop curriculum learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop curriculum learning: {str(e)}")

@app.post("/rl/start")
async def start_rl_training():
    """
    Start RL training for curriculum learning.

    Begins autonomous RL training to improve curriculum task performance.
    """
    try:
        prompt_controller.start_rl_training()
        logger.info("🤖 RL training started")
        return {"message": "RL training started successfully"}
    except Exception as e:
        logger.error(f"❌ Failed to start RL training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start RL training: {str(e)}")

@app.post("/rl/stop")
async def stop_rl_training():
    """
    Stop RL training.

    Ends the RL training process.
    """
    try:
        prompt_controller.stop_rl_training()
        logger.info("🤖 RL training stopped")
        return {"message": "RL training stopped successfully"}
    except Exception as e:
        logger.error(f"❌ Failed to stop RL training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop RL training: {str(e)}")

@app.get("/rl/status", response_model=RLTrainingStatusResponse)
async def get_rl_training_status():
    """
    Get RL training status.

    Returns current RL training metrics and progress.
    """
    try:
        status = prompt_controller.get_rl_training_status()
        return RLTrainingStatusResponse(**status)
    except Exception as e:
        logger.error(f"❌ Failed to get RL training status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get RL training status: {str(e)}")

# Health check endpoint - useful for monitoring and load balancers
@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint for monitoring system status.

    This demonstrates:
    1. How to implement health checks in microservices
    2. System status reporting
    3. Basic monitoring integration
    4. Performance and health metrics

    Educational Notes:
    - Health checks should be lightweight
    - Include key system metrics
    - Return appropriate HTTP status codes
    - Log health status for monitoring
    """
    start_time = time.time()

    try:
        # Gather system health metrics
        metrics = prompt_controller.get_learning_metrics()
        system_status = "healthy"

        # Check various system components
        health_checks = {
            'learning_system': metrics.get('total_interactions', 0) >= 0,  # Basic connectivity check
            'external_llm': external_llm_service is not None if get_external_llm_config().enabled else True,
            'persistence': True,  # Assume persistence is working if we reach this point
            'feedback_system': True
        }

        # Determine overall health
        if not all(health_checks.values()):
            system_status = "degraded"
            failed_components = [k for k, v in health_checks.items() if not v]
            logger.warning(f"⚠️  Health check: System degraded | Failed components: {failed_components}")
        else:
            logger.info("🟢 Health check: System healthy")

        # Log system health
        log_system_health("api_server", system_status, {
            'uptime': time.time() - start_time,  # Rough uptime estimate
            'total_interactions': metrics.get('total_interactions', 0),
            'success_rate': metrics.get('success_rate', 0.0),
            'components_status': health_checks
        })

        response = {
            "status": system_status,
            "timestamp": time.time(),
            "version": api_config.version,
            "metrics": {
                "total_interactions": metrics.get('total_interactions', 0),
                "success_rate": metrics.get('success_rate', 0.0),
                "components": health_checks
            }
        }

        return response

    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        log_system_health("api_server", "critical", {"error": str(e)})
        # Return degraded status but don't crash
        return {
            "status": "critical",
            "timestamp": time.time(),
            "version": api_config.version,
            "error": str(e)
        }

# Application startup event
@app.on_event("startup")
async def startup_event():
    """
    Code to run when the application starts up.

    This demonstrates:
    1. FastAPI lifecycle events
    2. Initialization tasks
    3. Resource setup
    4. Startup logging

    Educational Notes:
    - Use startup events for one-time initialization
    - Log important startup information
    - Validate system state before accepting requests
    """
    startup_start = time.time()

    logger.info("🚀 Starting LLM Learning System API")
    logger.info(f"📋 Environment: {config_manager.config.environment}")
    logger.info(f"🔧 Debug mode: {api_config.debug}")
    logger.info(f"🌐 CORS origins: {api_config.cors_origins}")
    logger.info(f"🤖 LLM config: actions={llm_config.num_actions}, alpha={llm_config.alpha}, epsilon={llm_config.epsilon}")

    # Initialize persistence service
    try:
        await persistence_service.initialize()
        log_service_initialization("PersistenceService", "success", {"startup_time": time.time() - startup_start})
        logger.info("💾 Persistence service initialized successfully")
    except Exception as e:
        log_service_initialization("PersistenceService", "failed", {"error": str(e)})
        logger.error(f"Failed to initialize persistence service: {e}")
        # Continue startup but log the error

    # Log system health at startup
    log_system_health("application", "healthy", {
        "startup_time": time.time() - startup_start,
        "environment": config_manager.config.environment,
        "debug_mode": api_config.debug
    })

    startup_duration = time.time() - startup_start
    logger.info(f"✅ Application startup complete in {startup_duration:.3f}s")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Code to run when the application shuts down.

    This demonstrates:
    1. Graceful shutdown handling
    2. Resource cleanup
    3. Final logging
    """
    logger.info("🛑 Shutting down LLM Learning System API")

    # Get final metrics before shutdown
    try:
        final_metrics = prompt_controller.get_learning_metrics()
        log_system_health("application", "shutting_down", {
            "final_interactions": final_metrics.get('total_interactions', 0),
            "final_success_rate": final_metrics.get('success_rate', 0.0)
        })
    except Exception as e:
        logger.warning(f"Could not get final metrics during shutdown: {e}")

    # Close persistence service connections
    try:
        await persistence_service.close()
        logger.info("💾 Persistence service connections closed")
    except Exception as e:
        logger.error(f"Error closing persistence service: {e}")

    logger.info("👋 Application shutdown complete")

if __name__ == "__main__":
    """
    Main entry point for running the application with Uvicorn.

    This section teaches you:
    1. How to run FastAPI with Uvicorn ASGI server
    2. Server configuration options
    3. Development vs production settings
    4. Command-line application startup

    Educational Notes:
    - Uvicorn is the recommended ASGI server for FastAPI
    - Host "0.0.0.0" makes the server accessible from other machines
    - reload=True is great for development but not for production
    - Consider using gunicorn with uvicorn workers in production
    """
    import uvicorn

    logger.info("Starting server with Uvicorn...")

    # Use configuration for server settings
    uvicorn.run(
        "app:app",  # module:variable format
        host=api_config.host,
        port=api_config.port,
        reload=api_config.debug,  # Auto-reload in development
        log_level="info",
        access_log=True  # Log all requests
    )