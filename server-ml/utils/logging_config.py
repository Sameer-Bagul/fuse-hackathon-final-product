"""
Logging Configuration Utility - Educational Implementation

This module teaches you how to implement proper logging in Python applications.
Logging is crucial for debugging, monitoring, and understanding what's happening
in your application during development and production.

Why logging matters:
- Debugging: Track down bugs by seeing what your code is doing
- Monitoring: Keep track of application health and performance
- Auditing: Record important events for compliance
- Learning: Understand the flow of execution

Design Patterns Used:
- Singleton Pattern: Ensure consistent logging configuration across the app
- Factory Pattern: Create different logger instances for different modules
- Configuration Pattern: Centralized logging setup

Best Practices Demonstrated:
- Different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Structured logging with context
- Log rotation to prevent disk space issues
- Proper formatting for readability
- Environment-based configuration
"""

import logging
import logging.handlers
import sys
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }

        # Add extra fields if present
        if hasattr(record, 'learning_data'):
            log_entry['learning_data'] = record.learning_data

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class LoggingConfig:
    """
    Centralized logging configuration for the LLM Learning System.

    This class demonstrates how to set up professional-grade logging that:
    1. Scales from development to production
    2. Provides different detail levels
    3. Handles log rotation automatically
    4. Works across multiple modules

    Educational Notes:
    - Logging configuration should be done once at application startup
    - Different environments need different logging levels
    - Log files should rotate to prevent filling up disk space
    - Structured logging makes logs easier to parse and analyze
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern - ensure only one logging configuration exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logging configuration if not already done."""
        if not self._initialized:
            self._progress_trackers = {}  # Track progress for long-running operations
            self._setup_logging()
            self._initialized = True

    def _setup_logging(self):
        """Set up comprehensive logging configuration."""

        # Clear any existing handlers to avoid duplicates
        logging.getLogger().handlers.clear()

        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Capture all levels, filter at handler level

        # Define log formats
        # Human-readable format for console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # JSON format for structured logging
        json_formatter = JSONFormatter()

        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Show INFO and above in console
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # File handler with rotation for production logs
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        # Rotating file handler - keeps last 5 log files, 10MB each
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "llm_learning.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)

        # JSON structured log file for analysis
        json_file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "llm_learning.jsonl",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3
        )
        json_file_handler.setLevel(logging.INFO)  # Structured logs for INFO and above
        json_file_handler.setFormatter(json_formatter)
        root_logger.addHandler(json_file_handler)

        # Error-only file for quick problem identification
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)  # Only errors and critical
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'File: %(filename)s:%(lineno)d\n'
            'Function: %(funcName)s\n'
            'Process: %(process)d\n'
            'Thread: %(thread)d\n'
            '---\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)

        # Specialized handlers for different log types
        self._setup_specialized_handlers(log_dir, console_formatter, json_formatter)

        # Log the initialization
        logger = logging.getLogger(__name__)
        logger.info("🚀 Logging system initialized successfully")
        logger.info(f"📁 Log files will be written to: {log_dir}")
        logger.info("📊 Structured JSON logs enabled for analysis")
        logger.info("🎯 Specialized logging handlers configured")

    def _setup_specialized_handlers(self, log_dir: Path, console_formatter: logging.Formatter, json_formatter: JSONFormatter):
        """Set up specialized handlers for different log categories."""

        # Performance logs
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / "performance.log",
            maxBytes=20*1024*1024,
            backupCount=5
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(console_formatter)
        perf_handler.addFilter(lambda record: record.name == "performance")
        logging.getLogger("performance").addHandler(perf_handler)
        logging.getLogger("performance").setLevel(logging.INFO)

        # API logs
        api_handler = logging.handlers.RotatingFileHandler(
            log_dir / "api.log",
            maxBytes=20*1024*1024,
            backupCount=5
        )
        api_handler.setLevel(logging.INFO)
        api_handler.setFormatter(console_formatter)
        api_handler.addFilter(lambda record: record.name == "api")
        logging.getLogger("api").addHandler(api_handler)
        logging.getLogger("api").setLevel(logging.INFO)

        # Learning events logs
        learning_handler = logging.handlers.RotatingFileHandler(
            log_dir / "learning.log",
            maxBytes=20*1024*1024,
            backupCount=5
        )
        learning_handler.setLevel(logging.INFO)
        learning_handler.setFormatter(console_formatter)
        learning_handler.addFilter(lambda record: record.name == "learning")
        logging.getLogger("learning").addHandler(learning_handler)
        logging.getLogger("learning").setLevel(logging.INFO)

        # External LLM logs
        llm_handler = logging.handlers.RotatingFileHandler(
            log_dir / "external_llm.log",
            maxBytes=20*1024*1024,
            backupCount=5
        )
        llm_handler.setLevel(logging.INFO)
        llm_handler.setFormatter(console_formatter)
        llm_handler.addFilter(lambda record: record.name == "external_llm")
        logging.getLogger("external_llm").addHandler(llm_handler)
        logging.getLogger("external_llm").setLevel(logging.INFO)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger for a specific module.

        Args:
            name: Usually __name__ of the calling module

        Returns:
            Configured logger instance

        Educational Notes:
        - Always use __name__ as the logger name for proper hierarchy
        - This creates a logger that inherits the root configuration
        - Different modules can have different log levels if needed
        """
        return logging.getLogger(name)

    def set_module_level(self, module_name: str, level: str):
        """
        Set logging level for a specific module.

        Args:
            module_name: Name of the module (e.g., 'controllers.prompt_controller')
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

        Educational Notes:
        - Sometimes you want different verbosity for different parts of your app
        - For example, set database module to WARNING but API to DEBUG
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if level.upper() in level_map:
            logger = logging.getLogger(module_name)
            logger.setLevel(level_map[level.upper()])
            self.get_logger(__name__).info(f"Set {module_name} logging level to {level}")
        else:
            self.get_logger(__name__).warning(f"Invalid log level: {level}")

    def log_performance_metric(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics in a structured way.

        Args:
            operation: Name of the operation being measured
            duration: Time taken in seconds
            metadata: Additional context information

        Educational Notes:
        - Performance logging helps identify bottlenecks
        - Structured logging makes it easier to analyze with tools
        - Include relevant context for debugging
        """
        logger = self.get_logger("performance")

        if metadata:
            # Structured logging with context
            logger.info(f"PERFORMANCE: {operation} completed in {duration:.3f}s", extra={
                'operation': operation,
                'duration': duration,
                'metadata': metadata
            })
        else:
            logger.info(f"PERFORMANCE: {operation} completed in {duration:.3f}s")

    def log_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """
        Log API requests for monitoring and debugging.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            status_code: HTTP status code
            duration: Request duration in seconds

        Educational Notes:
        - API logging helps monitor usage patterns
        - Status codes help identify error rates
        - Duration helps spot performance issues
        """
        logger = self.get_logger("api")

        if status_code >= 400:
            logger.warning(f"API {method} {endpoint} -> {status_code} ({duration:.3f}s)")
        else:
            logger.info(f"API {method} {endpoint} -> {status_code} ({duration:.3f}s)")

    def log_learning_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log learning-specific events for analysis.

        Args:
            event_type: Type of learning event (e.g., 'prompt_processed', 'model_updated')
            details: Event-specific details

        Educational Notes:
        - Domain-specific logging helps track learning progress
        - Structured data enables automated analysis
        - This is crucial for understanding how your ML system behaves
        """
        logger = self.get_logger("learning")

        # Add timestamp and event type to details
        enhanced_details = {
            'event_type': event_type,
            'timestamp': logging.LogRecord('', 0, '', 0, '', (), None).getMessage(),  # Current time
            **details
        }

        logger.info(f"LEARNING EVENT: {event_type}", extra={'learning_data': enhanced_details})

    def log_service_initialization(self, service_name: str, status: str, details: Optional[Dict[str, Any]] = None):
        """
        Log service initialization details and status.

        Args:
            service_name: Name of the service being initialized
            status: Initialization status ('starting', 'success', 'failed', 'warning')
            details: Additional initialization details
        """
        logger = self.get_logger("service_init")

        status_icons = {
            'starting': '🚀',
            'success': '✅',
            'failed': '❌',
            'warning': '⚠️'
        }

        icon = status_icons.get(status, 'ℹ️')
        message = f"{icon} Service '{service_name}' initialization: {status.upper()}"

        if details:
            message += f" - {details}"

        if status == 'failed':
            logger.error(message)
        elif status == 'warning':
            logger.warning(message)
        else:
            logger.info(message)

    def log_system_health(self, component: str, health_status: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Log system health and performance indicators.

        Args:
            component: System component name
            health_status: Health status ('healthy', 'degraded', 'critical', 'unknown')
            metrics: Performance metrics
        """
        logger = self.get_logger("system_health")

        status_icons = {
            'healthy': '🟢',
            'degraded': '🟡',
            'critical': '🔴',
            'unknown': '⚪'
        }

        icon = status_icons.get(health_status, '⚪')
        message = f"{icon} System Health - {component}: {health_status.upper()}"

        if metrics:
            metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
            message += f" | Metrics: {metrics_str}"

        if health_status in ['critical', 'degraded']:
            logger.warning(message)
        else:
            logger.info(message)

    def log_curriculum_status(self, learner_id: str, status: str, details: Optional[Dict[str, Any]] = None):
        """
        Log curriculum and meta-learning status updates.

        Args:
            learner_id: Learner identifier
            status: Status type ('progress_update', 'skill_learned', 'difficulty_adjusted', etc.)
            details: Status details
        """
        logger = self.get_logger("curriculum")

        message = f"📚 Curriculum Status - Learner {learner_id}: {status.upper()}"

        if details:
            if 'progress' in details:
                message += f" | Progress: {details['progress']:.1f}%"
            if 'current_skill' in details:
                message += f" | Current Skill: {details['current_skill']}"
            if 'difficulty' in details:
                message += f" | Difficulty: {details['difficulty']}"

        logger.info(message)

    def log_external_llm_interaction(self, provider: str, model: str, operation: str,
                                   cost: Optional[float] = None, latency: Optional[float] = None,
                                   tokens: Optional[Dict[str, int]] = None, error: Optional[str] = None):
        """
        Log external LLM interactions and costs.

        Args:
            provider: LLM provider name
            model: Model name
            operation: Operation type ('generate', 'embed', 'moderate', etc.)
            cost: Cost in USD
            latency: Response latency in seconds
            tokens: Token usage breakdown
            error: Error message if any
        """
        logger = self.get_logger("external_llm")

        icon = '❌' if error else '🤖'
        message = f"{icon} External LLM - {provider}/{model} | Operation: {operation.upper()}"

        if cost is not None:
            message += f" | Cost: ${cost:.4f}"
        if latency is not None:
            message += f" | Latency: {latency:.3f}s"
        if tokens:
            total_tokens = tokens.get('total', sum(tokens.values()))
            message += f" | Tokens: {total_tokens}"

        if error:
            logger.error(f"{message} | Error: {error}")
        else:
            logger.info(message)

    def log_feedback_processing(self, user_id: str, feedback_type: str, details: Optional[Dict[str, Any]] = None):
        """
        Log user feedback processing.

        Args:
            user_id: User identifier
            feedback_type: Type of feedback ('rating', 'correction', 'comment', etc.)
            details: Feedback details
        """
        logger = self.get_logger("feedback")

        message = f"💬 Feedback Processing - User {user_id}: {feedback_type.upper()}"

        if details:
            if 'rating' in details:
                message += f" | Rating: {details['rating']}/5"
            if 'category' in details:
                message += f" | Category: {details['category']}"
            if 'processed' in details:
                message += f" | Processed: {details['processed']}"

        logger.info(message)

    def log_persistence_operation(self, operation: str, state_type: str, status: str,
                                details: Optional[Dict[str, Any]] = None):
        """
        Log persistence operations.

        Args:
            operation: Operation type ('save', 'load', 'backup', 'restore', etc.)
            state_type: Type of state being operated on
            status: Operation status ('started', 'success', 'failed')
            details: Operation details
        """
        logger = self.get_logger("persistence")

        status_icons = {
            'started': '⏳',
            'success': '💾',
            'failed': '❌'
        }

        icon = status_icons.get(status, 'ℹ️')
        message = f"{icon} Persistence - {operation.upper()} {state_type}: {status.upper()}"

        if details:
            if 'version' in details:
                message += f" | Version: {details['version']}"
            if 'size_mb' in details:
                message += f" | Size: {details['size_mb']:.2f}MB"
            if 'duration' in details:
                message += f" | Duration: {details['duration']:.3f}s"

        if status == 'failed':
            logger.error(message)
        else:
            logger.info(message)

    def start_progress_tracker(self, operation: str, total_steps: Optional[int] = None) -> str:
        """
        Start tracking progress for a long-running operation.

        Args:
            operation: Name of the operation
            total_steps: Total number of steps (optional)

        Returns:
            Progress tracker ID
        """
        tracker_id = f"{operation}_{threading.current_thread().ident}_{datetime.now().timestamp()}"
        self._progress_trackers[tracker_id] = {
            'operation': operation,
            'total_steps': total_steps,
            'current_step': 0,
            'start_time': datetime.now(),
            'last_update': datetime.now()
        }

        logger = self.get_logger("progress")
        message = f"▶️  Started: {operation}"
        if total_steps:
            message += f" (0/{total_steps})"
        logger.info(message)

        return tracker_id

    def update_progress(self, tracker_id: str, step: Optional[int] = None, message: Optional[str] = None):
        """
        Update progress for a tracked operation.

        Args:
            tracker_id: Progress tracker ID
            step: Current step number
            message: Optional progress message
        """
        if tracker_id not in self._progress_trackers:
            return

        tracker = self._progress_trackers[tracker_id]
        tracker['last_update'] = datetime.now()

        if step is not None:
            tracker['current_step'] = step

        logger = self.get_logger("progress")

        progress_msg = f"⏳ {tracker['operation']}"
        if tracker['total_steps']:
            progress_msg += f" ({tracker['current_step']}/{tracker['total_steps']})"
        if message:
            progress_msg += f" - {message}"

        logger.info(progress_msg)

    def complete_progress(self, tracker_id: str, success: bool = True, message: Optional[str] = None):
        """
        Complete a progress-tracked operation.

        Args:
            tracker_id: Progress tracker ID
            success: Whether operation completed successfully
            message: Completion message
        """
        if tracker_id not in self._progress_trackers:
            return

        tracker = self._progress_trackers[tracker_id]
        duration = (datetime.now() - tracker['start_time']).total_seconds()

        logger = self.get_logger("progress")

        icon = '✅' if success else '❌'
        status = 'Completed' if success else 'Failed'

        completion_msg = f"{icon} {status}: {tracker['operation']}"
        if tracker['total_steps']:
            completion_msg += f" ({tracker['current_step']}/{tracker['total_steps']})"
        completion_msg += f" in {duration:.2f}s"
        if message:
            completion_msg += f" - {message}"

        if success:
            logger.info(completion_msg)
        else:
            logger.error(completion_msg)

        # Clean up tracker
        del self._progress_trackers[tracker_id]


# Global instance for easy access
logging_config = LoggingConfig()

# Convenience functions for easy importing
def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a configured logger."""
    return logging_config.get_logger(name)

def log_performance(operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function for performance logging."""
    logging_config.log_performance_metric(operation, duration, metadata)

def log_api_request(method: str, endpoint: str, status_code: int, duration: float):
    """Convenience function for API request logging."""
    logging_config.log_api_request(method, endpoint, status_code, duration)

def log_learning_event(event_type: str, details: Dict[str, Any]):
    """Convenience function for learning event logging."""
    logging_config.log_learning_event(event_type, details)

def log_service_initialization(service_name: str, status: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function for service initialization logging."""
    logging_config.log_service_initialization(service_name, status, details)

def log_system_health(component: str, health_status: str, metrics: Optional[Dict[str, Any]] = None):
    """Convenience function for system health logging."""
    logging_config.log_system_health(component, health_status, metrics)

def log_curriculum_status(learner_id: str, status: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function for curriculum status logging."""
    logging_config.log_curriculum_status(learner_id, status, details)

def log_external_llm_interaction(provider: str, model: str, operation: str,
                               cost: Optional[float] = None, latency: Optional[float] = None,
                               tokens: Optional[Dict[str, int]] = None, error: Optional[str] = None):
    """Convenience function for external LLM interaction logging."""
    logging_config.log_external_llm_interaction(provider, model, operation, cost, latency, tokens, error)

def log_feedback_processing(user_id: str, feedback_type: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function for feedback processing logging."""
    logging_config.log_feedback_processing(user_id, feedback_type, details)

def log_persistence_operation(operation: str, state_type: str, status: str,
                            details: Optional[Dict[str, Any]] = None):
    """Convenience function for persistence operation logging."""
    logging_config.log_persistence_operation(operation, state_type, status, details)

def start_progress_tracker(operation: str, total_steps: Optional[int] = None) -> str:
    """Convenience function for starting progress tracking."""
    return logging_config.start_progress_tracker(operation, total_steps)

def update_progress(tracker_id: str, step: Optional[int] = None, message: Optional[str] = None):
    """Convenience function for updating progress."""
    logging_config.update_progress(tracker_id, step, message)

def complete_progress(tracker_id: str, success: bool = True, message: Optional[str] = None):
    """Convenience function for completing progress tracking."""
    logging_config.complete_progress(tracker_id, success, message)