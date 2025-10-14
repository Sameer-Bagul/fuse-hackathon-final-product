"""
Configuration Management - Educational Implementation

This module teaches you how to manage application configuration properly.
Configuration management is crucial for making applications flexible and
environment-aware (development, testing, production).

Why configuration management matters:
- Different environments need different settings (database URLs, API keys, etc.)
- Hard-coded values make code inflexible and insecure
- Configuration should be validated and type-safe
- Changes shouldn't require code redeployment

Design Patterns Used:
- Singleton Pattern: Single source of configuration truth
- Factory Pattern: Create config objects based on environment
- Strategy Pattern: Different loading strategies (file, env vars, etc.)

Best Practices Demonstrated:
- Environment variable priority over config files
- Type validation and defaults
- Configuration validation on startup
- Separation of sensitive and non-sensitive config
- Documentation of all configuration options
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Configuration for the LLM learning system."""
    num_actions: int = 10
    alpha: float = 0.1  # Learning rate
    epsilon: float = 0.1  # Exploration rate
    max_episodes: int = 1000
    reward_scale: float = 1.0

    def validate(self):
        """Validate LLM configuration values."""
        if not (0 < self.alpha <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, got {self.alpha}")
        if not (0 <= self.epsilon <= 1):
            raise ValueError(f"Epsilon must be between 0 and 1, got {self.epsilon}")
        if self.num_actions < 1:
            raise ValueError(f"num_actions must be positive, got {self.num_actions}")


@dataclass
class APIConfig:
    """Configuration for the FastAPI application."""
    host: str = "0.0.0.0"
    port: int = 8082
    debug: bool = False
    title: str = "LLM Learning System API"
    description: str = "FastAPI for LLM learning services with MVC architecture"
    version: str = "1.0.0"
    cors_origins: list = None

    def __post_init__(self):
        if self.cors_origins is None:
            env_origins = os.getenv('CORS_ORIGINS')
            if env_origins:
                self.cors_origins = [origin.strip() for origin in env_origins.split(',')]
            else:
                self.cors_origins = ["http://localhost:3000", "http://localhost:5173", "http://localhost:3001", "http://localhost:8081"]  # React dev servers

    def validate(self):
        """Validate API configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")


@dataclass
class DatabaseConfig:
    """Configuration for data persistence."""
    enabled: bool = False  # Start with in-memory for simplicity
    url: str = "sqlite:///llm_learning.db"
    connection_pool_size: int = 5
    connection_timeout: int = 30

    def validate(self):
        """Validate database configuration."""
        if self.connection_pool_size < 1:
            raise ValueError("Connection pool size must be positive")
        if self.connection_timeout < 1:
            raise ValueError("Connection timeout must be positive")


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

    def validate(self):
        """Validate logging configuration."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}, got {self.level}")


@dataclass
class LLMProvider:
    """Configuration for external LLM providers."""
    name: str
    base_url: str
    api_version: str = "v1"
    models: Dict[str, Dict[str, Any]] = None
    rate_limits: Dict[str, Any] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if self.models is None:
            self.models = {}
        if self.rate_limits is None:
            self.rate_limits = {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "tokens_per_minute": 100000
            }

    def validate(self):
        """Validate LLM provider configuration."""
        if not self.name:
            raise ValueError("Provider name cannot be empty")
        if not self.base_url:
            raise ValueError("Base URL cannot be empty")
        if self.timeout < 1:
            raise ValueError("Timeout must be positive")
        if self.retry_attempts < 0:
            raise ValueError("Retry attempts must be non-negative")


@dataclass
class APICredentials:
    """Secure API credentials management."""
    provider_name: str
    api_key: Optional[str] = None
    api_key_env_var: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    additional_headers: Dict[str, str] = None

    def __post_init__(self):
        if self.additional_headers is None:
            self.additional_headers = {}

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment variable or direct value."""
        if self.api_key_env_var:
            return os.getenv(self.api_key_env_var)
        return self.api_key

    def validate(self):
        """Validate API credentials configuration."""
        if not self.provider_name:
            raise ValueError("Provider name cannot be empty")
        if not self.api_key and not self.api_key_env_var:
            raise ValueError("Either api_key or api_key_env_var must be provided")


@dataclass
class CostTracking:
    """Cost tracking and monitoring configuration."""
    provider_name: str
    input_token_cost_per_1k: float = 0.0
    output_token_cost_per_1k: float = 0.0
    currency: str = "USD"
    monthly_budget_limit: Optional[float] = None
    daily_budget_limit: Optional[float] = None
    enable_cost_alerts: bool = True
    alert_threshold_percentage: float = 80.0

    def validate(self):
        """Validate cost tracking configuration."""
        if not self.provider_name:
            raise ValueError("Provider name cannot be empty")
        if self.input_token_cost_per_1k < 0:
            raise ValueError("Input token cost must be non-negative")
        if self.output_token_cost_per_1k < 0:
            raise ValueError("Output token cost must be non-negative")
        if not (0 <= self.alert_threshold_percentage <= 100):
            raise ValueError("Alert threshold percentage must be between 0 and 100")


@dataclass
class ExternalLLMConfig:
    """Configuration for external LLM integration."""
    enabled: bool = False
    default_provider: str = "openai"
    fallback_to_internal: bool = True
    providers: Dict[str, LLMProvider] = None
    credentials: Dict[str, APICredentials] = None
    cost_tracking: Dict[str, CostTracking] = None
    hybrid_learning_enabled: bool = True
    model_comparison_enabled: bool = True

    def __post_init__(self):
        if self.providers is None:
            self.providers = {}
        if self.credentials is None:
            self.credentials = {}
        if self.cost_tracking is None:
            self.cost_tracking = {}

    def validate(self):
        """Validate external LLM configuration."""
        if self.enabled:
            if not self.providers:
                raise ValueError("At least one provider must be configured when external LLMs are enabled")
            if self.default_provider and self.default_provider not in self.providers:
                raise ValueError(f"Default provider '{self.default_provider}' not found in providers")


@dataclass
class AppConfig:
    """Main application configuration container."""
    llm: LLMConfig
    api: APIConfig
    database: DatabaseConfig
    logging: LoggingConfig
    external_llm: ExternalLLMConfig = None
    environment: str = "development"

    def __post_init__(self):
        if self.external_llm is None:
            self.external_llm = ExternalLLMConfig()

    def validate(self):
        """Validate all configuration sections."""
        self.llm.validate()
        self.api.validate()
        self.database.validate()
        self.logging.validate()
        self.external_llm.validate()

        valid_envs = ['development', 'testing', 'staging', 'production']
        if self.environment not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}, got {self.environment}")


class ConfigManager:
    """
    Configuration manager that loads and validates application settings.

    This class demonstrates professional configuration management:
    1. Multiple loading sources (files, environment variables)
    2. Environment-specific overrides
    3. Validation and type safety
    4. Singleton pattern for consistency

    Educational Notes:
    - Configuration should be loaded once at startup
    - Environment variables override file config for security
    - Validation prevents runtime errors from bad config
    - Type hints help catch configuration errors early
    """

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._config = self._load_config()
            self._validate_config()

    def _load_config(self) -> AppConfig:
        """Load configuration from multiple sources."""

        # Start with defaults
        config = AppConfig(
            llm=LLMConfig(),
            api=APIConfig(),
            database=DatabaseConfig(),
            logging=LoggingConfig(),
            external_llm=ExternalLLMConfig()
        )

        # Load from config file if it exists
        config_file = Path(__file__).parent.parent / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                self._merge_config(config, file_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")

        # Override with environment variables (highest priority)
        self._load_from_env(config)

        # Set environment
        config.environment = os.getenv('APP_ENV', 'development')

        return config

    def _merge_config(self, config: AppConfig, file_config: Dict[str, Any]):
        """Merge file configuration into the config object."""
        # This is a simplified merge - in production you'd want a more robust solution
        if 'llm' in file_config:
            for key, value in file_config['llm'].items():
                if hasattr(config.llm, key):
                    setattr(config.llm, key, value)

        if 'api' in file_config:
            for key, value in file_config['api'].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)

        # Similar for other sections...

    def _load_from_env(self, config: AppConfig):
        """Load configuration from environment variables."""

        # LLM settings
        if 'LLM_NUM_ACTIONS' in os.environ:
            config.llm.num_actions = int(os.environ['LLM_NUM_ACTIONS'])
        if 'LLM_ALPHA' in os.environ:
            config.llm.alpha = float(os.environ['LLM_ALPHA'])
        if 'LLM_EPSILON' in os.environ:
            config.llm.epsilon = float(os.environ['LLM_EPSILON'])

        # API settings
        config.api.host = os.getenv('API_HOST', config.api.host)
        if 'API_PORT' in os.environ:
            config.api.port = int(os.environ['API_PORT'])
        config.api.debug = os.getenv('API_DEBUG', 'false').lower() == 'true'

        # Database settings
        config.database.enabled = os.getenv('DB_ENABLED', 'false').lower() == 'true'
        config.database.url = os.getenv('DATABASE_URL', config.database.url)

        # Logging settings
        config.logging.level = os.getenv('LOG_LEVEL', config.logging.level)

        # External LLM settings
        config.external_llm.enabled = os.getenv('EXTERNAL_LLM_ENABLED', 'false').lower() == 'true'
        config.external_llm.default_provider = os.getenv('EXTERNAL_LLM_DEFAULT_PROVIDER', config.external_llm.default_provider)
        config.external_llm.fallback_to_internal = os.getenv('EXTERNAL_LLM_FALLBACK_TO_INTERNAL', 'true').lower() == 'true'
        config.external_llm.hybrid_learning_enabled = os.getenv('EXTERNAL_LLM_HYBRID_LEARNING', 'true').lower() == 'true'
        config.external_llm.model_comparison_enabled = os.getenv('EXTERNAL_LLM_MODEL_COMPARISON', 'true').lower() == 'true'

    def _validate_config(self):
        """Validate the loaded configuration."""
        try:
            self._config.validate()
            logger.info("Configuration validation successful")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    @property
    def config(self) -> AppConfig:
        """Get the current configuration."""
        return self._config

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by section and key.

        Args:
            section: Configuration section (llm, api, database, logging)
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        section_obj = getattr(self._config, section, None)
        if section_obj and hasattr(section_obj, key):
            return getattr(section_obj, key)
        return default

    def save_config(self, filepath: Union[str, Path]):
        """Save current configuration to a JSON file."""
        config_dict = asdict(self._config)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    def reload_config(self):
        """Reload configuration from sources."""
        self._config = self._load_config()
        self._validate_config()
        logger.info("Configuration reloaded")


# Global instance
config_manager = ConfigManager()

# Convenience functions
def get_config() -> AppConfig:
    """Get the global configuration."""
    return config_manager.config

def get_llm_config() -> LLMConfig:
    """Get LLM-specific configuration."""
    return config_manager.config.llm

def get_api_config() -> APIConfig:
    """Get API-specific configuration."""
    return config_manager.config.api

def get_db_config() -> DatabaseConfig:
    """Get database configuration."""
    return config_manager.config.database

def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return config_manager.config.logging

def get_external_llm_config() -> ExternalLLMConfig:
    """Get external LLM configuration."""
    return config_manager.config.external_llm