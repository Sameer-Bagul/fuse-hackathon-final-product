import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

try:
    import openai
    from openai import OpenAI
except ImportError:
    OpenAI = None
    openai = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    Anthropic = None
    anthropic = None

from utils.config import get_external_llm_config, LLMProvider, APICredentials, CostTracking
from utils.logging_config import get_logger, log_external_llm_interaction

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from external LLM."""
    response: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    name: str
    provider: str
    context_window: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    capabilities: List[str]
    max_tokens: int
    description: str = ""


@dataclass
class CostMetrics:
    """Cost tracking metrics."""
    provider: str
    total_cost: float
    monthly_cost: float
    daily_cost: float
    total_tokens_input: int
    total_tokens_output: int
    request_count: int
    last_updated: datetime


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Check if request can proceed within rate limits."""
        async with self.lock:
            now = datetime.now()

            # Clean old requests
            self.requests = [req for req in self.requests if now - req < timedelta(hours=1)]

            # Check limits
            recent_minute = sum(1 for req in self.requests if now - req < timedelta(minutes=1))
            recent_hour = len(self.requests)

            if recent_minute >= self.requests_per_minute or recent_hour >= self.requests_per_hour:
                return False

            self.requests.append(now)
            return True

    async def wait_for_slot(self) -> None:
        """Wait until a request slot is available."""
        while not await self.acquire():
            await asyncio.sleep(1)


class ExternalLLMService:
    """
    External LLM Service for integrating with multiple LLM providers.

    Supports OpenAI GPT, Anthropic Claude, and other providers with:
    - Rate limiting and cost tracking
    - Model comparison capabilities
    - Hybrid learning integration
    - Fallback mechanisms
    """

    def __init__(self):
        self.config = get_external_llm_config()
        self.logger = logger

        # Initialize clients
        self.clients = {}
        self.rate_limiters = {}
        self.cost_trackers = {}

        # Model registry
        self.models = self._initialize_models()

        # Cost tracking
        self.cost_metrics = {}
        self._load_cost_metrics()

        self.logger.info("ExternalLLMService initialized")

    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize available models from configuration."""
        models = {}

        # Default models for common providers
        default_models = {
            "openai": {
                "gpt-4": ModelInfo(
                    name="gpt-4",
                    provider="openai",
                    context_window=8192,
                    input_cost_per_1k=0.03,
                    output_cost_per_1k=0.06,
                    capabilities=["text-generation", "code", "reasoning"],
                    max_tokens=4096,
                    description="Most capable GPT-4 model"
                ),
                "gpt-3.5-turbo": ModelInfo(
                    name="gpt-3.5-turbo",
                    provider="openai",
                    context_window=4096,
                    input_cost_per_1k=0.0015,
                    output_cost_per_1k=0.002,
                    capabilities=["text-generation", "code"],
                    max_tokens=4096,
                    description="Fast and cost-effective GPT-3.5 model"
                )
            },
            "anthropic": {
                "claude-3-opus": ModelInfo(
                    name="claude-3-opus",
                    provider="anthropic",
                    context_window=200000,
                    input_cost_per_1k=0.015,
                    output_cost_per_1k=0.075,
                    capabilities=["text-generation", "code", "reasoning", "analysis"],
                    max_tokens=4096,
                    description="Most capable Claude model"
                ),
                "claude-3-sonnet": ModelInfo(
                    name="claude-3-sonnet",
                    provider="anthropic",
                    context_window=200000,
                    input_cost_per_1k=0.003,
                    output_cost_per_1k=0.015,
                    capabilities=["text-generation", "code", "reasoning"],
                    max_tokens=4096,
                    description="Balanced performance and cost"
                )
            }
        }

        # Load from config if available
        for provider_name, provider in self.config.providers.items():
            if provider_name in default_models:
                models.update({f"{provider_name}:{model.name}": model
                             for model in default_models[provider_name].values()})

            # Override with config-specific models
            for model_name, model_config in provider.models.items():
                key = f"{provider_name}:{model_name}"
                models[key] = ModelInfo(
                    name=model_name,
                    provider=provider_name,
                    **model_config
                )

        return models

    def _initialize_client(self, provider_name: str):
        """Initialize API client for a provider."""
        if provider_name in self.clients:
            return self.clients[provider_name]

        if provider_name not in self.config.credentials:
            raise ValueError(f"No credentials configured for provider: {provider_name}")

        credentials = self.config.credentials[provider_name]
        api_key = credentials.get_api_key()

        if not api_key:
            raise ValueError(f"API key not found for provider: {provider_name}")

        try:
            if provider_name == "openai" and OpenAI:
                self.clients[provider_name] = OpenAI(api_key=api_key)
            elif provider_name == "anthropic" and Anthropic:
                self.clients[provider_name] = Anthropic(api_key=api_key)
            else:
                raise ValueError(f"Unsupported provider: {provider_name}")

            # Initialize rate limiter
            provider_config = self.config.providers.get(provider_name)
            if provider_config and provider_config.rate_limits:
                self.rate_limiters[provider_name] = RateLimiter(
                    requests_per_minute=provider_config.rate_limits.get("requests_per_minute", 60),
                    requests_per_hour=provider_config.rate_limits.get("requests_per_hour", 1000)
                )

            self.logger.info(f"Initialized client for provider: {provider_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize client for {provider_name}: {str(e)}")
            raise

    def _load_cost_metrics(self):
        """Load cost metrics from storage."""
        # In a real implementation, this would load from a database
        # For now, initialize empty metrics
        for provider_name in self.config.providers:
            self.cost_metrics[provider_name] = CostMetrics(
                provider=provider_name,
                total_cost=0.0,
                monthly_cost=0.0,
                daily_cost=0.0,
                total_tokens_input=0,
                total_tokens_output=0,
                request_count=0,
                last_updated=datetime.now()
            )

    async def generate_response(self, prompt: str, provider: str = None, model: str = None,
                              temperature: float = 0.7, max_tokens: int = None) -> LLMResponse:
        """
        Generate response from external LLM with comprehensive error handling.

        Args:
            prompt: Input prompt text
            provider: Provider name (uses default if None)
            model: Model name (uses provider default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse object with generated text and metadata
        """
        start_time = time.time()

        try:
            # Input validation
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")
            if len(prompt.strip()) == 0:
                raise ValueError("Prompt cannot be empty or whitespace only")
            if len(prompt) > 100000:  # Reasonable limit
                raise ValueError("Prompt too long (max 100,000 characters)")

            # Validate temperature
            if not (0.0 <= temperature <= 2.0):
                raise ValueError("Temperature must be between 0.0 and 2.0")

            # Use default provider if not specified
            if not provider:
                provider = self.config.default_provider

            if provider not in self.config.providers:
                raise ValueError(f"Unknown provider: {provider}")

            # Check if provider credentials are available
            if provider not in self.config.credentials:
                raise ValueError(f"No credentials configured for provider: {provider}")

            credentials = self.config.credentials[provider]
            api_key = credentials.get_api_key()
            if not api_key:
                raise ValueError(f"API key not available for provider: {provider}")

            # Initialize client if needed
            self._initialize_client(provider)

            # Get model info
            if not model:
                # Use first available model for provider
                provider_models = [m for m in self.models.values() if m.provider == provider]
                if not provider_models:
                    raise ValueError(f"No models available for provider: {provider}")
                model = provider_models[0].name

            model_key = f"{provider}:{model}"
            if model_key not in self.models:
                raise ValueError(f"Unknown model: {model} for provider: {provider}")

            model_info = self.models[model_key]

            # Apply rate limiting
            if provider in self.rate_limiters:
                await self.rate_limiters[provider].wait_for_slot()

            # Set max tokens with safety limits
            if not max_tokens:
                max_tokens = min(model_info.max_tokens, 2048)  # Reasonable default
            max_tokens = min(max_tokens, model_info.max_tokens)  # Respect model limits
            max_tokens = max(max_tokens, 1)  # Minimum 1 token

            # Check cost limits before making API call
            if self._would_exceed_cost_limits(provider, model_info):
                raise ValueError(f"Request would exceed cost limits for provider: {provider}")

            # Generate response with retry logic
            response_text, usage = await self._call_provider_api_with_retry(
                provider, model, prompt, temperature, max_tokens
            )

            # Validate response
            if not response_text or not isinstance(response_text, str):
                raise ValueError("Invalid response received from provider")

            # Calculate cost
            cost = self._calculate_cost(provider, usage.get('input_tokens', 0),
                                       usage.get('output_tokens', 0))

            # Update cost metrics
            self._update_cost_metrics(provider, cost, usage)

            latency = time.time() - start_time

            # Log successful external LLM interaction
            log_external_llm_interaction(
                provider=provider,
                model=model,
                operation='generate',
                cost=cost,
                latency=latency,
                tokens={
                    'input': usage.get('input_tokens', 0),
                    'output': usage.get('output_tokens', 0),
                    'total': usage.get('total_tokens', 0)
                }
            )

            self.logger.info(f"🤖 External LLM response generated | {provider}/{model} | "
                           f"Cost: ${cost:.4f} | Latency: {latency:.3f}s | "
                           f"Tokens: {usage.get('total_tokens', 0)}")

            return LLMResponse(
                response=response_text,
                provider=provider,
                model=model,
                input_tokens=usage.get('input_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                total_tokens=usage.get('total_tokens', 0),
                cost=cost,
                latency=latency,
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model_info": asdict(model_info)
                }
            )

        except ValueError as e:
            # Input validation errors
            latency = time.time() - start_time
            self.logger.warning(f"Validation error in generate_response: {str(e)}")

            # Log failed external LLM interaction
            log_external_llm_interaction(
                provider=provider or "unknown",
                model=model or "unknown",
                operation='generate',
                error=f"Validation error: {str(e)}"
            )

            return LLMResponse(
                response="",
                provider=provider or "unknown",
                model=model or "unknown",
                latency=latency,
                error=f"Validation error: {str(e)}"
            )

        except Exception as e:
            # Unexpected errors
            latency = time.time() - start_time
            self.logger.error(f"Unexpected error generating response: {str(e)}")

            # Mask sensitive information in error messages
            safe_error = self._sanitize_error_message(str(e))

            # Log failed external LLM interaction
            log_external_llm_interaction(
                provider=provider or "unknown",
                model=model or "unknown",
                operation='generate',
                error=f"API error: {safe_error}",
                latency=latency
            )

            return LLMResponse(
                response="",
                provider=provider or "unknown",
                model=model or "unknown",
                latency=latency,
                error=f"API error: {safe_error}"
            )

    async def _call_provider_api(self, provider: str, model: str, prompt: str,
                               temperature: float, max_tokens: int) -> tuple:
        """Call the appropriate provider API."""
        client = self.clients[provider]

        if provider == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            response_text = response.choices[0].message.content
            usage = {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

        elif provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            usage = {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            }

        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return response_text, usage

    def _calculate_cost(self, provider: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost for tokens used."""
        if provider not in self.config.cost_tracking:
            return 0.0

        cost_config = self.config.cost_tracking[provider]
        input_cost = (input_tokens / 1000) * cost_config.input_token_cost_per_1k
        output_cost = (output_tokens / 1000) * cost_config.output_token_cost_per_1k

        return input_cost + output_cost

    def _update_cost_metrics(self, provider: str, cost: float, usage: Dict[str, int]):
        """Update cost tracking metrics."""
        if provider not in self.cost_metrics:
            return

        metrics = self.cost_metrics[provider]
        metrics.total_cost += cost
        metrics.total_tokens_input += usage.get('input_tokens', 0)
        metrics.total_tokens_output += usage.get('output_tokens', 0)
        metrics.request_count += 1
        metrics.last_updated = datetime.now()

        # Reset daily/monthly costs if needed (simplified)
        # In production, this would be more sophisticated
        now = datetime.now()
        if now.day != metrics.last_updated.day:
            metrics.daily_cost = cost
        else:
            metrics.daily_cost += cost

        if now.month != metrics.last_updated.month:
            metrics.monthly_cost = cost
        else:
            metrics.monthly_cost += cost

    async def compare_models(self, prompt: str, providers_models: List[Dict[str, str]],
                           temperature: float = 0.7) -> Dict[str, LLMResponse]:
        """
        Compare responses from multiple models.

        Args:
            prompt: Input prompt
            providers_models: List of {"provider": str, "model": str} dicts
            temperature: Sampling temperature

        Returns:
            Dict mapping model keys to LLMResponse objects
        """
        if not self.config.model_comparison_enabled:
            raise ValueError("Model comparison is disabled in configuration")

        tasks = []
        for pm in providers_models:
            provider = pm.get("provider")
            model = pm.get("model")
            if not provider or not model:
                continue

            task = self.generate_response(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=temperature
            )
            tasks.append((f"{provider}:{model}", task))

        # Execute all requests concurrently
        results = {}
        for model_key, task in tasks:
            try:
                results[model_key] = await task
            except Exception as e:
                self.logger.error(f"Error comparing model {model_key}: {str(e)}")
                results[model_key] = LLMResponse(
                    response="",
                    provider=model_key.split(":")[0],
                    model=model_key.split(":")[1],
                    error=str(e)
                )

        return results

    def get_available_models(self) -> Dict[str, ModelInfo]:
        """Get all available models."""
        return self.models.copy()

    def get_cost_metrics(self, provider: str = None) -> Union[CostMetrics, Dict[str, CostMetrics]]:
        """Get cost metrics for provider(s)."""
        if provider:
            return self.cost_metrics.get(provider)
        return self.cost_metrics.copy()

    def switch_provider(self, new_provider: str) -> bool:
        """Switch the default provider."""
        if new_provider not in self.config.providers:
            return False

        self.config.default_provider = new_provider
        self.logger.info(f"Switched default provider to: {new_provider}")
        return True

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all configured providers."""
        status = {}
        for provider_name in self.config.providers:
            status[provider_name] = {
                "configured": True,
                "client_initialized": provider_name in self.clients,
                "has_credentials": provider_name in self.config.credentials,
                "rate_limiter": provider_name in self.rate_limiters,
                "models_available": len([m for m in self.models.values() if m.provider == provider_name])
            }
        return status

    def _would_exceed_cost_limits(self, provider: str, model_info: ModelInfo) -> bool:
        """Check if a request would exceed cost limits."""
        if provider not in self.config.cost_tracking:
            return False

        cost_config = self.config.cost_tracking[provider]

        # Check monthly budget
        if cost_config.monthly_budget_limit:
            current_monthly = self.cost_metrics.get(provider, CostMetrics(provider=provider)).monthly_cost
            estimated_cost = (model_info.input_cost_per_1k * 100 + model_info.output_cost_per_1k * 50) / 1000  # Rough estimate
            if current_monthly + estimated_cost > cost_config.monthly_budget_limit:
                return True

        # Check daily budget
        if cost_config.daily_budget_limit:
            current_daily = self.cost_metrics.get(provider, CostMetrics(provider=provider)).daily_cost
            if current_daily + estimated_cost > cost_config.daily_budget_limit:
                return True

        return False

    async def _call_provider_api_with_retry(self, provider: str, model: str, prompt: str,
                                          temperature: float, max_tokens: int) -> tuple:
        """Call provider API with retry logic."""
        provider_config = self.config.providers.get(provider)
        max_retries = provider_config.retry_attempts if provider_config else 3
        retry_delay = provider_config.retry_delay if provider_config else 1.0

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await self._call_provider_api(provider, model, prompt, temperature, max_tokens)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API call attempt {attempt + 1} failed for {provider}: {str(e)}")

                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    break

        # All retries failed
        raise last_exception

    def _sanitize_error_message(self, error_msg: str) -> str:
        """Sanitize error messages to avoid exposing sensitive information."""
        # Remove potential API keys, tokens, or other sensitive data
        import re

        # Remove patterns that look like API keys
        error_msg = re.sub(r'Bearer\s+[A-Za-z0-9_-]{20,}', 'Bearer [REDACTED]', error_msg)
        error_msg = re.sub(r'api[_-]?key[_-]?[=:]\s*[A-Za-z0-9_-]{20,}', 'api_key=[REDACTED]', error_msg)
        error_msg = re.sub(r'[A-Za-z0-9_-]{32,}', '[REDACTED]', error_msg)  # Generic long token pattern

        return error_msg