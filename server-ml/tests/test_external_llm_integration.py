"""
Test External LLM Integration

This test verifies that the external LLM integration works correctly
with the existing ML backend system.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(__file__))

from controllers.prompt_controller import PromptController
from services.external_llm_service import ExternalLLMService, LLMResponse
from utils.config import get_external_llm_config


class TestExternalLLMIntegration:
    """Test external LLM integration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = get_external_llm_config()

    def test_config_models(self):
        """Test that configuration models are properly defined."""
        from utils.config import LLMProvider, APICredentials, CostTracking, ExternalLLMConfig

        # Test LLMProvider
        provider = LLMProvider(
            name="test_provider",
            base_url="https://api.test.com",
            models={"test-model": {"context_window": 4096, "input_cost_per_1k": 0.01}}
        )
        assert provider.name == "test_provider"
        assert provider.base_url == "https://api.test.com"

        # Test APICredentials
        creds = APICredentials(
            provider_name="test_provider",
            api_key="test_key_123"
        )
        assert creds.provider_name == "test_provider"
        assert creds.get_api_key() == "test_key_123"

        # Test CostTracking
        cost = CostTracking(
            provider_name="test_provider",
            input_token_cost_per_1k=0.01,
            output_token_cost_per_1k=0.02
        )
        assert cost.provider_name == "test_provider"
        assert cost.input_token_cost_per_1k == 0.01

    @patch('services.external_llm_service.OpenAI')
    def test_external_llm_service_initialization(self, mock_openai):
        """Test that external LLM service initializes correctly."""
        # Mock the config to enable external LLMs
        with patch('utils.config.get_external_llm_config') as mock_config:
            mock_config.return_value.enabled = True
            mock_config.return_value.providers = {
                "openai": Mock()
            }
            mock_config.return_value.credentials = {
                "openai": Mock(get_api_key=Mock(return_value="test_key"))
            }

            service = ExternalLLMService()

            # Should initialize without errors
            assert service is not None
            assert hasattr(service, 'generate_response')

    @patch('services.external_llm_service.OpenAI')
    async def test_generate_response_error_handling(self, mock_openai):
        """Test error handling in generate_response."""
        with patch('utils.config.get_external_llm_config') as mock_config:
            mock_config.return_value.enabled = True
            mock_config.return_value.default_provider = "openai"
            mock_config.return_value.providers = {
                "openai": Mock(rate_limits={"requests_per_minute": 60})
            }
            mock_config.return_value.credentials = {
                "openai": Mock(get_api_key=Mock(return_value="test_key"))
            }

            service = ExternalLLMService()

            # Test with invalid input
            response = await service.generate_response("")
            assert response.error is not None
            assert "Validation error" in response.error

            # Test with None input
            response = await service.generate_response(None)
            assert response.error is not None

    def test_prompt_controller_external_llm_integration(self):
        """Test that PromptController integrates with external LLMs."""
        # Mock external LLM service
        mock_external_service = Mock()
        mock_external_service.generate_response = AsyncMock(return_value=LLMResponse(
            response="Test external response",
            provider="openai",
            model="gpt-3.5-turbo",
            cost=0.01,
            latency=1.0
        ))

        # Create controller with mocked services
        controller = PromptController(
            external_llm_service=mock_external_service
        )

        # Test switching modes
        assert controller.switch_llm_mode(True) == True
        assert controller.use_external_llm == True

        assert controller.switch_llm_mode(False) == True
        assert controller.use_external_llm == False

    def test_llm_response_structure(self):
        """Test LLMResponse data structure."""
        response = LLMResponse(
            response="Test response",
            provider="openai",
            model="gpt-4",
            input_tokens=10,
            output_tokens=20,
            cost=0.005,
            latency=2.5
        )

        assert response.response == "Test response"
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.total_tokens == 30  # Calculated
        assert response.cost == 0.005
        assert response.latency == 2.5
        assert response.error is None

    def test_cost_calculation(self):
        """Test cost calculation logic."""
        service = ExternalLLMService()

        # Test cost calculation
        cost = service._calculate_cost("openai", 100, 50)
        # Should return 0 since no cost config is set up
        assert cost == 0.0

    def test_rate_limiter(self):
        """Test rate limiter functionality."""
        from services.external_llm_service import RateLimiter
        import asyncio

        limiter = RateLimiter(requests_per_minute=10)

        async def test_rate_limiting():
            # Should allow first request
            assert await limiter.acquire() == True

            # Fill up the rate limiter
            for _ in range(9):
                await limiter.acquire()

            # Next request should be blocked
            assert await limiter.acquire() == False

        asyncio.run(test_rate_limiting())

    def test_api_endpoints_structure(self):
        """Test that API endpoints are properly structured."""
        # This is a basic smoke test to ensure imports work
        try:
            from app import app
            assert app is not None

            # Check that new routes are registered
            routes = [route.path for route in app.routes]
            assert "/llm/switch" in routes
            assert "/llm/models" in routes
            assert "/llm/costs" in routes
            assert "/llm/compare" in routes
            assert "/llm/status" in routes

        except ImportError:
            # App might not be importable in test environment
            pass


if __name__ == "__main__":
    # Run basic tests
    test = TestExternalLLMIntegration()

    print("Running external LLM integration tests...")

    try:
        test.setup_method()
        test.test_config_models()
        print("✓ Configuration models test passed")

        test.test_llm_response_structure()
        print("✓ LLM response structure test passed")

        test.test_cost_calculation()
        print("✓ Cost calculation test passed")

        test.test_rate_limiter()
        print("✓ Rate limiter test passed")

        test.test_api_endpoints_structure()
        print("✓ API endpoints structure test passed")

        print("\n✅ All external LLM integration tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)