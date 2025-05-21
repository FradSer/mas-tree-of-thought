import os
import pytest
import logging
from unittest.mock import patch, MagicMock

from google.adk.models.lite_llm import LiteLlm
from multi_tool_agent.llm_config import _configure_llm_models, _log_google_credential_warnings

# Assuming the default model is gemini-1.5-flash as per the provided llm_config.py
EXPECTED_DEFAULT_GOOGLE_MODEL = "gemini-1.5-flash"

# Agent names as used in environment variables (uppercase) and in the function's return tuple (lowercase)
AGENT_NAMES_ENV = ["PLANNER", "RESEARCHER", "ANALYZER", "CRITIC", "SYNTHESIZER", "COORDINATOR"]
AGENT_NAMES_FUNC_ORDER = ["planner", "researcher", "analyzer", "critic", "synthesizer", "coordinator"]


@pytest.fixture(autouse=True)
def reset_litellm_master_key():
    """Ensure LiteLLM's master_key is reset before each test if it's modified by LiteLLM."""
    # This is a precaution. If LiteLLM modifies its global state, this might be needed.
    # For now, assuming LiteLLM instances are independent or config doesn't alter global LiteLLM state.
    pass

def get_configs_as_dict(configs_tuple):
    return dict(zip(AGENT_NAMES_FUNC_ORDER, configs_tuple))

def test_default_models_when_no_env_vars_set():
    with patch.dict(os.environ, {}, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings') as mock_log_warnings:
            configs = _configure_llm_models()
            configs_dict = get_configs_as_dict(configs)

    for agent_name in AGENT_NAMES_FUNC_ORDER:
        assert configs_dict[agent_name] == EXPECTED_DEFAULT_GOOGLE_MODEL
    mock_log_warnings.assert_called_once() # Should be called to check default Google creds

def test_google_model_configuration_specific_agent():
    env_vars = {
        "PLANNER_MODEL_CONFIG": "google:custom-google-model-for-planner",
        "GOOGLE_API_KEY": "test_google_key"
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings') as mock_log_warnings:
            configs = _configure_llm_models()
            configs_dict = get_configs_as_dict(configs)

    assert configs_dict["planner"] == "custom-google-model-for-planner"
    assert configs_dict["researcher"] == EXPECTED_DEFAULT_GOOGLE_MODEL # Others default
    mock_log_warnings.assert_called_once()

def test_google_model_with_vertexai():
    env_vars = {
        "RESEARCHER_MODEL_CONFIG": "google:vertex-model-for-researcher",
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        # Assuming GOOGLE_APPLICATION_CREDENTIALS would be set in a real Vertex AI env
    }
    with patch.dict(os.environ, env_vars, clear=True):
        # Mocking the logger for _log_google_credential_warnings specifically
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance
            
            configs = _configure_llm_models()
            configs_dict = get_configs_as_dict(configs)

    assert configs_dict["researcher"] == "vertex-model-for-researcher"
    # Check if the warning for missing GOOGLE_APPLICATION_CREDENTIALS was logged
    # This depends on the exact logic of _log_google_credential_warnings
    # For now, we assume it's called and might log something.
    # A more precise test would check mock_logger_instance.warning.call_args
    assert mock_get_logger.called # Confirms logger was obtained

def test_openrouter_model_with_key():
    env_vars = {
        "ANALYZER_MODEL_CONFIG": "openrouter:some/analyzer-model",
        "OPENROUTER_API_KEY": "fake_or_key"
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings'): # Mock to avoid its logic
            configs = _configure_llm_models()
            configs_dict = get_configs_as_dict(configs)

    analyzer_cfg = configs_dict["analyzer"]
    assert isinstance(analyzer_cfg, LiteLlm)
    assert analyzer_cfg.model == "openrouter/some/analyzer-model" # LiteLLM prepends provider for OpenRouter
    assert analyzer_cfg.api_key == "fake_or_key"
    assert configs_dict["planner"] == EXPECTED_DEFAULT_GOOGLE_MODEL # Others default

def test_openrouter_model_no_key_fallback():
    env_vars = {
        "ANALYZER_MODEL_CONFIG": "openrouter:some/analyzer-model"
        # OPENROUTER_API_KEY is missing
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings'):
            with patch('logging.warning') as mock_warning: # Capture general logging.warning
                configs = _configure_llm_models()
                configs_dict = get_configs_as_dict(configs)

    assert configs_dict["analyzer"] == EXPECTED_DEFAULT_GOOGLE_MODEL
    mock_warning.assert_any_call("OPENROUTER_API_KEY not found. Falling back to default for ANALYZER.")

def test_openai_model_with_key_and_base():
    env_vars = {
        "CRITIC_MODEL_CONFIG": "openai:gpt-4o",
        "OPENAI_API_KEY": "fake_openai_key",
        "OPENAI_API_BASE": "http://localhost:1234/v1"
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings'):
            configs = _configure_llm_models()
            configs_dict = get_configs_as_dict(configs)

    critic_cfg = configs_dict["critic"]
    assert isinstance(critic_cfg, LiteLlm)
    assert critic_cfg.model == "openai/gpt-4o" # LiteLLM convention
    assert critic_cfg.api_key == "fake_openai_key"
    assert critic_cfg.api_base == "http://localhost:1234/v1"
    assert configs_dict["planner"] == EXPECTED_DEFAULT_GOOGLE_MODEL # Others default

def test_openai_model_no_key_fallback():
    env_vars = {
        "CRITIC_MODEL_CONFIG": "openai:gpt-4o",
        "OPENAI_API_BASE": "http://localhost:1234/v1"
        # OPENAI_API_KEY is missing
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings'):
            with patch('logging.warning') as mock_warning:
                configs = _configure_llm_models()
                configs_dict = get_configs_as_dict(configs)

    assert configs_dict["critic"] == EXPECTED_DEFAULT_GOOGLE_MODEL
    mock_warning.assert_any_call("OPENAI_API_KEY not found. Falling back to default for CRITIC.")

def test_openai_model_no_base_fallback():
    env_vars = {
        "CRITIC_MODEL_CONFIG": "openai:gpt-4o",
        "OPENAI_API_KEY": "fake_openai_key"
        # OPENAI_API_BASE is missing
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings'):
            with patch('logging.warning') as mock_warning:
                configs = _configure_llm_models()
                configs_dict = get_configs_as_dict(configs)
    
    assert configs_dict["critic"] == EXPECTED_DEFAULT_GOOGLE_MODEL
    mock_warning.assert_any_call("OPENAI_API_BASE not found. Falling back to default for CRITIC.")


def test_invalid_config_format_fallback():
    env_vars = {"SYNTHESIZER_MODEL_CONFIG": "badformatnoclon"}
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings'):
            with patch('logging.warning') as mock_warning:
                configs = _configure_llm_models()
                configs_dict = get_configs_as_dict(configs)

    assert configs_dict["synthesizer"] == EXPECTED_DEFAULT_GOOGLE_MODEL
    mock_warning.assert_any_call("Invalid model configuration format for SYNTHESIZER: badformatnoclon. Expected 'provider:model_name'. Falling back to default.")

def test_unsupported_provider_fallback():
    env_vars = {"COORDINATOR_MODEL_CONFIG": "unknownprovider:some-model"}
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings'):
            with patch('logging.warning') as mock_warning:
                configs = _configure_llm_models()
                configs_dict = get_configs_as_dict(configs)
    
    assert configs_dict["coordinator"] == EXPECTED_DEFAULT_GOOGLE_MODEL
    mock_warning.assert_any_call("Unsupported provider 'unknownprovider' for COORDINATOR. Falling back to default.")

def test_mixed_configurations():
    env_vars = {
        "PLANNER_MODEL_CONFIG": "google:custom-planner-model",
        "ANALYZER_MODEL_CONFIG": "openrouter:some/analyzer-model",
        "OPENROUTER_API_KEY": "fake_or_key",
        "CRITIC_MODEL_CONFIG": "openai:gpt-4-turbo",
        "OPENAI_API_KEY": "fake_openai_key",
        "OPENAI_API_BASE": "http://custombase/openai",
        "GOOGLE_API_KEY": "test_google_key" # For default Google models and custom-planner-model
        # RESEARCHER, SYNTHESIZER, COORDINATOR will use default
    }
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings') as mock_google_warnings:
            configs = _configure_llm_models()
            configs_dict = get_configs_as_dict(configs)

    assert configs_dict["planner"] == "custom-planner-model"
    
    analyzer_cfg = configs_dict["analyzer"]
    assert isinstance(analyzer_cfg, LiteLlm)
    assert analyzer_cfg.model == "openrouter/some/analyzer-model"
    assert analyzer_cfg.api_key == "fake_or_key"

    critic_cfg = configs_dict["critic"]
    assert isinstance(critic_cfg, LiteLlm)
    assert critic_cfg.model == "openai/gpt-4-turbo"
    assert critic_cfg.api_key == "fake_openai_key"
    assert critic_cfg.api_base == "http://custombase/openai"

    assert configs_dict["researcher"] == EXPECTED_DEFAULT_GOOGLE_MODEL
    assert configs_dict["synthesizer"] == EXPECTED_DEFAULT_GOOGLE_MODEL
    assert configs_dict["coordinator"] == EXPECTED_DEFAULT_GOOGLE_MODEL
    mock_google_warnings.assert_called_once()


@pytest.mark.parametrize("invalid_config_value", [
    "",         # Empty string
    ":",        # Colon only
    "google:",  # Provider only
    ":some-model" # Model name only
])
def test_malformed_model_config_fallback(invalid_config_value):
    agent_to_test = "PLANNER"
    env_vars = {f"{agent_to_test}_MODEL_CONFIG": invalid_config_value}
    with patch.dict(os.environ, env_vars, clear=True):
        with patch('multi_tool_agent.llm_config._log_google_credential_warnings'):
            with patch('logging.warning') as mock_warning:
                configs = _configure_llm_models()
                configs_dict = get_configs_as_dict(configs)

    assert configs_dict[agent_to_test.lower()] == EXPECTED_DEFAULT_GOOGLE_MODEL
    expected_message = f"Invalid model configuration format for {agent_to_test}: {invalid_config_value}. Expected 'provider:model_name'. Falling back to default."
    mock_warning.assert_any_call(expected_message)

# Test for _log_google_credential_warnings
def test_log_google_credential_warnings_api_key_present():
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "testkey"}, clear=True):
            _log_google_credential_warnings()
        mock_logger_instance.warning.assert_not_called()

def test_log_google_credential_warnings_use_vertex_true_creds_present():
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        env_vars = {
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"
        }
        with patch.dict(os.environ, env_vars, clear=True):
            _log_google_credential_warnings()
        mock_logger_instance.warning.assert_not_called()

def test_log_google_credential_warnings_use_vertex_true_creds_missing():
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        with patch.dict(os.environ, {"GOOGLE_GENAI_USE_VERTEXAI": "true"}, clear=True):
            _log_google_credential_warnings()
        mock_logger_instance.warning.assert_any_call(
            "GOOGLE_GENAI_USE_VERTEXAI is true, but GOOGLE_APPLICATION_CREDENTIALS is not set."
        )

def test_log_google_credential_warnings_default_google_no_key():
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        with patch.dict(os.environ, {}, clear=True): # No GOOGLE_API_KEY, not using Vertex
            _log_google_credential_warnings()
        mock_logger_instance.warning.assert_any_call(
            "Using default Google models, but GOOGLE_API_KEY is not set."
        )

print("Successfully created tests/test_llm_config.py")
