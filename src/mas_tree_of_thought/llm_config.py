import os
import logging
from typing import Tuple, Dict, Optional

from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger(__name__)

# --- Model Configuration ---

def _configure_llm_models() -> Tuple[
    # Planner, Researcher, Analyzer, Critic, Synthesizer, Coordinator
    str | LiteLlm,
    str | LiteLlm,
    str | LiteLlm,
    str | LiteLlm,
    str | LiteLlm,
    str | LiteLlm,
]:
    """
    Configure LLM models for each specialist agent and the coordinator.

    Reads configurations from environment variables (e.g., PLANNER_MODEL_CONFIG=provider:model_name).
    Supported providers: 'google', 'openrouter', 'openai'.

    Falls back to a default Google model if:
    - Environment variable is not set or invalid
    - OpenRouter is specified without an API key
    - OpenAI is specified without required API key/base

    Environment Variables:
    - <AGENT_NAME>_MODEL_CONFIG: Specifies provider and model (e.g., "openrouter:google/gemini-2.5-pro", "openai:gpt-4o")
    - GOOGLE_GENAI_USE_VERTEXAI: Set to "true" to use Vertex AI (requires GOOGLE_CLOUD_PROJECT/LOCATION)
    - OPENROUTER_API_KEY: Required for OpenRouter models
    - OPENAI_API_KEY: Required for OpenAI models (or compatible endpoints)
    - OPENAI_API_BASE: Required for OpenAI models (or compatible endpoints)
    - GOOGLE_API_KEY: Required for Google AI Studio models
    - GOOGLE_CLOUD_PROJECT: Required for Vertex AI
    - GOOGLE_CLOUD_LOCATION: Required for Vertex AI

    Returns:
        Tuple[str | LiteLlm, ...]: A tuple containing model configurations for
                                  (Planner, Researcher, Analyzer, Critic, Synthesizer, Coordinator)
    """
    # Default Google model if no specific configuration is found
    default_google_model = "gemini-2.0-flash"

    # Flags for API usage (Vertex vs Google AI Studio)
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY") # Added OpenAI Key check
    openai_base = os.environ.get("OPENAI_API_BASE") # Added OpenAI Base check

    # Agent names must match the order in the return tuple
    agent_names = ["planner", "researcher", "analyzer", "critic", "synthesizer", "coordinator"]
    agent_configs: Dict[str, str | LiteLlm] = {}

    logger.info("--- Configuring Specialist Agent LLMs ---")

    for agent_name in agent_names:
        env_var_name = f"{agent_name.upper()}_MODEL_CONFIG"
        config_str = os.environ.get(env_var_name)
        final_config: str | LiteLlm = default_google_model # Default config

        if config_str:
            logger.info(f"Found config for {agent_name.capitalize()} from {env_var_name}: '{config_str}'")
            try:
                provider, model_name = config_str.strip().split(":", 1)
                provider = provider.lower()

                if provider == "openrouter":
                    if openrouter_key:
                        try:
                            final_config = LiteLlm(model=f"openrouter/{model_name}")
                            logger.info(f"  -> Configured {agent_name.capitalize()} for OpenRouter: openrouter/{model_name}")
                        except Exception as e:
                            logger.error(f"  -> Failed to configure LiteLlm for {agent_name.capitalize()} (openrouter/{model_name}): {e}. Falling back to default ({default_google_model}).")
                            final_config = default_google_model # Fallback
                            # Log fallback credential warnings
                            _log_google_credential_warnings(use_vertex, default_google_model, is_fallback=True)
                    else:
                        logger.warning(f"  -> OpenRouter specified for {agent_name.capitalize()} ('{model_name}'), but OPENROUTER_API_KEY not found. Falling back to default ({default_google_model}).")
                        final_config = default_google_model # Fallback
                        _log_google_credential_warnings(use_vertex, default_google_model, is_fallback=True)

                elif provider == "openai": # Added OpenAI provider block
                    if openai_key and openai_base:
                        try:
                            # LiteLLM uses API key/base from env vars automatically when specified.
                            # The model string often doesn't need the provider prefix if base/key are set.
                            # Prepend 'openai/' to the model name for LiteLlm
                            # This explicitly tells LiteLLM to use the OpenAI logic
                            # and rely on OPENAI_API_KEY/OPENAI_API_BASE env vars.
                            final_config = LiteLlm(model=f"openai/{model_name}") # Corrected line
                            logger.info(f"  -> Configured {agent_name.capitalize()} for OpenAI compatible endpoint: openai/{model_name} (using OPENAI_API_KEY & OPENAI_API_BASE)")
                        except Exception as e:
                            logger.error(f"  -> Failed to configure LiteLlm for {agent_name.capitalize()} using OpenAI provider (openai/{model_name}): {e}. Falling back to default ({default_google_model}).")
                            final_config = default_google_model # Fallback
                            _log_google_credential_warnings(use_vertex, default_google_model, is_fallback=True)
                    else:
                        missing_vars = []
                        if not openai_key: missing_vars.append("OPENAI_API_KEY")
                        if not openai_base: missing_vars.append("OPENAI_API_BASE")
                        logger.warning(f"  -> OpenAI provider specified for {agent_name.capitalize()} ('{model_name}'), but required environment variable(s) [{', '.join(missing_vars)}] not found. Falling back to default ({default_google_model}).")
                        final_config = default_google_model # Fallback
                        _log_google_credential_warnings(use_vertex, default_google_model, is_fallback=True)

                elif provider == "google":
                    final_config = model_name
                    logger.info(f"  -> Configured {agent_name.capitalize()} for Google model: {model_name}")
                    _log_google_credential_warnings(use_vertex, model_name)
                else:
                    logger.warning(f"  -> Invalid provider '{provider}' in {env_var_name}. Falling back to default ({default_google_model}).")
                    final_config = default_google_model # Fallback
                    _log_google_credential_warnings(use_vertex, default_google_model, is_fallback=True)

            except ValueError:
                logger.warning(f"  -> Invalid format in {env_var_name} (expected 'provider:model_name'). Falling back to default ({default_google_model}).")
                final_config = default_google_model # Fallback
                _log_google_credential_warnings(use_vertex, default_google_model, is_fallback=True)
            except Exception as e:
                 logger.error(f"  -> Error processing {env_var_name} ('{config_str}'): {e}. Falling back to default ({default_google_model}).")
                 final_config = default_google_model # Fallback
                 _log_google_credential_warnings(use_vertex, default_google_model, is_fallback=True)

        else:
            # Environment variable not set, use default
            logger.info(f"{agent_name.capitalize()} using default Google model: {default_google_model} (set {env_var_name} to override).")
            final_config = default_google_model
            _log_google_credential_warnings(use_vertex, default_google_model)

        agent_configs[agent_name] = final_config

    logger.info("--- LLM Configuration Complete ---")

    # Return configurations in the specified order
    return (
        agent_configs["planner"],
        agent_configs["researcher"],
        agent_configs["analyzer"],
        agent_configs["critic"],
        agent_configs["synthesizer"],
        agent_configs["coordinator"],
    )

# --- Helper for Credential Warnings ---
def _log_google_credential_warnings(use_vertex: bool, model_name: str, is_fallback: bool = False):
    """Logs warnings if required Google credentials are missing."""
    prefix = f"     ({'Fallback ' if is_fallback else ''}Google model '{model_name}' "
    if use_vertex:
        if not os.environ.get("GOOGLE_CLOUD_PROJECT") or not os.environ.get("GOOGLE_CLOUD_LOCATION"):
            logger.warning(f"{prefix}using Vertex AI, but GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION env vars not set.)")
    else:
        if not os.environ.get("GOOGLE_API_KEY"):
            logger.warning(f"{prefix}using Google AI Studio, but GOOGLE_API_KEY env var not set.)")

# --- Exported Model Configurations ---
(
    planner_model_config,
    researcher_model_config,
    analyzer_model_config,
    critic_model_config,
    synthesizer_model_config,
    coordinator_model_config,
) = _configure_llm_models() 