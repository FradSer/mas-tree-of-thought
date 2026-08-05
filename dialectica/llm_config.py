"""Dynamic LLM configuration factory for ADK 2.0.

Reads DEFAULT_MODEL_CONFIG from environment (provider:model_name format).
Provides a factory to create model configs for dynamically spawned agents.
"""

import logging
import os
from typing import Any

from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-3.5-flash"


def _log_credential_warnings(model_name: str) -> None:
    """Log warnings if required Google credentials are missing."""
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    if use_vertex:
        if not os.environ.get("GOOGLE_CLOUD_PROJECT") or not os.environ.get(
            "GOOGLE_CLOUD_LOCATION"
        ):
            logger.warning(
                "Using Vertex AI with '%s', but GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set.",
                model_name,
            )
    else:
        if not os.environ.get("GOOGLE_API_KEY"):
            logger.warning(
                "Using Google AI Studio with '%s', but GOOGLE_API_KEY not set.",
                model_name,
            )


def _parse_model_config(config_str: str) -> str | LiteLlm:
    """Parse 'provider:model_name' into an ADK-compatible model config.

    Supported providers: 'google', 'openrouter', 'openai'.
    Returns model name string for Google, or LiteLlm instance for others.

    A bare model name with no 'provider:' prefix (or an empty string) silently
    falls back to ``_DEFAULT_MODEL`` with NO warning logged — unlike every
    other fallback path in this function. Always pass the ``provider:model``
    form (e.g. ``"google:gemini-3.1-pro-preview"``), not a bare model name.
    """
    if not config_str or ":" not in config_str:
        return _DEFAULT_MODEL

    try:
        provider, model_name = config_str.strip().split(":", 1)
        provider = provider.lower()

        if provider == "google":
            _log_credential_warnings(model_name)
            return model_name

        if provider == "openrouter":
            if os.environ.get("OPENROUTER_API_KEY"):
                logger.info("OpenRouter model: %s", model_name)
                return LiteLlm(model=model_name)
            logger.warning(
                "OPENROUTER_API_KEY not set, falling back to %s", _DEFAULT_MODEL
            )
            return _DEFAULT_MODEL

        if provider == "openai":
            api_base = os.environ.get("OPENAI_API_BASE")
            if os.environ.get("OPENAI_API_KEY") and api_base:
                openai_model = f"openai/{model_name}"
                logger.info("OpenAI-compatible model: %s at %s", openai_model, api_base)
                # Pass api_base explicitly: recent LiteLLM no longer reads the
                # OPENAI_API_BASE env var for the 'openai/'-prefixed custom base.
                kwargs: dict[str, Any] = {"api_base": api_base}
                # Qwen/thinking models expose a long reasoning trace by default,
                # which dominates latency for code evals. Allow disabling it via
                # the standard chat_template_kwargs flag (no-op for non-Qwen).
                if os.environ.get("DIALECTICA_DISABLE_THINKING", "").lower() == "true":
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                return LiteLlm(model=openai_model, **kwargs)
            logger.warning(
                "OpenAI credentials missing, falling back to %s", _DEFAULT_MODEL
            )
            return _DEFAULT_MODEL

        logger.warning(
            "Unknown provider '%s', falling back to %s", provider, _DEFAULT_MODEL
        )
        return _DEFAULT_MODEL

    except (ValueError, KeyError) as e:
        logger.error("Failed to parse model config '%s': %s", config_str, e)
        return _DEFAULT_MODEL


def get_model_config(role: str | None = None) -> str | LiteLlm:
    """Get model config for a role, with optional role-specific env override.

    Checks {ROLE}_MODEL_CONFIG first, then DEFAULT_MODEL_CONFIG.
    Falls back to gemini-3.5-flash if neither is set.
    """
    if role:
        role_override = os.environ.get(f"{role.upper()}_MODEL_CONFIG")
        if role_override:
            logger.info("Role-specific config for '%s': %s", role, role_override)
            return _parse_model_config(role_override)

    default_str = os.environ.get("DEFAULT_MODEL_CONFIG", f"google:{_DEFAULT_MODEL}")
    return _parse_model_config(default_str)
