import os  # Environment variable access
import logging # Standard logging library
import re # Regular expressions for score parsing
import time # Time library for rate limiting delays
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple

from google.genai import types # Google AI types for Event content

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent # Explicit import for LlmAgent type hint
from google.adk.tools import google_search, FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm  # For using OpenRouter models
from google.adk.events import Event
from typing_extensions import override
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Thought Data Model ---

class ThoughtData(BaseModel):
    """
    Data model for a single node in the Tree of Thoughts.

    Used for validating arguments passed to the thought registration/validation tool.
    Ensures consistency and structural integrity of metadata for each thought node
    managed by the Coordinator agent.
    """
    parentId: Optional[str] = Field(
        None,
        description="ID of the parent thought node. Null for the root node."
    )
    thoughtId: str = Field(
        ...,
        description="Unique identifier for this thought node (e.g., 'node-0', 'node-1a')."
    )
    thought: str = Field(
        ...,
        description="Core content of the current thought/step.",
        min_length=1
    )
    # Score assigned by the evaluation process (higher is better)
    evaluationScore: Optional[float] = Field(
        None,
        description="Score reflecting the promise of this thought path (0-10)."
    )
    status: str = Field(
        "active",
        description="Current status (e.g., 'active', 'evaluated', 'pruned')."
    )
    # Depth of this node in the tree (root is 0)
    depth: int = Field(
        ...,
        description="Node depth in the tree, starting from 0 for the root.",
        ge=0
    )

    # Pydantic model configuration
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # --- Validators ---
    @field_validator('parentId')
    @classmethod
    def validate_parent_id_format(cls, v: Optional[str]) -> Optional[str]:
        # Basic check for parent ID format
        if v is not None and not isinstance(v, str):
            raise ValueError('parentId must be a string if set')
        return v

    def dict(self) -> Dict[str, Any]:
        """Convert thought data to a dictionary for serialization."""
        return self.model_dump(exclude_none=True)

# --- Sequential Thinking Tool renamed and repurposed ---
# The original Sequential Thinking Tool has been adapted to serve as a
# validator for thought node metadata within the ToT framework.

def validate_thought_node_data(
    parentId: Optional[str],
    thoughtId: str,
    thought: str,
    depth: int,
    evaluationScore: Optional[float] = None,
    status: str = "active"
) -> Dict[str, Any]:
    """
    Validate metadata for a single node in the Tree of Thoughts.

    This function acts as a validation layer before the Coordinator adds a node
    to the thought tree. It uses the ThoughtData model to ensure data integrity.
    It does NOT perform the thinking or manage the tree structure itself;
    that responsibility lies with the calling Coordinator agent.

    Args:
        parentId (Optional[str]): ID of the parent node (None for root)
        thoughtId (str): Unique ID for this new thought node
        thought (str): Core content/idea of this thought node
        depth (int): Depth of this node in the tree (root=0)
        evaluationScore (Optional[float]): Initial evaluation score (optional)
        status (str): Initial status of the node (default: 'active')

    Returns:
        Dict[str, Any]: Dictionary containing validated metadata if successful,
                        or an error dictionary if validation fails.
                        Includes 'validation_status' key.
    """
    try:
        # Ensure critical params are provided
        if thought is None or thought.strip() == "":
            return {
                "error": "The 'thought' parameter cannot be empty",
                "validation_status": "validation_error"
            }
            
        if thoughtId is None or thoughtId.strip() == "":
            return {
                "error": "The 'thoughtId' parameter cannot be empty",
                "validation_status": "validation_error"
            }
            
        if depth is None:
            return {
                "error": "The 'depth' parameter is required",
                "validation_status": "validation_error"
            }
            
        # --- Validation using Pydantic Model ---
        thought_input_data = {
            "parentId": parentId,
            "thoughtId": thoughtId,
            "thought": thought,
            "depth": depth,
            "evaluationScore": evaluationScore,
            "status": status,
        }
        current_thought_data = ThoughtData(**thought_input_data)

        # --- Build Result for Agent ---
        # Return validated data, Coordinator handles adding to tree
        result_data = {
            "validatedThoughtId": current_thought_data.thoughtId,
            "parentId": current_thought_data.parentId,
            "depth": current_thought_data.depth,
            "status": current_thought_data.status,
            "thoughtContent": current_thought_data.thought,
            "evaluationScore": current_thought_data.evaluationScore,
            "validation_status": "success",
            "message": f"Thought node '{current_thought_data.thoughtId}' data is valid. Thought content: '{current_thought_data.thought[:100]}'"
        }
        return result_data

    except ValidationError as e:
        return {
            "error": f"Input validation failed: {e.errors()}",
            "validation_status": "validation_error"
        }
    except Exception as e:
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "validation_status": "failed"
        }

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
                            final_config = LiteLlm(model=model_name)
                            logger.info(f"  -> Configured {agent_name.capitalize()} for OpenRouter: {model_name}")
                        except Exception as e:
                            logger.error(f"  -> Failed to configure LiteLlm for {agent_name.capitalize()} ({model_name}): {e}. Falling back to default ({default_google_model}).")
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

# --- Specialist Agent Definitions ---
# These agents perform specific sub-tasks delegated by the ToT Coordinator.
# They operate based on their assigned instructions and configured models.
# Tools can be added to enhance their capabilities (e.g., google_search for Researcher).

# Get individual model configurations immediately before use
(
    planner_config,
    researcher_config,
    analyzer_config,
    critic_config,
    synthesizer_config,
    coordinator_config,
) = _configure_llm_models()

# Helper to create standard instructions
def _create_agent_instruction(specialist_name: str, core_task: str, extra_guidance: List[str] = []) -> str:
    """
    Create a standardized instruction string for specialist agents.

    This helper ensures consistent formatting and includes common elements
    like the current date/time and the agent's role.

    Args:
        specialist_name (str): Name of the specialist agent (e.g., "Strategic Planner")
        core_task (str): Primary task description for the agent
        extra_guidance (List[str], optional): Additional instructions or constraints

    Returns:
        str: Formatted instruction string with newlines escaped for ADK
    """
    base_instruction = [
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n",
        f"You are the {specialist_name} specialist.\\n",
        "You receive specific sub-tasks from the Coordinator.\\n",
        "**Your Task:**\\n",
        core_task,
    ]
    base_instruction.extend(extra_guidance)
    return "\\n".join(base_instruction) # Use \\n for literal newline in ADK instructions

# Planner Agent
planner_agent = Agent(
    name="Planner",
    model=planner_config, # Use specific config
    description="Develops strategic plans and roadmaps based on delegated sub-tasks. Identifies alternative options.",
    instruction=_create_agent_instruction(
        specialist_name="Strategic Planner",
        core_task=(
            "Your primary role is to generate 'thoughts'. Remember, a 'thought' represents an intermediate step or a partial solution towards solving the overall problem.\\n"
            " 1. Read the specific planning instruction provided (e.g., generate initial strategies, expand on a current thought).\\n"
            " 2. Generate potential 'thoughts' (intermediate steps, partial solutions, strategies, next actions, or questions) based on the instruction.\\n"
            " 3. **Crucially: If multiple viable strategies or paths exist, you MUST explicitly list them as distinct options/thoughts** for the Coordinator to consider for branching. Treat each distinct option as a separate 'thought'. **Ensure these thoughts represent genuinely different directions or approaches, not just variations of the same core idea.**\\n"
            " 4. Identify potential roadblocks or critical decision points within your generated thoughts.\\n"
            " 5. **Output Formatting:** Return ONLY the generated thoughts, each on a new line. Avoid introductory phrases or summaries.\\n"
            " **Example Scenario 1 (Initial Strategies for 'Write a story about a lost dog'):**\\n"
            "   Thought 1: Focus on the dog's perspective and journey home.\\n"
            "   Thought 2: Focus on the owner's search and emotional state.\\n"
            "   Thought 3: Introduce a kind stranger who helps the dog.\\n"
            " **Example Scenario 2 (Next steps for 'Focus on the dog's perspective'):**\\n"
            "   Thought 1: Describe the moment the dog realized it was lost.\\n"
            "   Thought 2: Detail the dog's first encounter with an unfamiliar challenge (e.g., crossing a busy street).\\n"
            "   Thought 3: Explore the dog's memories of its owner."


        ),
        # Note: The previous numbered list was integrated into the core_task description.
        # Removed the explicit reading from session state here as the user indicated issues.
        # The dynamic instructions in helper methods will still attempt to set state.
    ),
    input_schema=None,
    tools=[]
)

# Researcher Agent
researcher_agent = Agent(
    name="Researcher",
    model=researcher_config, # Use specific config
    description="Gathers and validates information, highlighting conflicts and uncertainties.",
    instruction=_create_agent_instruction(
        specialist_name="Information Gatherer",
        core_task=(
            " 1. Read the specific research task provided by the Coordinator.\\n"
            " 2. **Actively use the `google_search` tool** for external information gathering based on the task.\\n"
            " 3. Validate information where possible.\\n"
            " 4. Structure findings clearly.\\n"
            " 5. **Crucially: You MUST explicitly report any significant conflicting information, major uncertainties, or data points suggesting multiple possible interpretations.** These are vital signals for the Coordinator.\\n"
            " 6. Return ONLY the research findings, highlighting conflicts/uncertainties."
        ),
        extra_guidance=[] # No extra guidance needed beyond core task
    ),
    tools=[google_search] # Add google_search tool here
)

# Analyzer Agent
analyzer_agent = Agent(
    name="Analyzer",
    model=analyzer_config, # Use specific config
    description="Performs analysis, identifying inconsistencies and assumptions, and provides a structured evaluation score.",
    instruction=_create_agent_instruction(
        specialist_name="Core Analyst",
        core_task=(
            " 1. Read the analysis task provided by the Coordinator (which includes the thought and research findings).\\n"
            " 2. Analyze the soundness, feasibility, and logical consistency of the thought, considering the research findings.\\n"
            " 3. **Crucially: Identify any invalidated assumptions, logical inconsistencies, or contradictions**, especially if they impact previous conclusions.\\n"
            " 4. Generate concise insights based on your analysis.\\n"
            " 5. **Provide a structured evaluation score:** At the end of your response, include a line formatted exactly as: `Evaluation Score: [score]/10`, where [score] is an integer from 0 (lowest promise/soundness) to 10 (highest promise/soundness) based on your analysis.\\n"
            " 6. **Provide a termination recommendation:** Include a line formatted exactly as: `Termination Recommendation: [True/False]` (True if path should stop, False if it should continue).\\n"
            " 7. Return ONLY the analysis, insights, highlighted inconsistencies, the score line, and the termination recommendation line."
        )
    ),
    input_schema=None,
    tools=[]
)

# Critic Agent
critic_agent = Agent(
    name="Critic",
    model=critic_config, # Use specific config
    description="Critically evaluates ideas, identifies flaws, MUST suggest alternatives/revisions, and provides a structured evaluation score.",
    instruction=_create_agent_instruction(
        specialist_name="Quality Controller (Critic)",
        core_task=(
            " 1. Read the critique task provided by the Coordinator (which includes the thought and research findings).\\n"
            " 2. Critically evaluate the thought for flaws, biases, or weaknesses, considering the research findings.\\n"
            " 3. Identify potential biases, flaws, logical fallacies, or weaknesses.\\n"
            " 4. **Crucially: Suggest specific improvements AND propose at least one concrete alternative approach or direction.** If a major flaw invalidates a previous thought, clearly state this and suggest targeting that thought for revision.\\n"
            " 5. Formulate a constructive response containing your evaluation, identified flaws, and mandatory suggestions/alternatives.\\n"
            " 6. **Provide a structured evaluation score:** At the end of your response, include a line formatted exactly as: `Evaluation Score: [score]/10`, where [score] is an integer from 0 (lowest viability/promise) to 10 (highest viability/promise) based on your critique.\\n"
            " 7. **Provide a termination recommendation:** Include a line formatted exactly as: `Termination Recommendation: [True/False]` (True if path should stop, False if it should continue).\\n"
            " 8. Return ONLY the critique, suggestions/alternatives, the score line, and the termination recommendation line."
        )
    ),
    input_schema=None,
    tools=[]
)

# Synthesizer Agent
synthesizer_agent = Agent(
    name="Synthesizer",
    model=synthesizer_config, # Use specific config
    description="Integrates information or forms conclusions based on delegated synthesis sub-tasks.",
    instruction=_create_agent_instruction(
        specialist_name="Integration Specialist",
        core_task=(
            " 1. Read the synthesis task provided by the Coordinator (which includes the initial problem and high-scoring thoughts).\\n"
            " 2. Integrate the information from the provided thoughts to address the initial problem.\\n"
            " 3. Connect elements, identify overarching themes, draw conclusions, or formulate the final answer as requested.\\n"
            " 4. Distill inputs into clear, structured insights.\\n"
            " 5. Formulate a response presenting the synthesized information or conclusion.\\n"
            " 6. Return ONLY the synthesized output."
        )
    ),
    input_schema=None,
    tools=[]
)

# --- Custom Coordinator Agent Definition ---

class ToTCoordinator(BaseAgent):
    """
    Tree of Thoughts (ToT) Coordinator with Beam Search Implementation.

    This coordinator orchestrates a multi-agent workflow using Tree of Thoughts (ToT) 
    methodology combined with Beam Search for efficient exploration of solution spaces.

    Key Components:
    - Tree Management: Maintains a tree structure where each node represents a thought/step
    - Beam Search: Keeps track of k most promising nodes at each level
    - State Management: Handles thought_tree, active_beam, and workflow phases
    - Agent Coordination: Delegates tasks to specialist agents (Planner, Researcher, etc.)

    Workflow Phases:
    1. Initialization: Create root node and generate initial strategies
    2. Main Loop: 
       - Generate candidate thoughts from active beam
       - Evaluate thoughts using research, analysis, and critique
       - Select best k nodes for next beam
    3. Synthesis: Generate final result from best path

    The coordinator ensures robust error handling, logging, and state management
    throughout the workflow execution.
    """

    # --- Field Declarations for Pydantic ---
    # Declare sub-agents and tools as instance attributes with type hints
    planner: LlmAgent
    researcher: LlmAgent  # Research agent for information gathering
    analyzer: LlmAgent
    critic: LlmAgent
    synthesizer: LlmAgent
    validator: FunctionTool

    # beam_width: int = Field(default=3, description="Number of top nodes (k) to keep in the beam.") # REMOVED
    # max_depth: int = Field(default=5, description="Maximum depth of the tree to explore.") # REMOVED
    model: LlmAgent | LiteLlm | str

    # --- Rate Limiting Configuration ---
    use_free_tier_rate_limiting: bool = Field(default=False, description="Enable rate limiting for Google AI Studio free tier.")
    free_tier_sleep_seconds: float = Field(default=2.0, description="Seconds to sleep between calls when rate limiting.")
    use_vertex_ai: bool = Field(default=False, description="Whether Vertex AI is configured.")
    # --- End Rate Limiting Fields ---

    # Allow arbitrary types in Pydantic model
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        planner: LlmAgent,
        researcher: LlmAgent,
        analyzer: LlmAgent,
        critic: LlmAgent,
        synthesizer: LlmAgent,
        validator: FunctionTool,
        # beam_width: int = 3, # REMOVED
        # max_depth: int = 5, # REMOVED
        model: LlmAgent | LiteLlm | str = None,
    ):
        """
        Initialize the Tree of Thoughts Beam Search Coordinator.

        This coordinator manages a team of specialist agents to explore and solve complex problems
        using a tree-based approach with beam search optimization.

        Args:
            name (str): Identifier for this coordinator instance
            planner (LlmAgent): Agent responsible for generating strategic thoughts and next steps
            researcher (LlmAgent): Agent for gathering and validating information about thoughts
            analyzer (LlmAgent): Agent for evaluating thought soundness and feasibility
            critic (LlmAgent): Agent for identifying flaws and suggesting improvements
            synthesizer (LlmAgent): Agent for combining information and generating final results
            validator (FunctionTool): Tool for validating thought node data structure
            # beam_width (int, optional): Number of top nodes to maintain in beam. Defaults to 3 # REMOVED
            # max_depth (int, optional): Maximum depth to explore in thought tree. Defaults to 5 # REMOVED
            model (Union[LlmAgent, LiteLlm, str], optional): Model for coordinator's own processing. Defaults to None

        Configuration:
            - Rate limiting and API configurations are read from environment variables
            - Sub-agents are organized into a framework-compatible list
            - State management tools are initialized for tracking the thought tree
        """
        # --- Read Rate Limiting Env Vars (overriding defaults if set) --- 
        use_free_tier_rate_limiting_env = os.environ.get("USE_FREE_TIER_RATE_LIMITING", "false").lower() == "true"
        try:
            free_tier_sleep_seconds_env = float(os.environ.get("FREE_TIER_SLEEP_SECONDS", "2.0"))
        except ValueError:
            logger.warning(f"Invalid FREE_TIER_SLEEP_SECONDS value. Using default: 2.0")
            free_tier_sleep_seconds_env = 2.0
        use_vertex_ai_env = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
        # --- End Reading Env Vars ---

        # Define the sub_agents list for the framework - these are the agents
        # that this custom agent will directly invoke in its _run_async_impl
        sub_agents_list = [
            planner,
            researcher, # Add researcher here
            analyzer,
            critic,
            synthesizer,
        ]
        
        # Call super().__init__() with all required parameters
        # Pass the values read from environment or defaults to super init
        super().__init__(
            name=name,
            # description=f"Tree of Thoughts Beam Search (width={beam_width}, depth={max_depth})", # UPDATED
            description=f"Tree of Thoughts Coordinator with LLM-driven exploration",
            model=model,
            planner=planner,
            researcher=researcher, # Pass researcher to super
            analyzer=analyzer,
            critic=critic,
            synthesizer=synthesizer,
            validator=validator,
            # beam_width=beam_width, # REMOVED
            # max_depth=max_depth, # REMOVED
            use_free_tier_rate_limiting=use_free_tier_rate_limiting_env,
            free_tier_sleep_seconds=free_tier_sleep_seconds_env,
            use_vertex_ai=use_vertex_ai_env, # Pass Vertex status too
            sub_agents=sub_agents_list,  # Pass the explicit sub_agents list
        )
        
        # Log initialization
        # logger.info(f"[{self.name}] ToT Coordinator initialized with beam width {beam_width} and max depth {max_depth}") # UPDATED
        logger.info(f"[{self.name}] ToT Coordinator initialized for LLM-driven exploration.")

    # --- Helper methods for state management ---
    def _get_state_value(self, ctx: InvocationContext, key: str, default: Any = None) -> Any:
        """
        Safely retrieve a value from the session state.

        Args:
            ctx (InvocationContext): Current invocation context
            key (str): Key to retrieve from state
            default (Any, optional): Default value if key not found. Defaults to None

        Returns:
            Any: Retrieved value or default if not found
        """
        return ctx.session.state.get(key, default)

    def _set_state_value(self, ctx: InvocationContext, key: str, value: Any):
        """
        Set a value in the session state.

        Args:
            ctx (InvocationContext): Current invocation context
            key (str): Key to store value under
            value (Any): Value to store
        """
        ctx.session.state[key] = value

    def _get_thought_tree(self, ctx: InvocationContext) -> Dict[str, Any]:
        """
        Retrieve the thought tree from state, initializing if not present.

        The thought tree is a dictionary storing all nodes and their relationships,
        representing the current state of problem exploration.

        Args:
            ctx (InvocationContext): Current invocation context

        Returns:
            Dict[str, Any]: The thought tree structure
        """
        return ctx.session.state.setdefault("thought_tree", {})

    def _get_active_beam(self, ctx: InvocationContext) -> List[str]:
        """
        Retrieve the active beam from state, initializing if not present.

        The active beam contains IDs of the current most promising nodes
        being explored in the beam search process.

        Args:
            ctx (InvocationContext): Current invocation context

        Returns:
            List[str]: List of node IDs in the current beam
        """
        return ctx.session.state.setdefault("active_beam", [])

    def _set_active_beam(self, ctx: InvocationContext, beam: List[str]):
        """
        Update the active beam in state.

        Args:
            ctx (InvocationContext): Current invocation context
            beam (List[str]): New list of node IDs to set as active beam
        """
        self._set_state_value(ctx, "active_beam", beam)
        
    def _update_node(self, ctx: InvocationContext, node_id: str, data: Dict[str, Any]):
        """
        Update a node's data in the thought tree.

        This method handles both creating new nodes and updating existing ones,
        ensuring that existing data is properly merged with updates.

        Args:
            ctx (InvocationContext): Current invocation context
            node_id (str): Identifier of the node to update
            data (Dict[str, Any]): New data to merge into the node
        """
        tree = self._get_thought_tree(ctx)
        if node_id in tree:
            # Ensure existing data isn't completely overwritten if not present in new data
            existing_data = tree[node_id]
            existing_data.update(data)
            tree[node_id] = existing_data
        else:
            tree[node_id] = data

    # --- Helper method for score extraction ---
    def _extract_score(self, text: str) -> Optional[float]:
        """
        Extract evaluation score from agent output text.

        Looks for patterns like 'Evaluation Score: X/10' and converts to float.
        Includes validation and error handling for robustness.

        Args:
            text (str): Text to extract score from

        Returns:
            Optional[float]: Extracted score between 0-10, or None if not found/invalid
        """
        if not isinstance(text, str):
            return None
        match = re.search(r"Evaluation Score:\s*(\d{1,2}(?:\.\d+)?)/10", text)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 10:
                    return score
                else:
                    logger.warning(f"Extracted score {score} out of range (0-10). Ignoring.")
                    return None
            except ValueError:
                logger.warning(f"Could not convert extracted score '{match.group(1)}' to float.")
                return None
        logger.debug(f"Could not find score pattern in text: '{text[:100]}...'")
        return None

    # --- Helper method for termination recommendation extraction ---
    def _extract_termination_recommendation(self, text: str) -> bool:
        """
        Extract termination recommendation from agent output text.

        Looks for patterns like 'Termination Recommendation: True' or 'Termination Recommendation: False'.
        Defaults to False if pattern not found or invalid.

        Args:
            text (str): Text to extract recommendation from

        Returns:
            bool: True if termination is recommended, False otherwise.
        """
        if not isinstance(text, str):
            return False
        # Match "Termination Recommendation:" followed by True or False (case-insensitive)
        match = re.search(r"Termination Recommendation:\s*(True|False)", text, re.IGNORECASE)
        if match:
            recommendation = match.group(1).lower()
            return recommendation == 'true'
        logger.debug(f"Could not find termination recommendation pattern in text: '{text[:100]}...'")
        return False # Default to False if not explicitly found

    # --- Helper method to safely run Sub Agents and filter empty model events ---
    async def _run_sub_agent_safely(self, agent: LlmAgent, ctx: InvocationContext, dynamic_instruction: str | None = None) -> AsyncGenerator[Event, None]:
        """
        Safely execute a sub-agent, passing dynamic instruction via ctx.user_content.

        Handles rate limits and filters empty model events.

        Args:
            agent (LlmAgent): The specialist agent to run
            ctx (InvocationContext): Current invocation context (user_content will be temporarily modified)
            dynamic_instruction (str | None, optional): Specific instruction/task for this run.

        Yields:
            Event: Valid events from the sub-agent execution

        Note:
            Rate limiting is only applied for Google AI Studio models when enabled.
            This method temporarily REPLACES ctx.user_content for the sub-agent call.
        """
        # --- Rate Limiting Check --- 
        is_google_studio_model = (isinstance(agent.model, str) and 
                                not self.use_vertex_ai)

        if self.use_free_tier_rate_limiting and is_google_studio_model:
            sleep_duration = self.free_tier_sleep_seconds
            logger.info(f"[{self.name}] Rate limit active: Sleeping for {sleep_duration:.1f}s before calling {agent.name}")
            time.sleep(sleep_duration)

        # --- Context Modification --- 
        original_user_content = None
        has_original_user_content = False
        if hasattr(ctx, 'user_content'):
            original_user_content = ctx.user_content
            has_original_user_content = True
        
        temp_user_content = None
        if dynamic_instruction:
            temp_user_content = types.Content(parts=[types.Part(text=dynamic_instruction)])
        else:
            # If no dynamic instruction, we might want to pass empty content 
            # or the original, depending on expected agent behavior.
            # Passing None might be safest if agent expects optional content.
            # For now, let's ensure user_content exists if we modify it.
            # If dynamic_instruction is None, we won't modify user_content.
            pass 

        try:
            # Temporarily set user_content for this specific run
            if dynamic_instruction:
                logger.debug(f"[{self.name}] Temporarily setting ctx.user_content for {agent.name}")
                ctx.user_content = temp_user_content
            
            # Execute agent using the modified context
            # Pass ctx as the first positional argument
            async for event in agent.run_async(ctx): 
                if hasattr(event, 'role') and event.role == 'model':
                    is_empty_model_event = (
                        event.content and
                        not event.content.parts
                    )
                    if is_empty_model_event:
                        logger.warning(f"[{self.name}] Filtering empty model event from {agent.name}")
                        continue
                yield event

        except Exception as e:
            logger.error(f"[{self.name}] Error during sub-agent run ({agent.name}): {e}")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Error calling agent {agent.name}: {str(e)}")]),
            )
        finally:
            # Restore original user_content
            if dynamic_instruction: # Only restore if we actually changed it
                logger.debug(f"[{self.name}] Restoring original ctx.user_content for {agent.name}.")
                if has_original_user_content:
                    ctx.user_content = original_user_content
                elif hasattr(ctx, 'user_content'): # If it didn't exist before, remove it? Or set to None?
                     # Setting to None might be safer if the attribute is expected
                     ctx.user_content = None 
                     # Alternatively, del ctx.user_content if possible/safe

    # --- Core Execution Logic ---
    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the Tree of Thoughts beam search algorithm.

        This is the main orchestration method that implements the complete workflow:

        Phase 1 - Initialization:
        - Create root node from initial problem
        - Generate and validate initial strategy nodes
        - Set up initial beam with most promising strategies

        Phase 2 - Main Search Loop:
        - For each iteration (up to max_depth):
          a. Generate next thoughts from active beam nodes
          b. Research and evaluate generated thoughts
          c. Select top-k thoughts for next beam
          d. Update node statuses (active/pruned)
        - Stop if beam empty or max depth reached

        Phase 3 - Synthesis:
        - Identify best path through tree
        - Generate final answer using path context
        - Handle any errors and provide meaningful output

        Args:
            ctx (InvocationContext): Current invocation context

        Yields:
            Event: Progress updates and final results through events

        Note:
            The method maintains detailed logging throughout execution
            and handles error cases gracefully at each phase.
        """
        logger.info(f"[{self.name}] Starting Tree of Thoughts workflow.")
        
        # --- Phase 1: Initialization ---
        logger.info(f"[{self.name}] Phase 1: Initializing thought tree...")
        
        # Call initialization helper which creates root node and initial paths
        async for event in self._initialize_workflow(ctx):
            yield event
            
        # Check if initialization succeeded by looking at generated nodes
        newly_added_ids = self._get_state_value(ctx, '_initialize_workflow_result', [])
        if not newly_added_ids:
            logger.error(f"[{self.name}] Initialization failed - no initial nodes created.")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text="Failed to initialize thought tree.")])
            )
            return

        # Explicitly mark the initial nodes as active for the first generation step
        thought_tree = self._get_thought_tree(ctx)
        for node_id in newly_added_ids:
            if node_id in thought_tree:
                 # Use the helper method to update the node safely
                 self._update_node(ctx, node_id, {"status": "active"})
                 logger.info(f"[{self.name}] Marked initial node {node_id} as active.")
            else:
                 logger.warning(f"[{self.name}] Node {node_id} from initialization result not found in tree.")

        # Initial beam starts with all generated initial strategies
        self._set_active_beam(ctx, newly_added_ids)
        logger.info(f"[{self.name}] Initial active paths set with {len(newly_added_ids)} nodes.")

        # --- Phase 2: Main Beam Search Loop --- # Renamed to Exploration Loop
        # max_iterations = self.max_depth # REMOVED
        current_depth = 1 # Root is at depth 0, initial nodes at depth 1
        iteration_count = 0

        # Loop indefinitely until the active_beam is empty
        while True: # Loop now relies on beam emptying
            iteration_count += 1
            thought_tree = self._get_thought_tree(ctx)
            active_beam = self._get_active_beam(ctx)

            # Check termination condition
            if not active_beam:
                logger.info(f"[{self.name}] All paths terminated or pruned. Stopping exploration loop.")
                break

            # Removed max_depth check
            # if current_depth >= self.max_depth:
            #    logger.info(f"[{self.name}] Reached maximum depth {self.max_depth}, terminating search.")
            #    break

            logger.info(f"[{self.name}] Exploration Iteration {iteration_count}, Current Depth {current_depth}, Active Paths {len(active_beam)}")

            # Step 2a: Generate next thoughts from active beam
            logger.info(f"[{self.name}] Step 2a: Generating next thoughts...")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Generating thoughts (iteration {iteration_count})...")])
            )
            
            async for event in self._generate_next_thoughts(ctx):
                yield event
                
            generated_ids = self._get_state_value(ctx, '_generate_next_thoughts_result', [])
            if not generated_ids:
                logger.warning(f"[{self.name}] No new thoughts generated, terminating search.")
                break
                
            # Step 2b: Evaluate generated thoughts
            logger.info(f"[{self.name}] Step 2b: Evaluating {len(generated_ids)} thoughts...")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Evaluating {len(generated_ids)} new thoughts...")])
            )
            
            async for event in self._evaluate_thoughts(ctx):
                yield event
                
            # Step 2c: Select next beam (best nodes) -> Renamed to Select Active Paths
            logger.info(f"[{self.name}] Step 2c: Selecting active paths for next iteration...")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text="Selecting viable paths for next iteration...")])
            )
            
            new_beam = await self._select_next_beam(ctx) # Function now selects based on termination flag
            self._set_active_beam(ctx, new_beam)
            if new_beam:
                logger.info(f"[{self.name}] New active paths selected: {len(new_beam)} nodes.")
            else:
                logger.info(f"[{self.name}] Selection resulted in no active paths left.")
                # The loop will terminate in the next iteration check

            # Increment depth for next iteration
            # Note: Depth tracking might be less critical now but kept for logging/potential future use
            current_depth += 1

        # --- Phase 3: Synthesis ---
        logger.info(f"[{self.name}] Phase 3: Synthesizing final result...")
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text="Synthesizing final result from best path...")])
        )
        
        async for event in self._synthesize_result(ctx):
            yield event
            
        # Check the result stored in state by _synthesize_result
        synthesizer_result = self._get_state_value(ctx, '_synthesize_result_result', {})

        # Log completion, the final result event should have already been yielded by _synthesize_result
        if "error" in synthesizer_result:
             logger.error(f"[{self.name}] Synthesis failed: {synthesizer_result['error']}")
        logger.info(f"[{self.name}] Tree of Thoughts workflow complete.")

    # --- Initialization Step (Modified Planner Interaction) ---
    async def _initialize_workflow(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Initialize the thought tree and generate initial strategies.

        This method performs the critical setup phase:
        1. Create and validate root node from initial problem
        2. Generate distinct high-level strategies using Planner agent
        3. Validate and add strategy nodes as children of root
        4. Handle existing tree state and error cases

        Implementation Details:
        - Uses validator tool to ensure node data integrity
        - Implements robust parsing of Planner output with fallbacks
        - Maintains detailed logging of the initialization process
        - Stores results in session state for subsequent phases

        Args:
            ctx (InvocationContext): Current invocation context

        Yields:
            Event: Validation and initialization progress events

        State Updates:
            - '_initialize_workflow_result': List of generated node IDs
            - 'initial_problem': The root problem being solved
            - Updates thought tree with root and strategy nodes
        """
        logger.info(f"[{self.name}] Initializing workflow.")
        self._set_state_value(ctx, '_initialize_workflow_result', [])
        thought_tree = self._get_thought_tree(ctx)
        if thought_tree:
             logger.warning(f"[{self.name}] Tree already exists - skipping initialization.")
             # Check for appropriate existing nodes to determine next phase
             nodes_to_evaluate = [nid for nid, data in thought_tree.items() if data.get('status') == 'generated']
             if nodes_to_evaluate:
                 return # Let evaluate phase handle it
             active_beam = self._get_active_beam(ctx)
             if active_beam:
                 return # Let generate phase handle it
             logger.error(f"[{self.name}] Inconsistent state detected.")
             return

        root_id = "root"
        initial_problem = self._get_state_value(ctx, "initial_problem")
        if not initial_problem and ctx.user_content:
            initial_problem = ctx.user_content.parts[0].text if ctx.user_content.parts else "Default initial problem"
        elif not initial_problem:
            initial_problem = "Solve the initial problem."

        logger.info(f"[{self.name}] Root problem: '{initial_problem}'")
        self._set_state_value(ctx, "initial_problem", initial_problem)

        validation_args = { # Validate root node
            "parentId": None, "thoughtId": root_id, "thought": initial_problem,
            "depth": 0, "status": "active",
        }
        validation_result = await self.validator.run_async(tool_context=ctx, args=validation_args)
        # Yield event *after* checking success/failure
        if validation_result.get("validation_status") == "success":
            self._update_node(ctx, root_id, validation_result) # Update tree first
            logger.info(f"[{self.name}] Root node added to tree.")
            # Yield improved validation event with thought content
            yield Event(
                author=self.validator.name, invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Validated Root Node ({root_id}): '{validation_result.get('thoughtContent', 'N/A')[:100]}...'")])
            )
        else:
            logger.error(f"[{self.name}] Root node validation failed: {validation_result.get('error')}. ")
            # Yield clearer failure event
            yield Event(
                author=self.validator.name, invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Validation Failed for Root Node ({root_id}): {validation_result.get('error', 'Unknown error')}")])
            )
            return # Stop initialization if root fails validation

        # 4. Call Planner to generate distinct initial STRATEGIES, not just a definition
        logger.info(f"[{self.name}] Calling Planner to generate initial *strategies*.")
        try:
            # --- STRONGLY REVISED PLANNER INSTRUCTION for Initialization ---
            # Ask for a reasonable fixed number of initial strategies, e.g., 3
            num_initial_strategies = 3 # Define a fixed number
            planner_instruction = (
                f"Your **sole task** right now is to generate exactly **{num_initial_strategies} distinct high-level strategies ('thoughts')** "
                f"to approach the problem: '{initial_problem}'. Remember, a 'thought' represents an intermediate step or a partial solution. **These {num_initial_strategies} thoughts should explore different directions.**\\n"
                f"**CRITICAL FORMATTING REQUIREMENT:**\\n"
                f"1. Each strategy/thought MUST be a concise phrase or sentence describing an approach.\\n"
                f"2. You MUST output *only* the {num_initial_strategies} thoughts, each on a new line.\\n"
                f"3. **DO NOT** include any introductory text, explanations, numbering (like 'Strategy N:'), or any other text before, between, or after the thought list. "
                f"Your entire output should consist only of the {num_initial_strategies} lines, each representing a distinct initial thought/strategy.\\n"
                f"**Example Output for 'Analyze climate change impact':**\\n"
                f"Focus on economic impacts.\\n"
                f"Analyze effects on biodiversity.\\n"
                f"Investigate sea-level rise projections."
            )
            ctx.session.state["planner_instruction"] = planner_instruction
            # --- END OF REVISED INSTRUCTION ---

            planner_output = ""
            logger.info(f"[{self.name}] Sending request to Planner for initial strategies...")
            async for event in self._run_sub_agent_safely(self.planner, ctx, dynamic_instruction=planner_instruction):
                yield event
                if event.content and event.content.parts:
                    planner_output = event.content.parts[0].text
            logger.info(f"[{self.name}] Received planner output for initial strategies:\n{planner_output}")

            # --- Use Regex Parsing First ---
            initial_strategies = []
            # Regex to find "Strategy N: Strategy Text" (case-insensitive start)
            strategy_pattern = re.compile(r"^\s*Strategy\s*\d+:\s*(.*)", re.MULTILINE | re.IGNORECASE)
            matches = strategy_pattern.findall(planner_output)

            if matches:
                initial_strategies = [match.strip() for match in matches]
                logger.info(f"[{self.name}] Successfully extracted {len(initial_strategies)} strategies using format 'Strategy N: ...'.")
            else:
                # If specific format fails, log a warning and try a simpler split as fallback
                logger.warning(f"[{self.name}] Planner did not follow the 'Strategy N:' format. Attempting fallback newline split.")
                # Fallback: Split by newline, filter out empty lines and potentially markdown bullets
                potential_strategies = [p.strip() for p in planner_output.split('\n') if p.strip()]
                # Basic filtering: assume strategies are reasonably long and not just bullets
                initial_strategies = [s for s in potential_strategies if len(s.split()) > 2 and not s.startswith(('*', '-'))]
                if initial_strategies:
                     logger.info(f"[{self.name}] Extracted {len(initial_strategies)} potential strategies via fallback newline split.")
                else:
                     logger.warning(f"[{self.name}] Fallback newline split also failed to find viable strategies.")


            if not initial_strategies:
                logger.error(f"[{self.name}] Planner failed to generate initial strategies in any recognizable format. Using generic fallback.")
                initial_strategies = [f"Develop a comprehensive answer for: {initial_problem}"]
            elif len(initial_strategies) > num_initial_strategies:
                 logger.warning(f"[{self.name}] Planner generated {len(initial_strategies)} strategies, more than the requested {num_initial_strategies}. Using all generated.")
            elif len(initial_strategies) < num_initial_strategies:
                 logger.warning(f"[{self.name}] Planner generated {len(initial_strategies)} strategies, fewer than the requested {num_initial_strategies}.")

            # Proceed with all strategies identified by the planner
            logger.info(f"[{self.name}] Proceeding with {len(initial_strategies)} initial strategies identified by the planner.")
            # --- End of Parsing Logic ---

            # 5. Process each strategy into a child node
            newly_added_ids = []
            for i, strategy_thought in enumerate(initial_strategies):
                child_id = f"{root_id}-{i}"
                child_validation_args = {
                    "parentId": root_id, "thoughtId": child_id,
                    "thought": strategy_thought, # Use the strategy as the thought content
                    "depth": 1, "status": "generated", # Ready for generation/evaluation
                }
                child_validation_result = await self.validator.run_async(tool_context=ctx, args=child_validation_args)
                # Yield event *after* checking success/failure
                if child_validation_result.get("validation_status") == "success":
                    self._update_node(ctx, child_id, child_validation_result) # Update tree first
                    newly_added_ids.append(child_id)
                    logger.info(f"[{self.name}] Added initial strategy node: '{strategy_thought[:50]}...' ({child_id})")
                    # Yield improved validation event with thought content
                    yield Event(
                        author=self.validator.name, invocation_id=ctx.invocation_id,
                        content=types.Content(parts=[types.Part(text=f"Validated Initial Strategy ({child_id}): '{child_validation_result.get('thoughtContent', 'N/A')[:100]}...'")])
                    )
                else:
                    logger.warning(f"[{self.name}] Validation failed for strategy path {i}: {child_validation_result.get('error')}")
                    # Yield clearer failure event
                    yield Event(
                        author=self.validator.name, invocation_id=ctx.invocation_id,
                        content=types.Content(parts=[types.Part(text=f"Validation Failed for Initial Strategy ({child_id}): {child_validation_result.get('error', 'Unknown error')}")])
                    )

            # 6. Store results
            self._set_state_value(ctx, '_initialize_workflow_result', newly_added_ids)
            logger.info(f"[{self.name}] Initialization complete with {len(newly_added_ids)} initial strategy paths.")

        except Exception as e:
            logger.error(f"[{self.name}] Planner failed during initialization: {str(e)}")
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Error during initialization: {str(e)}")])
            )

    # --- Generation Step ---
    async def _generate_next_thoughts(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Generate next level of thoughts from current active beam nodes.

        This method implements the expansion phase of beam search:
        1. Process each node in active beam
        2. Use dynamic generation count based on node scores and depth
        3. Generate and validate child thoughts for each active node
        4. Store results for evaluation phase

        Key Features:
        - Dynamic generation count adjusts based on:
          - Parent node's evaluation score
          - Current depth in tree
          - Base beam width parameter
        - Maintains node relationships in tree structure
        - Handles validation and error cases for each generated thought

        Args:
            ctx (InvocationContext): Current invocation context

        Yields:
            Event: Generation progress and validation events

        State Updates:
            - '_generate_next_thoughts_result': List of newly generated node IDs
            - Updates thought tree with new child nodes
        """
        active_beam = self._get_active_beam(ctx)
        thought_tree = self._get_thought_tree(ctx)
        logger.info(f"[{self.name}] Starting Generation Step for beam: {active_beam}")
        newly_generated_ids_this_round = []
        self._set_state_value(ctx, '_generate_next_thoughts_result', [])

        for parent_id in active_beam:
            parent_node = thought_tree.get(parent_id)
            if not parent_node or parent_node.get('status') != 'active':
                logger.debug(f"[{self.name}] Skipping generation for non-active node: {parent_id}")
                continue

            parent_thought = parent_node.get("thoughtContent", "")
            parent_depth = parent_node.get("depth", 0)

            logger.info(f"[{self.name}] Calling Planner to expand node: {parent_id} ('{parent_thought[:50]}...')")
            try:
                # --- Dynamic Generation Count Logic (Removed dependency on beam_width) ---
                base_num_to_generate = 2 # Use a fixed small base, e.g., 2
                parent_score = parent_node.get("evaluationScore")
                score_adjustment = 0
                if parent_score is not None:
                    if parent_score >= 8.0:
                        score_adjustment = 1 # Generate more if promising
                    elif parent_score < 4.0: # More aggressive pruning if score is low
                        score_adjustment = -1 # Generate fewer if not promising

                # Depth adjustment logic - NO self.max_depth here
                depth_adjustment = 0
                # You could optionally add depth logic based on absolute depth, e.g.:
                # if parent_depth >= 5: # Check against a fixed number, not self.max_depth
                #    depth_adjustment = -1

                num_to_generate = max(1, base_num_to_generate + score_adjustment + depth_adjustment)
                logger.info(f"[{self.name}] Dynamically determined to generate {num_to_generate} thoughts for node {parent_id} (base={base_num_to_generate}, score_adj={score_adjustment}, depth_adj={depth_adjustment})")

                # --- Instruction for expansion ---
                planner_instruction = (
                    f"The current thought/strategy is: '{parent_thought}'. "
                    f"It is at depth {parent_depth} and received a score of {parent_score if parent_score is not None else 'N/A'}. "
                    f"Based on this, generate exactly **{num_to_generate}** distinct and concrete **next thoughts** "
                    f"(intermediate steps, partial solutions, sub-topics, or specific questions) to explore *within* this thought/strategy. "
                    f"Focus on quality and relevance. Remember, a 'thought' represents an intermediate step or a partial solution. **Ensure these {num_to_generate} thoughts represent different directions for exploration.**\\n"
                    f"**CRITICAL FORMATTING REQUIREMENTS:**\\n"
                    f"1. List each clearly on a new line.\\n"
                    f"2. Output *only* the {num_to_generate} thoughts, one per line.\\n"
                    f"3. **DO NOT** include any introductory text, explanations, numbering, or any other text before, between, or after the list. Your entire output must be only the {num_to_generate} lines of next thoughts.\\n"
                    f"**Example Input Thought:** 'Analyze effects on biodiversity'\\n"
                    f"**Example Output Next Thoughts (if num_to_generate=2):**\\n"
                    f"Identify key species impacted by rising temperatures.\\n"
                    f"Research habitat loss in coastal regions."
                )
                ctx.session.state["planner_instruction"] = planner_instruction
                
                planner_output = ""
                async for event in self._run_sub_agent_safely(self.planner, ctx, dynamic_instruction=planner_instruction):
                    yield event
                    if event.content and event.content.parts:
                        planner_output = event.content.parts[0].text

                child_thoughts_text = planner_output
                child_thoughts = [p.strip() for p in child_thoughts_text.split('\n') if p.strip()]
                
                # Enforce generation count
                if len(child_thoughts) > num_to_generate:
                    logger.warning(f"[{self.name}] Planner returned {len(child_thoughts)} thoughts, exceeding request for {num_to_generate}. Truncating.")
                    child_thoughts = child_thoughts[:num_to_generate]
                elif len(child_thoughts) < num_to_generate:
                     logger.warning(f"[{self.name}] Planner returned {len(child_thoughts)} thoughts, less than requested {num_to_generate}.")

                if not child_thoughts:
                     logger.warning(f"[{self.name}] Planner returned no distinct thoughts for node {parent_id}. Output: {child_thoughts_text}")
                     continue

                # Skip empty thoughts before validation
                for i, child_thought in enumerate(child_thoughts):
                    if not child_thought:
                        logger.warning(f"[{self.name}] Skipping empty thought generated by Planner for parent {parent_id} at index {i}.")
                        continue

                for i, child_thought in enumerate(child_thoughts):
                    child_id = f"{parent_id}-gen{i}"
                    child_validation_args = {
                        "parentId": parent_id,
                        "thoughtId": child_id,
                        "thought": child_thought,
                        "depth": parent_depth + 1,
                        "status": "generated",
                    }
                    child_validation_result = await self.validator.run_async(tool_context=ctx, args=child_validation_args)
                    # Yield event *after* checking success/failure
                    if child_validation_result.get("validation_status") == "success":
                        self._update_node(ctx, child_id, child_validation_result) # Update tree first
                        newly_generated_ids_this_round.append(child_id)
                        logger.info(f"[{self.name}] Generated child node: {child_id}")
                        # Yield improved validation event with thought content
                        yield Event(
                            author=self.validator.name,
                            invocation_id=ctx.invocation_id,
                            content=types.Content(
                                parts=[types.Part(text=f"Validated Thought ({child_id}): '{child_validation_result.get('thoughtContent', 'N/A')[:100]}...'")]
                            )
                        )
                    else:
                        logger.warning(f"[{self.name}] Validation failed for generated child of {parent_id}: {child_validation_result.get('error')}")
                        # Yield clearer failure event
                        yield Event(
                            author=self.validator.name,
                            invocation_id=ctx.invocation_id,
                            content=types.Content(
                                parts=[types.Part(text=f"Validation Failed for Generated Thought ({child_id}): {child_validation_result.get('error', 'Unknown error')}")]
                            )
                        )

            except Exception as e:
                logger.error(f"[{self.name}] Planner failed for node {parent_id}: {e}")
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(parts=[types.Part(text=f"Planner failed for node {parent_id}: {e}")])
                )
                continue # Skip this parent if planner fails

        # Store result in state before returning
        self._set_state_value(ctx, '_generate_next_thoughts_result', newly_generated_ids_this_round)
        return # Return None implicitly

    # --- Evaluation Step (Modified) ---
    async def _evaluate_thoughts(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Evaluate generated thoughts using research, analysis, and critique.

        This method implements a comprehensive evaluation process:
        1. Research Phase:
           - Gather relevant information for each thought
           - Validate and contextualize findings
        2. Analysis Phase:
           - Evaluate soundness and feasibility
           - Generate numerical scores (0-10)
        3. Critique Phase:
           - Identify potential issues and improvements
           - Provide additional scoring perspective
        4. Integration:
           - Combine scores from different evaluators
           - Update node status and store evaluation data

        Key Features:
        - Multi-perspective evaluation using specialist agents
        - Research-backed analysis for informed decisions
        - Robust score extraction and combination
        - Detailed logging of evaluation process
        - Error handling for each evaluation step

        Args:
            ctx (InvocationContext): Current invocation context

        Yields:
            Event: Research findings and evaluation progress events

        State Updates:
            - '_evaluate_thoughts_result': List of evaluated node data
            - Updates thought tree with evaluation results
            - Marks nodes as 'evaluated'
        """
        thought_tree = self._get_thought_tree(ctx)
        nodes_to_evaluate = [
            (node_id, data) for node_id, data in thought_tree.items()
            if data.get('status') == 'generated'
        ]
        logger.info(f"[{self.name}] Starting Evaluation Step for {len(nodes_to_evaluate)} nodes.")
        evaluated_nodes_data = []

        self._set_state_value(ctx, '_evaluate_thoughts_result', [])

        if not nodes_to_evaluate:
            logger.info(f"[{self.name}] No nodes found with status 'generated' to evaluate.")
            return

        for node_id, node_data in nodes_to_evaluate:
            node_thought = node_data.get("thoughtContent", "")
            logger.info(f"[{self.name}] Evaluating node {node_id}: '{node_thought[:50]}...'")

            # --- Step 2a: Call Researcher ---
            research_findings = "No research conducted."
            try:
                research_instruction = (
                    f"Gather relevant information and context for the following thought using your search tool. "
                    f"Focus on facts, potential issues, or supporting data.\nThought: {node_thought}"
                )
                # Store instruction in session state for researcher agent
                ctx.session.state["researcher_instruction"] = research_instruction

                logger.info(f"[{self.name}] Calling Researcher for node {node_id}...")
                current_research_output = ""
                final_research_text = None
                async for event in self._run_sub_agent_safely(self.researcher, ctx, dynamic_instruction=research_instruction):
                    yield event
                    if event.content and event.content.parts and event.content.parts[0].text:
                        if final_research_text is None:
                             final_research_text = event.content.parts[0].text

                research_findings = final_research_text if final_research_text is not None else "Researcher returned no specific findings."
                logger.info(f"[{self.name}] Research completed for node {node_id}.")
                yield Event(
                    author=self.researcher.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(parts=[types.Part(text=f"Research findings for '{node_thought[:30]}...':\n{research_findings}")])
                )

            except Exception as e:
                logger.error(f"[{self.name}] Researcher failed for node {node_id}: {e}")
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(parts=[types.Part(text=f"Research error for node {node_id}: {e}")])
                )
                research_findings = f"Research step failed for node {node_id}. Proceeding without external research."


            # --- Step 2b: Call Analyzer and Critic (with research context) ---
            analyzer_score = None
            critic_score = None
            analyzer_output = ""
            critic_output = ""

            try:
                # Analyzer instruction includes research findings and termination check
                analyzer_instruction = (
                    f"Analyze the soundness, feasibility, and logical consistency of the following thought, "
                    f"considering the research findings below. Focus on its potential to lead towards a final solution.\n\n"
                    f"Thought: {node_thought}\n\n"
                    f"Research Findings:\n{research_findings}\n\n"
                    f"**Evaluation Tasks:**\n"
                    f"1. Provide your analysis focusing on the thought's promise and viability for continued exploration.\n"
                    f"2. Assess if this path seems highly promising, has hit a dead end, or might be close to a solution. Based on this, recommend whether to continue exploring this path.\n"
                    f"3. Output a structured score: `Evaluation Score: [0-10]/10` (0=dead end, 10=highly promising path).\n"
                    f"4. Output a termination recommendation: `Termination Recommendation: [True/False]` (True if path should stop, False if it should continue)."
                )
                ctx.session.state["analyzer_instruction"] = analyzer_instruction
                logger.info(f"[{self.name}] Calling Analyzer for node {node_id} with research context...")
                current_analyzer_output = ""
                final_analyzer_text = None
                async for event in self._run_sub_agent_safely(self.analyzer, ctx, dynamic_instruction=analyzer_instruction):
                    yield event
                    if event.content and event.content.parts and event.content.parts[0].text:
                         if final_analyzer_text is None:
                              final_analyzer_text = event.content.parts[0].text
                
                analyzer_output = final_analyzer_text if final_analyzer_text is not None else "[Analyzer returned empty output]"
                analyzer_score = self._extract_score(analyzer_output)
                logger.info(f"[{self.name}] Analyzer score for {node_id}: {analyzer_score}")
            except Exception as e:
                logger.error(f"[{self.name}] Analyzer failed for node {node_id}: {e}")
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(parts=[types.Part(text=f"Analysis error for node {node_id}: {e}")])
                )

            try:
                # Critic instruction includes research findings, alternatives, and termination check
                critic_instruction = (
                    f"Critically evaluate the following thought for flaws, biases, or weaknesses, considering the research findings below. "
                    f"Focus on whether this path is worth pursuing further.\n\n"
                    f"Thought: {node_thought}\n\n"
                    f"Research Findings:\n{research_findings}\n\n"
                    f"**Evaluation Tasks:**\n"
                    f"1. Provide your critique, identifying weaknesses that might hinder progress down this path.\n"
                    f"2. Suggest specific improvements OR concrete alternative *directions* if this path seems flawed.\n"
                    f"3. Assess the overall promise and viability of *continuing* down this path. Recommend whether to stop or proceed.\n"
                    f"4. Output a structured score: `Evaluation Score: [0-10]/10` (0=stop, 10=very promising to continue).\n"
                    f"5. Output a termination recommendation: `Termination Recommendation: [True/False]` (True if path should stop, False if it should continue)."

                )
                ctx.session.state["critic_instruction"] = critic_instruction
                logger.info(f"[{self.name}] Calling Critic for node {node_id} with research context...")
                current_critic_output = ""
                final_critic_text = None
                async for event in self._run_sub_agent_safely(self.critic, ctx, dynamic_instruction=critic_instruction):
                    yield event
                    if event.content and event.content.parts and event.content.parts[0].text:
                         if final_critic_text is None:
                              final_critic_text = event.content.parts[0].text
                
                critic_output = final_critic_text if final_critic_text is not None else "[Critic returned empty output]"
                critic_score = self._extract_score(critic_output)
                logger.info(f"[{self.name}] Critic score for {node_id}: {critic_score}")
            except Exception as e:
                logger.error(f"[{self.name}] Critic failed for node {node_id}: {e}")
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(parts=[types.Part(text=f"Critique error for node {node_id}: {e}")])
                )

            # --- Step 2c: Combine scores and update node ---
            scores = [s for s in [analyzer_score, critic_score] if s is not None]
            final_score = sum(scores) / len(scores) if scores else 1.0
            logger.info(f"[{self.name}] Node {node_id} final score: {final_score:.2f}")

            # --- Extract Termination Recommendation ---
            # Check both outputs, favor 'True' if either recommends termination
            analyzer_term_rec = self._extract_termination_recommendation(analyzer_output)
            critic_term_rec = self._extract_termination_recommendation(critic_output)
            final_termination_rec = analyzer_term_rec or critic_term_rec # Default to False if neither found
            logger.info(f"[{self.name}] Node {node_id} Termination Recommendation: {final_termination_rec} (Analyzer: {analyzer_term_rec}, Critic: {critic_term_rec})")
            # --- End Termination Extraction ---

            update_data = {
                "evaluationScore": final_score,
                "status": "evaluated", # Status before selection decides active/pruned/terminated
                "researchFindings": research_findings,
                "analyzerOutput": analyzer_output,
                "criticOutput": critic_output,
                "terminationRecommended": final_termination_rec, # Store the recommendation
            }
            self._update_node(ctx, node_id, update_data)
            evaluated_nodes_data.append({**node_data, **update_data}) # Add combined data

        logger.info(f"[{self.name}] Evaluation completed for {len(evaluated_nodes_data)} nodes.")
        self._set_state_value(ctx, '_evaluate_thoughts_result', evaluated_nodes_data)

    # --- Selection Step ---
    async def _select_next_beam(self, ctx: InvocationContext) -> List[str]:
        """
        Select the top-k thoughts for the next beam based on evaluation scores.

        This method implements the beam search selection process:
        1. Identify all evaluated nodes from current round
        2. Sort nodes by evaluation score (descending)
        3. Select top-k nodes based on beam_width
        4. Update node statuses (active/pruned)

        Selection Process:
        - Considers only nodes marked as 'evaluated'
        - Uses evaluation scores to rank nodes
        - Maintains beam width constraint
        - Updates node statuses for tracking

        Args:
            ctx (InvocationContext): Current invocation context

        Returns:
            List[str]: IDs of selected nodes for next beam

        State Updates:
            - Updates node statuses in thought tree:
              - Selected nodes -> 'active' # Nodes not recommended for termination
              - Non-selected nodes -> 'pruned' # Nodes not selected (implicitly includes low-score ones)
              - Nodes recommended for termination -> 'terminated_early'
            - Logs selection decisions and scores
        """
        thought_tree = self._get_thought_tree(ctx)
        
        # Find all evaluated nodes from this round
        nodes_to_consider = [
            data for data in thought_tree.values()
            if data.get('status') == 'evaluated'
        ]

        logger.info(f"[{self.name}] Selection: found {len(nodes_to_consider)} evaluated nodes to consider.")

        if not nodes_to_consider:
            logger.warning(f"[{self.name}] No evaluated nodes found for selection.")
            return []

        # Sort nodes by score (descending) - still useful for logging/understanding
        nodes_to_consider.sort(key=lambda x: x.get("evaluationScore", 0.0), reverse=True)

        # Log scores to help understand selection
        # Show more scores now that beam width isn't fixed
        log_limit = 10
        logger.info(f"[{self.name}] Node scores (Top {log_limit}): " +
                   ", ".join([f"{node.get('validatedThoughtId', 'unknown')}:{node.get('evaluationScore', 0.0):.2f} (Term:{node.get('terminationRecommended', 'F')})" 
                             for node in nodes_to_consider[:log_limit]]) +
                   (f" ... and {len(nodes_to_consider)-log_limit} more" if len(nodes_to_consider) > log_limit else ""))

        # Update node statuses - selected nodes are 'active', others are 'pruned' or 'terminated_early'
        selected_count = 0
        pruned_count = 0
        terminated_count = 0
        final_beam = [] # The actual beam: nodes to activate for the next iteration

        for node_data in nodes_to_consider:
            node_id = node_data["validatedThoughtId"]
            termination_recommended = node_data.get("terminationRecommended", False)

            # Decide based on termination recommendation
            if termination_recommended:
                self._update_node(ctx, node_id, {"status": "terminated_early"})
                terminated_count += 1
                # logger.info(f"[{self.name}] Node {node_id} was recommended for termination. Status: terminated_early.") # Reduced verbosity
            else:
                # If not terminated, it becomes active for the next round
                self._update_node(ctx, node_id, {"status": "active"})
                final_beam.append(node_id) # Add to the actual next beam
                selected_count += 1

            # Note: We no longer explicitly prune nodes *not* in a top-k list.
            # Nodes that are evaluated but not recommended for termination become active.
            # Nodes evaluated and recommended for termination become terminated_early.
            # There isn't a separate 'pruned' status assigned here unless we add logic
            # e.g., prune nodes below a certain absolute score threshold?
            # For now, keeping it simple: evaluated nodes are either terminated or active.

        # Adjust log message - pruned_count will be 0 with current logic
        logger.info(f"[{self.name}] Selection complete - {selected_count} nodes marked active for next iteration, {terminated_count} nodes terminated early.")
        logger.info(f"[{self.name}] Final active paths for next iteration: {final_beam}")
        return final_beam

    # --- Synthesis Step ---
    async def _synthesize_result(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Synthesize final result from the best path in the thought tree.
        Synthesize final result based on high-scoring thoughts identified during exploration.

        This method implements the final synthesis process:
        1. Best Path Selection:
           - Find highest scoring node in active beam
           - Fall back to best scored node in entire tree if needed
           - Use root as last resort if no scored nodes exist
        2. Path Construction:
           - Trace path from best node back to root
           - Build comprehensive path representation
           - Include scores and depth information
        1. High-Scoring Node Selection:
           - Filter all relevant nodes (evaluated, active, terminated_early) with scores.
           - Select nodes exceeding a score threshold (e.g., 7.0).
           - If none meet threshold, select top N highest-scoring nodes.
           - Fallback to root if no scored nodes found.
        2. Context Construction:
           - Gather information from selected high-scoring nodes.
           - Include the initial problem statement.
        3. Final Synthesis:
           - Call synthesizer agent with path context
           - Call synthesizer agent with the context of high-scoring thoughts.
           - Generate coherent final answer
           - Handle potential synthesis failures

        Key Features:
        - Robust best node selection with fallbacks
        - Comprehensive path tracing and formatting
        - Robust high-scoring node selection with threshold and fallback.
        - Context includes multiple promising perspectives.
        - Error handling at each synthesis step
        - Detailed logging of synthesis process

        Args:
            ctx (InvocationContext): Current invocation context

        Yields:
            Event: Synthesis progress and result events

        State Updates:
            - '_synthesize_result_result': Final synthesis output or error
            - Logs synthesis decisions and results
        """
        thought_tree = self._get_thought_tree(ctx)
        active_beam = self._get_active_beam(ctx) # Keep active_beam for potential logging/context, but not primary selection

        # Initialize result in state with a default error value
        default_error_result = {"error": "Synthesis could not find a suitable result.", "output": ""}
        self._set_state_value(ctx, '_synthesize_result_result', default_error_result)

        # --- Step 1: Select High-Scoring Nodes ---
        score_threshold = 7.0 # Define the score threshold for promising thoughts
        top_n_fallback = 3 # Number of top nodes to use if threshold isn't met

        # Filter relevant nodes with scores
        candidate_nodes = []
        for node_id, node_data in thought_tree.items():
            if node_data.get('status') in ['evaluated', 'active', 'terminated_early'] and node_data.get('evaluationScore') is not None:
                 candidate_nodes.append(node_data)

        if not candidate_nodes:
            logger.warning(f"[{self.name}] No nodes with evaluation scores found. Falling back to root node.")
            selected_nodes = [thought_tree.get("root")] if "root" in thought_tree else []
        else:
             # Sort candidates by score (descending) for thresholding and fallback
             candidate_nodes.sort(key=lambda x: x.get("evaluationScore", 0.0), reverse=True)

             # Apply threshold
             selected_nodes = [node for node in candidate_nodes if node.get("evaluationScore", 0.0) >= score_threshold]

             if not selected_nodes:
                 logger.warning(f"[{self.name}] No nodes met score threshold {score_threshold}. Selecting top {top_n_fallback} nodes.")
                 selected_nodes = candidate_nodes[:top_n_fallback] # Take the top N as fallback

        if not selected_nodes:
            logger.error(f"[{self.name}] Synthesis failed: No suitable nodes found even after fallback.")
            yield Event(
                 author=self.name,
                 invocation_id=ctx.invocation_id,
                 content=types.Content(parts=[types.Part(text="Synthesis failed: Could not identify any promising thoughts to synthesize from.")])
            )
            self._set_state_value(ctx, '_synthesize_result_result', {"error": "No promising thoughts found for synthesis.", "output": ""})
            return

        logger.info(f"[{self.name}] Selected {len(selected_nodes)} high-scoring nodes for synthesis.")

        # --- Step 2: Construct Context for Synthesizer ---
        initial_problem = self._get_state_value(ctx, "initial_problem", "Unknown initial problem")
        synthesis_context_parts = [
            f"Initial Problem: '{initial_problem}'\n",
            "High-Scoring Thoughts Identified:"
        ]

        for node in selected_nodes:
             node_info = (
                 f"- ID: {node.get('validatedThoughtId', 'N/A')}, "
                 f"Score: {node.get('evaluationScore', 'N/A'):.2f}, "
                 f"Thought: {node.get('thoughtContent', 'N/A')}"
             )
             synthesis_context_parts.append(node_info)

        synthesis_context = "\n".join(synthesis_context_parts)
        logger.info(f"[{self.name}] Synthesis context constructed with {len(selected_nodes)} nodes.")
        # Log the context for debugging if needed (can be verbose)
        # logger.debug(f"[{self.name}] Synthesis Context:\n{synthesis_context}")


        # --- Step 3: Call Synthesizer ---
        logger.info(f"[{self.name}] Calling synthesizer with context from high-scoring thoughts.")
        try:
            # Create synthesis instruction based on multiple promising thoughts
            synthesis_instruction = (
                 f"Synthesize the final answer or conclusion for the initial problem based on the following promising thoughts identified during exploration:\n\n"
                 f"{synthesis_context}\n\n"
                 f"Integrate the insights from these thoughts to provide a comprehensive and coherent final result. "
                 f"Address the initial problem directly."
            )

            # Store instruction in session state for synthesizer
            ctx.session.state["synthesizer_instruction"] = synthesis_instruction

            # Correct: Use async for to iterate over agent events
            synthesizer_output = ""
            final_synthesizer_text = None # To store the final output text
            async for event in self._run_sub_agent_safely(self.synthesizer, ctx, dynamic_instruction=synthesis_instruction):
                yield event
                # Capture the last non-empty model output as the final result
                if hasattr(event, 'role') and event.role == 'model' and event.author == self.synthesizer.name and event.content and event.content.parts and event.content.parts[0].text:
                     final_synthesizer_text = event.content.parts[0].text

            # Use the captured final text
            synthesizer_output = final_synthesizer_text if final_synthesizer_text is not None else "[Synthesizer returned empty output]"

            # Store result
            synthesizer_result = {"output": synthesizer_output}
            logger.info(f"[{self.name}] Synthesizer completed successfully.")
            self._set_state_value(ctx, '_synthesize_result_result', synthesizer_result)

        except Exception as e:
            logger.error(f"[{self.name}] Synthesizer failed: {str(e)}")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Synthesis failed: {str(e)}")])
            )

            # Update error result with more details
            error_result = {
                "error": f"Synthesizer failed: {str(e)}",
                "output": "Failed to generate final result due to an error."
            }
            self._set_state_value(ctx, '_synthesize_result_result', error_result)


# --- Tool and Agent Instantiation ---

# 1. Create a validator tool for thought node validation
validator_tool = FunctionTool(validate_thought_node_data)

# 2. Create our custom ToT Beam Search Coordinator agent
# This demonstrates the pattern from the StoryFlowAgent example
root_agent = ToTCoordinator(
    name="ToT_Coordinator",
    planner=planner_agent,
    researcher=researcher_agent,
    analyzer=analyzer_agent,
    critic=critic_agent,
    synthesizer=synthesizer_agent,
    validator=validator_tool,
    model=coordinator_config,
)

# Log the final agent configuration
logger.info(f"ToT Coordinator initialized for LLM-driven exploration.")

# Entry point for running the agent would typically be here,
# creating a Runner instance and calling run() or run_async().