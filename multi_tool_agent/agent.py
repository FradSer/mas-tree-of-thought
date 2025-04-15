import os  # Added for environment variables
import logging # Import logging
import re # Import regex for score extraction
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple # Add Tuple

from google.genai import types # Import for Event content

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent # Import LlmAgent explicitly for type hint
from google.adk.tools import google_search, FunctionTool # Add FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.models.lite_llm import LiteLlm  # Add LiteLlm for OpenRouter access
from google.adk.events import Event # Add Event
from typing_extensions import override # Add override
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Thought Data Model (Used for Tool Argument Validation and Typing) ---

class ThoughtData(BaseModel):
    """
    Represents the data structure for a single thought (node) in the Tree of Thoughts.
    Used to validate arguments for the 'register_thought_node' tool.
    Ensures consistency and structure in the thought metadata managed by the Coordinator.
    """
    parentId: Optional[str] = Field( # Use string ID for flexibility
        None,
        description="The ID of the parent thought node in the tree. None for the root."
    )
    thoughtId: str = Field( # Add explicit ID
        ...,
        description="A unique identifier for this thought node (e.g., 'node-0', 'node-1a', 'node-1b')."
    )
    thought: str = Field(
        ...,
        description="The core content of the current thought/step (e.g., 'Hypothesize strategy A', 'Evaluate feasibility of strategy A', 'Generate sub-steps for strategy A').",
        min_length=1
    )
    # Removed thoughtNumber, totalThoughts, nextThoughtNeeded - managed by search strategy
    # Removed isRevision, revisesThought, branchFromThought, branchId - handled by tree structure (parentId)
    evaluationScore: Optional[float] = Field(
        None,
        description="A score assigned by the evaluation process, indicating the promise of this thought path (higher is better)."
    )
    status: str = Field(
        "active", # Default status
        description="The current status of this thought node (e.g., 'active', 'evaluated', 'promising', 'pruned', 'solution')."
    )
    depth: int = Field( # Track depth for search strategies
        ...,
        description="The depth of this node in the tree (root is 0).",
        ge=0
    )

    # Pydantic model configuration
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # --- Validators ---
    # Simplified validators - complex relations handled by Coordinator logic
    @field_validator('parentId')
    @classmethod
    def validate_parent_id_format(cls, v: Optional[str]) -> Optional[str]:
        # Basic format check - could be more complex if needed
        if v is not None and not isinstance(v, str):
            raise ValueError('parentId must be a string if set')
        return v

    def dict(self) -> Dict[str, Any]:
        """Convert thought data to dictionary format for serialization"""
        return self.model_dump(exclude_none=True)

# --- Sequential Thinking Tool renamed and repurposed ---

def validate_thought_node_data(
    parentId: Optional[str],
    thoughtId: str,
    thought: str,
    depth: int,
    evaluationScore: Optional[float] = None,
    status: str = "active"
) -> Dict[str, Any]:
    """
    Validates metadata for a single node in the Tree of Thoughts.

    This tool acts as a validation layer for thought node metadata before the
    Coordinator adds it to the tree state. It does not perform the thinking
    or manage the tree structure itself; state is managed by the calling Coordinator.
    It ensures the basic structure and required fields are present.

    Args:
        parentId: The ID of the parent thought node. None for the root node only.
        thoughtId: A unique ID for this new thought node.
        thought: The core content/idea of this thought node.
        depth: The depth of this node in the tree (root=0).
        evaluationScore: Optional score from evaluation (higher is better).
        status: The initial status of the node (default: 'active').

    Returns:
        A dictionary containing the validated thought node metadata if successful,
        or an error dictionary if validation fails. Contains a 'validation_status' key.
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
            "message": f"Thought node '{current_thought_data.thoughtId}' data is valid."
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
    str | LiteLlm, # Root / Planner / Critic
    str,          # Researcher (Always Google)
    str | LiteLlm  # Analyzer / Synthesizer
]:
    """Configures LLM models for agent groups based on environment variables."""
    # --- Model Names --- 
    default_google_model = "gemini-2.0-flash"
    google_model_for_researcher = os.environ.get("LLM_MODEL", default_google_model)
    openrouter_high_capability_model = "openrouter/google/gemini-2.5-pro-preview-03-25" # For Root, Planner, Critic
    openrouter_standard_capability_model = "openrouter/google/gemini-2.0-flash-001" # For Analyzer, Synthesizer
    
    # --- Environment Flags --- 
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    # --- Configuration Variables --- 
    # Type hints accommodate fallback to string (Google model name)
    root_planner_critic_config: str | LiteLlm
    researcher_config: str 
    analyzer_synth_config: str | LiteLlm

    # --- Configure Researcher (Always Google Standard) --- 
    researcher_config = google_model_for_researcher
    logger.info(f"Researcher agent configured to use Google model: {researcher_config}")
    # Log Google credential warnings for the researcher model
    if use_vertex:
         if not os.environ.get("GOOGLE_CLOUD_PROJECT") or not os.environ.get("GOOGLE_CLOUD_LOCATION"):
              logger.warning(f"Researcher using Vertex AI ({researcher_config}), but GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION env vars not set.")
    else:
         if not os.environ.get("GOOGLE_API_KEY"):
              logger.warning(f"Researcher using Google AI Studio ({researcher_config}), but GOOGLE_API_KEY env var not set.")
    # --- End Researcher Configuration --- 

    if openrouter_key:
        logger.info("OPENROUTER_API_KEY found. Configuring other agents with specific OpenRouter models.")
        try:
            # Configure Group 1: Root, Planner, Critic
            root_planner_critic_config = LiteLlm(model=openrouter_high_capability_model)
            logger.info(f"Root, Planner, Critic agents configured for OpenRouter model: {openrouter_high_capability_model}")

            # Configure Group 3: Analyzer, Synthesizer
            analyzer_synth_config = LiteLlm(model=openrouter_standard_capability_model)
            logger.info(f"Analyzer, Synthesizer agents configured for OpenRouter model: {openrouter_standard_capability_model}")

        except Exception as e:
            logger.error(f"Failed to configure LiteLlm for OpenRouter (High: {openrouter_high_capability_model}, Standard: {openrouter_standard_capability_model}): {e}. Falling back to Google models for these agents.")
            # Fallback: Assign Google model used by Researcher to other groups
            root_planner_critic_config = researcher_config 
            analyzer_synth_config = researcher_config 
            logger.info(f"Fallback: Root, Planner, Critic, Analyzer, Synthesizer will use Google model: {researcher_config}")
            # Google cred warnings already logged for researcher_config

    else:
        # No OpenRouter key: All agents use the Google model defined for Researcher
        logger.info("OPENROUTER_API_KEY not found. All agents will use the same Google model as the Researcher.")
        root_planner_critic_config = researcher_config
        analyzer_synth_config = researcher_config
        # Google cred warnings already logged for researcher_config

    # Return the specific configurations for each group
    return root_planner_critic_config, researcher_config, analyzer_synth_config 

# Get model configurations
root_planner_critic_config, researcher_config, analyzer_synth_config = _configure_llm_models()

# --- Specialist Agent Definitions ---
# Note: For simplicity, these agents currently have no specific tools assigned.
# They rely solely on their instructions and the LLM's capabilities.
# You can add FunctionTool, APIHubToolset, etc., to them as needed.

# Helper to create standard instructions
def _create_agent_instruction(specialist_name: str, core_task: str, extra_guidance: List[str] = []) -> str:
    """Creates a standardized instruction string for specialist agents."""
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
    model=root_planner_critic_config, # Use configured Google model
    description="Develops strategic plans and roadmaps based on delegated sub-tasks. Identifies alternative options.",
    instruction=_create_agent_instruction(
        specialist_name="Strategic Planner",
        core_task=(
            " 1. Read the planning instruction from session state with key 'planner_instruction'.\\n"
            " 2. If no instruction is found, develop a general plan or roadmap.\\n"
            " 3. **Crucially: If multiple viable strategies or paths exist, you MUST explicitly list them as distinct options** for the Coordinator to consider for branching.\\n"
            " 4. Identify potential roadblocks or critical decision points within your plan.\\n"
            " 5. Return ONLY the planning output, clearly indicating any distinct options found."
        )
    ),
    input_schema=None,
    tools=[]
)

# Researcher Agent
researcher_agent = Agent(
    name="Researcher",
    model=researcher_config, # Use configured Google model
    description="Gathers and validates information, highlighting conflicts and uncertainties.",
    instruction=_create_agent_instruction(
        specialist_name="Information Gatherer",
        core_task=(
            " 1. Identify the specific information needed based on the instruction in session state key 'researcher_instruction'.\\n"
            " 2. **Actively use the `google_search` tool** for external information gathering.\\n"
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
    model=analyzer_synth_config, # Use configured Google model
    description="Performs analysis, identifying inconsistencies and assumptions, and provides a structured evaluation score.",
    instruction=_create_agent_instruction(
        specialist_name="Core Analyst",
        core_task=(
            " 1. Read the analysis instruction from session state with key 'analyzer_instruction'.\\n"
            " 2. If no instruction is found, perform general analysis on the provided content.\\n"
            " 3. **Crucially: Identify any invalidated assumptions, logical inconsistencies, or contradictions**, especially if they impact previous conclusions.\\n"
            " 4. Generate concise insights based on your analysis.\\n"
            " 5. **Provide a structured evaluation score:** At the end of your response, include a line formatted exactly as: `Evaluation Score: [score]/10`, where [score] is an integer from 0 (lowest promise/soundness) to 10 (highest promise/soundness) based on your analysis.\\n"
            " 6. Return ONLY the analysis, insights, highlighted inconsistencies, and the structured score line."
        )
    ),
    input_schema=None,
    tools=[]
)

# Critic Agent
critic_agent = Agent(
    name="Critic",
    model=root_planner_critic_config, # Use configured Google model
    description="Critically evaluates ideas, identifies flaws, MUST suggest alternatives/revisions, and provides a structured evaluation score.",
    instruction=_create_agent_instruction(
        specialist_name="Quality Controller (Critic)",
        core_task=(
            " 1. Read the critique instruction from session state with key 'critic_instruction'.\\n"
            " 2. If no instruction is found, provide a general critique of the content.\\n"
            " 3. Identify potential biases, flaws, logical fallacies, or weaknesses.\\n"
            " 4. **Crucially: Suggest specific improvements AND propose at least one concrete alternative approach or direction.** If a major flaw invalidates a previous thought, clearly state this and suggest targeting that thought for revision.\\n"
            " 5. Formulate a constructive response containing your evaluation, identified flaws, and mandatory suggestions/alternatives.\\n"
            " 6. **Provide a structured evaluation score:** At the end of your response, include a line formatted exactly as: `Evaluation Score: [score]/10`, where [score] is an integer from 0 (lowest viability/promise) to 10 (highest viability/promise) based on your critique.\\n"
            " 7. Return ONLY the critique, suggestions/alternatives, and the structured score line."
        )
    ),
    input_schema=None,
    tools=[]
)

# Synthesizer Agent
synthesizer_agent = Agent(
    name="Synthesizer",
    model=analyzer_synth_config, # Use configured Google model
    description="Integrates information or forms conclusions based on delegated synthesis sub-tasks.",
    instruction=_create_agent_instruction(
        specialist_name="Integration Specialist",
        core_task=(
            " 1. Read the synthesis instruction from session state with key 'synthesizer_instruction'.\\n"
            " 2. If no instruction is found, integrate the available information in a general way.\\n"
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

class ToTBeamSearchCoordinator(BaseAgent):
    """
    Orchestrates a Tree-of-Thought (ToT) workflow using Beam Search.

    Manages the execution flow, state (thought_tree, active_beam, workflow_phase),
    and calls specialist LLM agents (Planner, Analyzer, Critic, Synthesizer)
    as tools to perform specific tasks within the algorithm.
    
    This is a Custom Agent implementation that directly inherits from BaseAgent
    and implements custom orchestration logic in _run_async_impl.
    """

    # --- Field Declarations for Pydantic ---
    # Declare sub-agents and tools as instance attributes with type hints
    planner: LlmAgent
    researcher: LlmAgent # Add researcher agent
    analyzer: LlmAgent
    critic: LlmAgent
    synthesizer: LlmAgent
    validator: FunctionTool

    beam_width: int = Field(default=3, description="Number of top nodes (k) to keep in the beam.")
    max_depth: int = Field(default=5, description="Maximum depth of the tree to explore.")
    model: LlmAgent | LiteLlm | str 

    # model_config allows setting Pydantic configurations if needed
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        planner: LlmAgent,
        researcher: LlmAgent, # Add researcher parameter
        analyzer: LlmAgent,
        critic: LlmAgent,
        synthesizer: LlmAgent,
        validator: FunctionTool,
        beam_width: int = 3,
        max_depth: int = 5,
        model: LlmAgent | LiteLlm | str = None,
    ):
        """
        Initializes the ToTBeamSearchCoordinator.

        Args:
            name: The name of the agent.
            planner: LlmAgent for generating thoughts/strategies.
            researcher: LlmAgent for gathering information about thoughts. # Add description
            analyzer: LlmAgent for analyzing thought soundness.
            critic: LlmAgent for critiquing thoughts.
            synthesizer: LlmAgent for synthesizing final results.
            validator: FunctionTool for validating thought data.
            beam_width: Number of nodes to keep in the beam (k).
            max_depth: Maximum search depth.
            model: Optional model for the coordinator itself.
        """
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
        super().__init__(
            name=name,
            description=f"Tree of Thoughts Beam Search (width={beam_width}, depth={max_depth})",
            model=model,
            planner=planner,
            researcher=researcher, # Pass researcher to super
            analyzer=analyzer,
            critic=critic,
            synthesizer=synthesizer,
            validator=validator,
            beam_width=beam_width,
            max_depth=max_depth,
            sub_agents=sub_agents_list,  # Pass the explicit sub_agents list
        )
        
        # Log initialization
        logger.info(f"[{self.name}] ToT Coordinator initialized with beam width {beam_width} and max depth {max_depth}")

    # --- Helper methods for state management ---
    def _get_state_value(self, ctx: InvocationContext, key: str, default: Any = None) -> Any:
        """Safely gets a value from session state."""
        return ctx.session.state.get(key, default)

    def _set_state_value(self, ctx: InvocationContext, key: str, value: Any):
        """Sets a value in session state."""
        ctx.session.state[key] = value

    def _get_thought_tree(self, ctx: InvocationContext) -> Dict[str, Any]:
        """Gets the thought tree from state, initializing if necessary."""
        return ctx.session.state.setdefault("thought_tree", {})

    def _get_active_beam(self, ctx: InvocationContext) -> List[str]:
        """Gets the active beam from state, initializing if necessary."""
        return ctx.session.state.setdefault("active_beam", [])

    def _set_active_beam(self, ctx: InvocationContext, beam: List[str]):
        """Sets the active beam in state."""
        self._set_state_value(ctx, "active_beam", beam)
        
    def _update_node(self, ctx: InvocationContext, node_id: str, data: Dict[str, Any]):
        """Updates a node in the thought tree."""
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
        """Extracts evaluation score from text."""
        if not isinstance(text, str): # Add robustness
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

    # --- Helper method to safely run Sub Agents and filter empty model events ---
    async def _run_sub_agent_safely(self, agent: LlmAgent, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Wraps a sub-agent call to filter out model events with empty parts.

        Args:
            agent: The LlmAgent instance to run (e.g., self.planner, self.researcher).
            ctx: The InvocationContext.

        Yields:
            Valid events from the sub-agent.
        """
        async for event in agent.run_async(ctx):
            # --- FIX: Check for role attribute before accessing --- 
            if hasattr(event, 'role') and event.role == 'model':
                # Now safe to check content and parts for model role
                is_empty_model_event = (
                    event.content and
                    not event.content.parts
                )
                if is_empty_model_event:
                    logger.warning(f"[{self.name}] Filtering empty model event from {agent.name} response to prevent history pollution.")
                    continue # Skip yielding this event
            # --- END FIX ---
            
            # Yield all other events (user, tool_code, tool_calls, valid model) 
            # or events without a role attribute
            yield event 

    # --- Core Execution Logic ---
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the ToT Beam Search algorithm orchestration.
        
        Similar to the StoryFlowAgent example, we use a clear step-by-step workflow:
        1. Initialize the tree (generate root and initial nodes)
        2. Then enter the main beam search loop:
           a. Generate candidate thoughts
           b. Evaluate those thoughts
           c. Select the best (beam) nodes
        3. Finally, synthesize results from the best path
        
        We also include conditional logic for handling edge cases and errors.
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

        # Set beam to initial nodes for first iteration
        self._set_active_beam(ctx, newly_added_ids)
        logger.info(f"[{self.name}] Initial beam set with {len(newly_added_ids)} active nodes.")
        
        # --- Phase 2: Main Beam Search Loop ---
        max_iterations = self.max_depth
        current_depth = 1  # Root is at depth 0, initial nodes at depth 1
        
        for iteration in range(max_iterations):
            thought_tree = self._get_thought_tree(ctx)
            active_beam = self._get_active_beam(ctx)
            
            # Check termination condition
            if not active_beam:
                logger.info(f"[{self.name}] Beam is empty, terminating search.")
                break
                
            if current_depth >= self.max_depth:
                logger.info(f"[{self.name}] Reached maximum depth {self.max_depth}, terminating search.")
                break
                
            logger.info(f"[{self.name}] Iteration {iteration+1}/{max_iterations}, depth {current_depth}, beam size {len(active_beam)}")
            
            # Step 2a: Generate next thoughts from active beam
            logger.info(f"[{self.name}] Step 2a: Generating next thoughts...")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Generating thoughts (iteration {iteration+1})...")])
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
                
            # Step 2c: Select next beam (best nodes)
            logger.info(f"[{self.name}] Step 2c: Selecting best nodes for next beam...")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text="Selecting best thoughts for next iteration...")])
            )
            
            new_beam = await self._select_next_beam(ctx)
            if new_beam:
                self._set_active_beam(ctx, new_beam)
                logger.info(f"[{self.name}] New beam selected with {len(new_beam)} nodes.")
            else:
                logger.info(f"[{self.name}] Selection resulted in empty beam, terminating search.")
                break
                
            # Increment depth for next iteration
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
        Initializes the thought tree with a root node and generates initial STRATEGIES.

        This helper method demonstrates:
        1. How to call the validator tool and handle its response
        2. How to call a sub-agent (planner) to generate distinct strategies
        3. How to manage state through the ctx.session.state dictionary

        Stores the generated initial node IDs in ctx.session.state['_initialize_workflow_result']
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
        yield Event( # Yield validation event
            author=self.validator.name, invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text=f"Validated thought: {validation_result.get('message', 'No message')}")])
        )
        if validation_result.get("validation_status") != "success":
            logger.error(f"[{self.name}] Root node validation failed: {validation_result.get('error')}.")
            return
        self._update_node(ctx, root_id, validation_result)
        logger.info(f"[{self.name}] Root node added to tree.")

        # 4. Call Planner to generate distinct initial STRATEGIES, not just a definition
        logger.info(f"[{self.name}] Calling Planner to generate initial *strategies*.")
        try:
            # --- STRONGLY REVISED PLANNER INSTRUCTION for Initialization ---
            planner_instruction = (
                f"Your **sole task** right now is to generate exactly **{self.beam_width} distinct high-level strategies** "
                f"to approach the problem: '{initial_problem}'.\n"
                f"**CRITICAL FORMATTING REQUIREMENT:**\n"
                f"1. Each strategy MUST be a concise phrase or sentence describing an approach.\n"
                f"2. You MUST output *only* the strategies, each on a new line.\n"
                f"3. Each line MUST start *exactly* with 'Strategy N: ' (e.g., 'Strategy 1: Analyze philosophical aspects.').\n"
                f"**DO NOT** include any introductory text, explanations, definitions, or any other text before, between, or after the strategy list. "
                f"Your entire output should consist only of the lines matching 'Strategy N: ...'."
            )
            ctx.session.state["planner_instruction"] = planner_instruction
            # --- END OF REVISED INSTRUCTION ---

            planner_output = ""
            logger.info(f"[{self.name}] Sending request to Planner for initial strategies...")
            async for event in self._run_sub_agent_safely(self.planner, ctx):
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

            # Ensure we don't exceed beam width
            initial_strategies = initial_strategies[:self.beam_width]
            logger.info(f"[{self.name}] Finalizing with {len(initial_strategies)} initial strategies for the beam.")
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
                yield Event( # Yield validation event for strategy node
                    author=self.validator.name, invocation_id=ctx.invocation_id,
                    content=types.Content(parts=[types.Part(text=f"Validated thought: {child_validation_result.get('message', 'No message')}")])
                )

                if child_validation_result.get("validation_status") == "success":
                    self._update_node(ctx, child_id, child_validation_result)
                    newly_added_ids.append(child_id)
                    logger.info(f"[{self.name}] Added initial strategy node: '{strategy_thought[:50]}...' ({child_id})")
                else:
                    logger.warning(f"[{self.name}] Validation failed for strategy path {i}.")

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
        """Generates candidate thoughts for the next level based on the active beam.

        Stores the newly generated node IDs in ctx.session.state['_generate_next_thoughts_result']
        before returning.
        """
        active_beam = self._get_active_beam(ctx)
        thought_tree = self._get_thought_tree(ctx)
        logger.info(f"[{self.name}] Starting Generation Step for beam: {active_beam}")
        newly_generated_ids_this_round = []
        # Initialize result in state
        self._set_state_value(ctx, '_generate_next_thoughts_result', [])

        for parent_id in active_beam:
            parent_node = thought_tree.get(parent_id)
            # Ensure node exists and is marked active (selected in previous step)
            if not parent_node or parent_node.get('status') != 'active':
                logger.debug(f"[{self.name}] Skipping generation for non-active node: {parent_id}")
                continue

            parent_thought = parent_node.get("thoughtContent", "") # Get thought from validated data
            parent_depth = parent_node.get("depth", 0)

            if parent_depth >= self.max_depth:
                 logger.info(f"[{self.name}] Node {parent_id} is at max depth {self.max_depth}. Skipping generation.")
                 continue

            logger.info(f"[{self.name}] Calling Planner to expand node: {parent_id} ('{parent_thought[:50]}...')")
            try:
                # --- Dynamic Generation Count Logic --- 
                # Base number of thoughts to generate
                base_num_to_generate = self.beam_width 
                
                # Adjust based on parent node's score (if evaluated - might not be if coming from root)
                parent_score = parent_node.get("evaluationScore")
                score_adjustment = 0
                if parent_score is not None:
                    if parent_score >= 8.0: # High score - explore more
                        score_adjustment = 1
                    elif parent_score < 5.0: # Low score - explore less
                        score_adjustment = -1
                
                # Adjust based on depth (deeper nodes might need more focus)
                depth_adjustment = 0
                if parent_depth >= self.max_depth - 2: # Getting close to max depth
                     depth_adjustment = -1

                # Calculate final number, ensuring it's at least 1
                num_to_generate = max(1, base_num_to_generate + score_adjustment + depth_adjustment)
                logger.info(f"[{self.name}] Dynamically determined to generate {num_to_generate} thoughts for node {parent_id} (base={base_num_to_generate}, score_adj={score_adjustment}, depth_adj={depth_adjustment})")
                # --- End Dynamic Generation Count Logic ---

                # --- MODIFIED PLANNER INSTRUCTION for Expansion ---
                # Use the dynamically calculated num_to_generate
                planner_instruction = (
                    f"The current thought/strategy is: '{parent_thought}'. "
                    f"It is at depth {parent_depth} and received a score of {parent_score if parent_score is not None else 'N/A'}. " # Add context
                    f"Based on this, generate exactly **{num_to_generate}** distinct and concrete **next steps, sub-topics, or specific questions** "
                    f"to explore *within* this thought. Focus on quality and relevance. "
                    f"List each clearly on a new line."
                )
                ctx.session.state["planner_instruction"] = planner_instruction
                # --- END OF MODIFIED INSTRUCTION ---
                
                # Correct: Use async for to iterate over agent events
                planner_output = ""
                async for event in self._run_sub_agent_safely(self.planner, ctx):
                    # Pass through the events from the planner
                    yield event
                    # Extract output from the final event if possible
                    if event.content and event.content.parts:
                        planner_output = event.content.parts[0].text # Accumulate or take last? Assume last for simplicity

                child_thoughts_text = planner_output # Use accumulated/final output
                child_thoughts = [p.strip() for p in child_thoughts_text.split('\n') if p.strip()]
                
                # --- FIX: Enforce generation count --- 
                if len(child_thoughts) > num_to_generate:
                    logger.warning(f"[{self.name}] Planner returned {len(child_thoughts)} thoughts, exceeding request for {num_to_generate}. Truncating.")
                    child_thoughts = child_thoughts[:num_to_generate]
                elif len(child_thoughts) < num_to_generate:
                     logger.warning(f"[{self.name}] Planner returned {len(child_thoughts)} thoughts, less than requested {num_to_generate}.")
                # --- END FIX ---

                if not child_thoughts:
                     logger.warning(f"[{self.name}] Planner returned no distinct thoughts for node {parent_id}. Output: {child_thoughts_text}")
                     continue

                # --- FIX: Skip empty thoughts before validation --- 
                for i, child_thought in enumerate(child_thoughts):
                    if not child_thought: # Explicitly check for empty string AFTER strip
                        logger.warning(f"[{self.name}] Skipping empty thought generated by Planner for parent {parent_id} at index {i}.")
                        continue 
                # --- END FIX ---

                for i, child_thought in enumerate(child_thoughts):
                    # Create a unique ID that avoids collisions if planner returns same thought multiple times
                    child_id = f"{parent_id}-gen{i}"
                    child_validation_args = {
                        "parentId": parent_id,
                        "thoughtId": child_id,
                        "thought": child_thought,
                        "depth": parent_depth + 1,
                        "status": "generated", # Mark ready for evaluation
                    }
                    # Correct: await on FunctionTool.run_async
                    child_validation_result = await self.validator.run_async(tool_context=ctx, args=child_validation_args)
                    # Create event properly for child validation result
                    yield Event(
                        author=self.validator.name,
                        invocation_id=ctx.invocation_id,
                        content=types.Content(
                            parts=[types.Part(text=f"Validated thought: {child_validation_result.get('message', 'No message')}")]
                        )
                    )

                    if child_validation_result.get("validation_status") == "success":
                        self._update_node(ctx, child_id, child_validation_result)
                        newly_generated_ids_this_round.append(child_id)
                        logger.info(f"[{self.name}] Generated child node: {child_id}")
                    else:
                        logger.warning(f"[{self.name}] Validation failed for generated child of {parent_id}: {child_validation_result.get('error')}")

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
        Evaluates all thoughts currently marked as 'generated'. It now includes a research step.

        This helper method:
        1. Finds all nodes with status 'generated'
        2. Calls researcher for each node to gather information.
        3. Calls analyzer and critic for each node, providing research context.
        4. Combines scores and updates node status to 'evaluated'.
        5. Stores evaluation results (including research) in session state.

        Stores the evaluated node data (list of dicts) in
        ctx.session.state['_evaluate_thoughts_result']
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
            research_findings = "No research conducted." # Default value
            try:
                research_instruction = (
                    f"Gather relevant information and context for the following thought using your search tool. "
                    f"Focus on facts, potential issues, or supporting data.\nThought: {node_thought}"
                )
                # Store instruction in session state (researcher agent expects this)
                # Check if researcher expects instruction via session state or direct input
                # Assuming session state for now, based on other agents.
                # If it takes direct input, the call would be different.
                ctx.session.state["researcher_instruction"] = research_instruction

                logger.info(f"[{self.name}] Calling Researcher for node {node_id}...")
                current_research_output = ""
                final_research_text = None # Use None to indicate no valid output yet
                async for event in self._run_sub_agent_safely(self.researcher, ctx):
                    yield event # Pass through researcher events
                    if event.content and event.content.parts and event.content.parts[0].text:
                        # Capture the first non-empty text part found
                        if final_research_text is None:
                             final_research_text = event.content.parts[0].text

                # Use captured text or a default message if nothing valid was found
                research_findings = final_research_text if final_research_text is not None else "Researcher returned no specific findings."
                logger.info(f"[{self.name}] Research completed for node {node_id}.")
                # Yield an event summarizing research completion
                yield Event(
                    author=self.researcher.name, # Attribute research output to researcher
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
                # Analyzer instruction includes research findings
                analyzer_instruction = (
                    f"Analyze the soundness, feasibility, and logical consistency of the following thought, "
                    f"considering the research findings below. Provide an 'Evaluation Score: [0-10]/10' at the end.\n\n"
                    f"Thought: {node_thought}\n\n"
                    f"Research Findings:\n{research_findings}"
                )
                ctx.session.state["analyzer_instruction"] = analyzer_instruction
                logger.info(f"[{self.name}] Calling Analyzer for node {node_id} with research context...")
                current_analyzer_output = ""
                final_analyzer_text = None # Use None to indicate no valid output yet
                async for event in self._run_sub_agent_safely(self.analyzer, ctx):
                    yield event
                    if event.content and event.content.parts and event.content.parts[0].text:
                         # Capture the first non-empty text part found
                         if final_analyzer_text is None:
                              final_analyzer_text = event.content.parts[0].text
                
                # Use captured text or a default message if nothing valid was found
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
                # Critic instruction includes research findings
                critic_instruction = (
                    f"Critically evaluate the following thought for flaws, biases, or weaknesses, suggesting improvements "
                    f"or alternatives, considering the research findings below. Provide an 'Evaluation Score: [0-10]/10' "
                    f"based on its promise and completeness at the end.\n\n"
                    f"Thought: {node_thought}\n\n"
                    f"Research Findings:\n{research_findings}"
                )
                ctx.session.state["critic_instruction"] = critic_instruction
                logger.info(f"[{self.name}] Calling Critic for node {node_id} with research context...")
                current_critic_output = ""
                final_critic_text = None # Use None to indicate no valid output yet
                async for event in self._run_sub_agent_safely(self.critic, ctx):
                    yield event
                    if event.content and event.content.parts and event.content.parts[0].text:
                        # Capture the first non-empty text part found
                         if final_critic_text is None:
                              final_critic_text = event.content.parts[0].text
                
                # Use captured text or a default message if nothing valid was found
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

            update_data = {
                "evaluationScore": final_score,
                "status": "evaluated",
                "researchFindings": research_findings, # Store research findings
                "analyzerOutput": analyzer_output,
                "criticOutput": critic_output,
            }
            self._update_node(ctx, node_id, update_data)
            evaluated_nodes_data.append({**node_data, **update_data})

        logger.info(f"[{self.name}] Evaluation completed for {len(evaluated_nodes_data)} nodes.")
        self._set_state_value(ctx, '_evaluate_thoughts_result', evaluated_nodes_data)

    # --- Selection Step ---
    async def _select_next_beam(self, ctx: InvocationContext) -> List[str]:
        """
        Selects the top-k thoughts based on evaluation scores to form the new beam.
        
        This helper method:
        1. Finds nodes marked as 'evaluated'
        2. Sorts them by evaluation score (higher is better)
        3. Selects the top-k nodes based on beam_width
        4. Updates node statuses: selected nodes → 'active', others → 'pruned'
        
        Returns:
            List of node IDs selected for the next beam
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

        # Sort nodes by score (descending order - higher is better)
        nodes_to_consider.sort(key=lambda x: x.get("evaluationScore", 0.0), reverse=True)
        
        # Log scores to help understand selection
        logger.info(f"[{self.name}] Node scores: " + 
                   ", ".join([f"{node.get('validatedThoughtId', 'unknown')}:{node.get('evaluationScore', 0.0):.2f}" 
                             for node in nodes_to_consider[:5]]) +
                   (f" ... and {len(nodes_to_consider)-5} more" if len(nodes_to_consider) > 5 else ""))

        # Select top k nodes for the beam
        top_k_nodes = nodes_to_consider[:self.beam_width]
        top_k_ids = [node["validatedThoughtId"] for node in top_k_nodes]

        logger.info(f"[{self.name}] Selected top {len(top_k_ids)} nodes for new beam: {top_k_ids}")

        # Update node statuses - selected nodes are 'active', others are 'pruned'
        selected_count = 0
        pruned_count = 0
        
        for node_data in nodes_to_consider:
            node_id = node_data["validatedThoughtId"]
            if node_id in top_k_ids:
                self._update_node(ctx, node_id, {"status": "active"})
                selected_count += 1
            else:
                self._update_node(ctx, node_id, {"status": "pruned"})
                pruned_count += 1

        logger.info(f"[{self.name}] Selection complete - {selected_count} nodes marked active, {pruned_count} nodes pruned.")
        return top_k_ids

    # --- Synthesis Step ---
    async def _synthesize_result(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Synthesizes the final result from the best path found in the thought tree.
        
        This helper method:
        1. Identifies the best node based on evaluation score
        2. Traces the path from that node back to the root
        3. Calls the synthesizer agent to generate the final answer based on this path
        
        Stores the synthesizer result dictionary in
        ctx.session.state['_synthesize_result_result']
        """
        thought_tree = self._get_thought_tree(ctx)
        active_beam = self._get_active_beam(ctx)
        
        # Initialize result in state with a default error value
        default_error_result = {"error": "Synthesis could not find a suitable result.", "output": ""}
        self._set_state_value(ctx, '_synthesize_result_result', default_error_result)

        # --- Step 1: Find the best node ---
        best_node_id = None
        highest_score = -1.0
        
        # First check active beam nodes
        if active_beam:
            logger.info(f"[{self.name}] Looking for best node in active beam of {len(active_beam)} nodes.")
            for node_id in active_beam:
                node = thought_tree.get(node_id)
                # Correct: Safely handle score comparison, ensure None doesn't cause type error
                node_score = node.get("evaluationScore")
                if node and node_score is not None and node_score > highest_score:
                    highest_score = node_score
                    best_node_id = node_id
        
        # If no good node in active beam, check all evaluated/pruned nodes
        if not best_node_id:
            logger.info(f"[{self.name}] No suitable node in active beam, checking all nodes.")
            for node_id, node in thought_tree.items():
                # Correct: Safely handle score comparison, ensure None doesn't cause type error
                node_score = node.get("evaluationScore")
                if node.get("status") in ["evaluated", "active", "pruned"] and node_score is not None and node_score > highest_score:
                    highest_score = node_score
                    best_node_id = node_id

        # Fallback to root if needed
        if not best_node_id:
            logger.warning(f"[{self.name}] No evaluated nodes found. Using root node as fallback.")
            best_node_id = "root"

        logger.info(f"[{self.name}] Selected best node for synthesis: {best_node_id} (Score: {highest_score:.2f})")

        # --- Step 2: Trace path back to root ---
        path_to_root = []
        current_id = best_node_id
        max_depth_found = 0
        
        while current_id:
            node = thought_tree.get(current_id)
            if node:
                # Build a meaningful representation of this node in the path
                node_info = (
                    f"ID: {node.get('validatedThoughtId', current_id)}, "
                    f"Depth: {node.get('depth', '?')}, "
                    f"Score: {node.get('evaluationScore', 'N/A')}, "
                    f"Thought: {node.get('thoughtContent', 'N/A')}"
                )
                path_to_root.append(node_info)
                
                # Track maximum depth for logging
                max_depth_found = max(max_depth_found, node.get('depth', 0))
                
                # Move to parent
                current_id = node.get("parentId")
            else:
                logger.warning(f"[{self.name}] Node {current_id} referenced but not found in tree.")
                break
        
        # Reverse to get root-to-leaf order
        path_to_root.reverse()
        path_str = "\n -> ".join(path_to_root)
        
        logger.info(f"[{self.name}] Found path of length {len(path_to_root)}, max depth {max_depth_found}")

        # Get the original problem
        initial_problem = self._get_state_value(ctx, "initial_problem", "Unknown initial problem")
        
        # --- Step 3: Call Synthesizer ---
        logger.info(f"[{self.name}] Calling synthesizer with the best path.")
        try:
            # Create synthesis instruction with context from the best path
            synthesis_instruction = (
                f"Synthesize the final answer or conclusion for the initial problem: '{initial_problem}'.\n\n"
                f"The most promising path of thoughts is:\n{path_str}\n\n"
                f"Based on this path and the overall goal, provide the final synthesized result. "
                f"Be concise but comprehensive in your answer."
            )
            
            # Store instruction in session state for synthesizer
            ctx.session.state["synthesizer_instruction"] = synthesis_instruction
            
            # Correct: Use async for to iterate over agent events
            synthesizer_output = ""
            async for event in self._run_sub_agent_safely(self.synthesizer, ctx):
                # Pass through the events from the synthesizer
                yield event
                # Extract output from the final event if possible
                if event.content and event.content.parts:
                    synthesizer_output = event.content.parts[0].text # Accumulate or take last? Assume last.

            # Create a result structure similar to what was expected before
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
root_agent = ToTBeamSearchCoordinator(
    name="ToT_Coordinator",
    planner=planner_agent,
    researcher=researcher_agent, # Pass researcher agent instance
    analyzer=analyzer_agent,
    critic=critic_agent,
    synthesizer=synthesizer_agent,
    validator=validator_tool,
    beam_width=int(os.environ.get("BEAM_WIDTH", 3)),
    max_depth=int(os.environ.get("MAX_DEPTH", 5)),
    model=root_planner_critic_config,
)

# Log the agent creation
logger.info(f"ToT Beam Search Coordinator initialized with beam width={root_agent.beam_width}, depth={root_agent.max_depth}")

# Note: To use this agent, you would need to create a Runner instance
# and call run() or run_async() with session, user content, etc.
# See the example setup in the documentation.