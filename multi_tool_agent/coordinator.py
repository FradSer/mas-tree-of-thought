import os
import logging
import re
import time
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple

from google.genai import types # Google AI types for Event content
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent # Explicit import for LlmAgent type hint
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from google.adk.events import Event
from typing_extensions import override
from pydantic import Field

logger = logging.getLogger(__name__)

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
            description=f"Tree of Thoughts Coordinator with LLM-driven exploration",
            model=model,
            planner=planner,
            researcher=researcher, # Pass researcher to super
            analyzer=analyzer,
            critic=critic,
            synthesizer=synthesizer,
            validator=validator,
            use_free_tier_rate_limiting=use_free_tier_rate_limiting_env,
            free_tier_sleep_seconds=free_tier_sleep_seconds_env,
            use_vertex_ai=use_vertex_ai_env, # Pass Vertex status too
            sub_agents=sub_agents_list,  # Pass the explicit sub_agents list
        )
        
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
            
        synthesizer_result = self._get_state_value(ctx, '_synthesize_result_result', {})

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
             nodes_to_evaluate = [nid for nid, data in thought_tree.items() if data.get('status') == 'generated']
             if nodes_to_evaluate:
                 return 
             active_beam = self._get_active_beam(ctx)
             if active_beam:
                 return 
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

        validation_args = { 
            "parentId": None, "thoughtId": root_id, "thought": initial_problem,
            "depth": 0, "status": "active",
        }
        validation_result = await self.validator.run_async(tool_context=ctx, args=validation_args)
        if validation_result.get("validation_status") == "success":
            self._update_node(ctx, root_id, validation_result) 
            logger.info(f"[{self.name}] Root node added to tree.")
            yield Event(
                author=self.validator.name, invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Validated Root Node ({root_id}): '{validation_result.get('thoughtContent', 'N/A')[:100]}...'")])
            )
        else:
            logger.error(f"[{self.name}] Root node validation failed: {validation_result.get('error')}. ")
            yield Event(
                author=self.validator.name, invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=f"Validation Failed for Root Node ({root_id}): {validation_result.get('error', 'Unknown error')}")])
            )
            return 

        logger.info(f"[{self.name}] Calling Planner to generate initial *strategies*.")
        try:
            num_initial_strategies = 3 
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

            planner_output = ""
            logger.info(f"[{self.name}] Sending request to Planner for initial strategies...")
            async for event in self._run_sub_agent_safely(self.planner, ctx, dynamic_instruction=planner_instruction):
                yield event
                if event.content and event.content.parts:
                    planner_output = event.content.parts[0].text
            logger.info(f"[{self.name}] Received planner output for initial strategies:\n{planner_output}")

            initial_strategies = []
            strategy_pattern = re.compile(r"^\s*Strategy\s*\d+:\s*(.*)", re.MULTILINE | re.IGNORECASE)
            matches = strategy_pattern.findall(planner_output)

            if matches:
                initial_strategies = [match.strip() for match in matches]
                logger.info(f"[{self.name}] Successfully extracted {len(initial_strategies)} strategies using format 'Strategy N: ...'.")
            else:
                logger.warning(f"[{self.name}] Planner did not follow the 'Strategy N:' format. Attempting fallback newline split.")
                potential_strategies = [p.strip() for p in planner_output.split('\n') if p.strip()]
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

            logger.info(f"[{self.name}] Proceeding with {len(initial_strategies)} initial strategies identified by the planner.")

            newly_added_ids = []
            for i, strategy_thought in enumerate(initial_strategies):
                child_id = f"{root_id}-{i}"
                child_validation_args = {
                    "parentId": root_id, "thoughtId": child_id,
                    "thought": strategy_thought, 
                    "depth": 1, "status": "generated", 
                }
                child_validation_result = await self.validator.run_async(tool_context=ctx, args=child_validation_args)
                if child_validation_result.get("validation_status") == "success":
                    self._update_node(ctx, child_id, child_validation_result) 
                    newly_added_ids.append(child_id)
                    logger.info(f"[{self.name}] Added initial strategy node: '{strategy_thought[:50]}...' ({child_id})")
                    yield Event(
                        author=self.validator.name, invocation_id=ctx.invocation_id,
                        content=types.Content(parts=[types.Part(text=f"Validated Initial Strategy ({child_id}): '{child_validation_result.get('thoughtContent', 'N/A')[:100]}...'")])
                    )
                else:
                    logger.warning(f"[{self.name}] Validation failed for strategy path {i}: {child_validation_result.get('error')}")
                    yield Event(
                        author=self.validator.name, invocation_id=ctx.invocation_id,
                        content=types.Content(parts=[types.Part(text=f"Validation Failed for Initial Strategy ({child_id}): {child_validation_result.get('error', 'Unknown error')}")])
                    )

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
                base_num_to_generate = 2 
                parent_score = parent_node.get("evaluationScore")
                score_adjustment = 0
                if parent_score is not None:
                    if parent_score >= 8.0:
                        score_adjustment = 1 
                    elif parent_score < 4.0: 
                        score_adjustment = -1 

                depth_adjustment = 0

                num_to_generate = max(1, base_num_to_generate + score_adjustment + depth_adjustment)
                logger.info(f"[{self.name}] Dynamically determined to generate {num_to_generate} thoughts for node {parent_id} (base={base_num_to_generate}, score_adj={score_adjustment}, depth_adj={depth_adjustment})")

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
                
                if len(child_thoughts) > num_to_generate:
                    logger.warning(f"[{self.name}] Planner returned {len(child_thoughts)} thoughts, exceeding request for {num_to_generate}. Truncating.")
                    child_thoughts = child_thoughts[:num_to_generate]
                elif len(child_thoughts) < num_to_generate:
                     logger.warning(f"[{self.name}] Planner returned {len(child_thoughts)} thoughts, less than requested {num_to_generate}.")

                if not child_thoughts:
                     logger.warning(f"[{self.name}] Planner returned no distinct thoughts for node {parent_id}. Output: {child_thoughts_text}")
                     continue

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
                    if child_validation_result.get("validation_status") == "success":
                        self._update_node(ctx, child_id, child_validation_result) 
                        newly_generated_ids_this_round.append(child_id)
                        logger.info(f"[{self.name}] Generated child node: {child_id}")
                        yield Event(
                            author=self.validator.name,
                            invocation_id=ctx.invocation_id,
                            content=types.Content(
                                parts=[types.Part(text=f"Validated Thought ({child_id}): '{child_validation_result.get('thoughtContent', 'N/A')[:100]}...'")]
                            )
                        )
                    else:
                        logger.warning(f"[{self.name}] Validation failed for generated child of {parent_id}: {child_validation_result.get('error')}")
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
                continue 

        self._set_state_value(ctx, '_generate_next_thoughts_result', newly_generated_ids_this_round)
        return 

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

            research_findings = "No research conducted."
            try:
                research_instruction = (
                    f"Gather relevant information and context for the following thought using your search tool. "
                    f"Focus on facts, potential issues, or supporting data.\nThought: {node_thought}"
                )
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

            analyzer_score = None
            critic_score = None
            analyzer_output = ""
            critic_output = ""

            try:
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

            scores = [s for s in [analyzer_score, critic_score] if s is not None]
            final_score = sum(scores) / len(scores) if scores else 1.0
            logger.info(f"[{self.name}] Node {node_id} final score: {final_score:.2f}")

            analyzer_term_rec = self._extract_termination_recommendation(analyzer_output)
            critic_term_rec = self._extract_termination_recommendation(critic_output)
            final_termination_rec = analyzer_term_rec or critic_term_rec 
            logger.info(f"[{self.name}] Node {node_id} Termination Recommendation: {final_termination_rec} (Analyzer: {analyzer_term_rec}, Critic: {critic_term_rec})")

            update_data = {
                "evaluationScore": final_score,
                "status": "evaluated", 
                "researchFindings": research_findings,
                "analyzerOutput": analyzer_output,
                "criticOutput": critic_output,
                "terminationRecommended": final_termination_rec, 
            }
            self._update_node(ctx, node_id, update_data)
            evaluated_nodes_data.append({**node_data, **update_data}) 

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
        
        nodes_to_consider = [
            data for data in thought_tree.values()
            if data.get('status') == 'evaluated'
        ]

        logger.info(f"[{self.name}] Selection: found {len(nodes_to_consider)} evaluated nodes to consider.")

        if not nodes_to_consider:
            logger.warning(f"[{self.name}] No evaluated nodes found for selection.")
            return []

        nodes_to_consider.sort(key=lambda x: x.get("evaluationScore", 0.0), reverse=True)

        log_limit = 10
        logger.info(f"[{self.name}] Node scores (Top {log_limit}): " +
                   ", ".join([f"{node.get('validatedThoughtId', 'unknown')}:{node.get('evaluationScore', 0.0):.2f} (Term:{node.get('terminationRecommended', 'F')})" 
                             for node in nodes_to_consider[:log_limit]]) +
                   (f" ... and {len(nodes_to_consider)-log_limit} more" if len(nodes_to_consider) > log_limit else ""))

        selected_count = 0
        pruned_count = 0
        terminated_count = 0
        final_beam = [] 

        for node_data in nodes_to_consider:
            node_id = node_data["validatedThoughtId"]
            termination_recommended = node_data.get("terminationRecommended", False)

            if termination_recommended:
                self._update_node(ctx, node_id, {"status": "terminated_early"})
                terminated_count += 1
            else:
                self._update_node(ctx, node_id, {"status": "active"})
                final_beam.append(node_id) 
                selected_count += 1

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
        active_beam = self._get_active_beam(ctx) 

        default_error_result = {"error": "Synthesis could not find a suitable result.", "output": ""}
        self._set_state_value(ctx, '_synthesize_result_result', default_error_result)

        score_threshold = 7.0 
        top_n_fallback = 3 

        candidate_nodes = []
        for node_id, node_data in thought_tree.items():
            if node_data.get('status') in ['evaluated', 'active', 'terminated_early'] and node_data.get('evaluationScore') is not None:
                 candidate_nodes.append(node_data)

        if not candidate_nodes:
            logger.warning(f"[{self.name}] No nodes with evaluation scores found. Falling back to root node.")
            selected_nodes = [thought_tree.get("root")] if "root" in thought_tree else []
        else:
             candidate_nodes.sort(key=lambda x: x.get("evaluationScore", 0.0), reverse=True)

             selected_nodes = [node for node in candidate_nodes if node.get("evaluationScore", 0.0) >= score_threshold]

             if not selected_nodes:
                 logger.warning(f"[{self.name}] No nodes met score threshold {score_threshold}. Selecting top {top_n_fallback} nodes.")
                 selected_nodes = candidate_nodes[:top_n_fallback] 

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

        logger.info(f"[{self.name}] Calling synthesizer with context from high-scoring thoughts.")
        try:
            synthesis_instruction = (
                 f"Synthesize the final answer or conclusion for the initial problem based on the following promising thoughts identified during exploration:\n\n"
                 f"{synthesis_context}\n\n"
                 f"Integrate the insights from these thoughts to provide a comprehensive and coherent final result. "
                 f"Address the initial problem directly."
            )

            ctx.session.state["synthesizer_instruction"] = synthesis_instruction

            synthesizer_output = ""
            final_synthesizer_text = None 
            async for event in self._run_sub_agent_safely(self.synthesizer, ctx, dynamic_instruction=synthesis_instruction):
                yield event
                if hasattr(event, 'role') and event.role == 'model' and event.author == self.synthesizer.name and event.content and event.content.parts and event.content.parts[0].text:
                     final_synthesizer_text = event.content.parts[0].text

            synthesizer_output = final_synthesizer_text if final_synthesizer_text is not None else "[Synthesizer returned empty output]"

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

            error_result = {
                "error": f"Synthesizer failed: {str(e)}",
                "output": "Failed to generate final result due to an error."
            }
            self._set_state_value(ctx, '_synthesize_result_result', error_result) 