import json
import os  # Added for environment variables
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# --- Thought Data Model (Used for Tool Argument Validation and Typing) ---

class ThoughtData(BaseModel):
    """
    Represents the data structure for a single thought in the sequential
    thinking process. Used to validate arguments for the 'sequentialthinking' tool.
    """
    thought: str = Field(
        ...,
        description="The content of the current thought or step. Make it specific enough to imply the desired action (e.g., 'Analyze X', 'Critique Y', 'Plan Z', 'Research A').",
        min_length=1
    )
    thoughtNumber: int = Field(
        ...,
        description="The sequence number of this thought.",
        ge=1
    )
    totalThoughts: int = Field(
        ...,
        description="The estimated total thoughts required.",
        ge=1
    )
    nextThoughtNeeded: bool = Field(
        ...,
        description="Indicates if another thought step is needed after this one."
    )
    isRevision: bool = Field(
        False,
        description="Indicates if this thought revises a previous thought."
    )
    revisesThought: Optional[int] = Field(
        None,
        description="The number of the thought being revised, if isRevision is True.",
        ge=1
    )
    branchFromThought: Optional[int] = Field(
        None,
        description="The thought number from which this thought branches.",
        ge=1
    )
    branchId: Optional[str] = Field(
        None,
        description="An identifier for the branch, if branching."
    )
    needsMoreThoughts: bool = Field(
        False,
        description="Indicates if more thoughts are needed beyond the current estimate."
    )

    # Pydantic model configuration
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        # frozen=True, # Keep mutable for potential internal adjustments before validation
        arbitrary_types_allowed=True,
    )

    # --- Validators ---
    @field_validator('revisesThought')
    @classmethod
    def validate_revises_thought(cls, v: Optional[int], values: Dict[str, Any]) -> Optional[int]:
        is_revision = values.data.get('isRevision', False)
        if v is not None and not is_revision:
            raise ValueError('revisesThought can only be set when isRevision is True')
        if v is not None and 'thoughtNumber' in values.data and v >= values.data['thoughtNumber']:
             raise ValueError('revisesThought must be less than thoughtNumber')
        return v

    @field_validator('branchId')
    @classmethod
    def validate_branch_id(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        branch_from_thought = values.data.get('branchFromThought')
        if v is not None and branch_from_thought is None:
            raise ValueError('branchId can only be set when branchFromThought is set')
        return v

    @model_validator(mode='after')
    def validate_thought_numbers(self) -> 'ThoughtData':
        if self.branchFromThought is not None and self.branchFromThought >= self.thoughtNumber:
            raise ValueError('branchFromThought must be less than thoughtNumber')
        return self

    def dict(self) -> Dict[str, Any]:
        """Convert thought data to dictionary format for serialization"""
        return self.model_dump(exclude_none=True)

# --- Sequential Thinking Tool (Metadata Management) ---

def sequentialthinking(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    isRevision: bool = False,
    revisesThought: Optional[int] = None,
    branchFromThought: Optional[int] = None,
    branchId: Optional[str] = None,
    needsMoreThoughts: bool = False
) -> Dict[str, Any]:
    """
    Processes and validates the metadata for one step in a sequential thinking chain.

    This tool primarily validates the input structure using ThoughtData and returns
    the processed metadata. It does NOT maintain history itself; history and flow
    are managed by the calling Agent (the Coordinator).

    Args:
        thought: The content of the current thinking step.
        thoughtNumber: Current sequence number (≥1).
        totalThoughts: Estimated total thoughts needed.
        nextThoughtNeeded: Whether another thought step is needed.
        isRevision: Whether this revises previous thinking.
        revisesThought: Which thought number is being reconsidered.
        branchFromThought: If branching, the thought number of the branch point.
        branchId: Identifier for the branch.
        needsMoreThoughts: If more thoughts are needed beyond the current estimate.

    Returns:
        A dictionary containing the validated and processed thought metadata and status.
    """
    MIN_TOTAL_THOUGHTS = 3

    try:
        # --- Validation using Pydantic Model ---
        thought_input_data = {
            "thought": thought,
            "thoughtNumber": thoughtNumber,
            "totalThoughts": max(MIN_TOTAL_THOUGHTS, totalThoughts),
            "nextThoughtNeeded": nextThoughtNeeded,
            "isRevision": isRevision,
            "revisesThought": revisesThought,
            "branchFromThought": branchFromThought,
            "branchId": branchId,
            "needsMoreThoughts": needsMoreThoughts,
        }
        current_thought_data = ThoughtData(**thought_input_data)

        # --- Adjustments based on flags ---
        if current_thought_data.needsMoreThoughts and current_thought_data.thoughtNumber >= current_thought_data.totalThoughts:
            current_thought_data.totalThoughts = current_thought_data.thoughtNumber + 2
            current_thought_data.nextThoughtNeeded = True
            # Optional: print info if needed for debugging
            # print(f"Info: Extended totalThoughts to {current_thought_data.totalThoughts}")

        if current_thought_data.thoughtNumber >= current_thought_data.totalThoughts and not current_thought_data.needsMoreThoughts:
             current_thought_data.nextThoughtNeeded = False

        # --- Build Result for Agent ---
        result_data = {
            "processedThoughtNumber": current_thought_data.thoughtNumber,
            "estimatedTotalThoughts": current_thought_data.totalThoughts,
            "nextThoughtNeeded": current_thought_data.nextThoughtNeeded,
            "thoughtContent": current_thought_data.thought,
            "isRevision": current_thought_data.isRevision,
            "revisesThoughtNumber": current_thought_data.revisesThought,
            "isBranch": current_thought_data.branchFromThought is not None,
            "branchDetails": {
                "branchId": current_thought_data.branchId,
                "branchOrigin": current_thought_data.branchFromThought,
            } if current_thought_data.branchFromThought is not None else None,
            "status": "success",
            "message": f"Thought metadata {current_thought_data.thoughtNumber} validated."
        }
        return result_data

    except ValidationError as e:
        # Print validation errors to stderr for debugging
        return {
            "error": f"Input validation failed: {e.errors()}",
            "status": "validation_error"
        }
    except Exception as e:
        # Print general errors to stderr
        return {
            "error": f"An unexpected error occurred processing thought metadata: {str(e)}",
            "status": "failed"
        }

# --- Model Configuration ---

# Default Google model
DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash" # Use a stable Gemini model like 1.5 Flash
# Removed LiteLlm default model

# Get optional user override for the model name
LLM_MODEL_NAME = os.environ.get("LLM_MODEL") # User can override the default

# Determine if Vertex AI should be used
use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
google_model_name = LLM_MODEL_NAME or DEFAULT_GOOGLE_MODEL

# Configure the model, checking for necessary credentials/settings
active_model_config = google_model_name # Default to model name string
if use_vertex:
    # Check for Vertex AI specific environment variables
    if not os.environ.get("GOOGLE_CLOUD_PROJECT") or not os.environ.get("GOOGLE_CLOUD_LOCATION"):
        print("Warning: Using Vertex AI, but GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION environment variables are not set.")
        # No change to active_model_config needed, ADK handles the string name for Vertex
else:
    # Check for Google AI Studio API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: Using Google AI Studio (default), but GOOGLE_API_KEY environment variable is not set.")
    # No change to active_model_config needed, ADK handles the string name for AI Studio

# All agents will now use the same Google model configuration
researcher_model_config = active_model_config


# --- Specialist Agent Definitions ---
# Note: For simplicity, these agents currently have no specific tools assigned.
# They rely solely on their instructions and the LLM's capabilities.
# You can add FunctionTool, APIHubToolset, etc., to them as needed.

planner_agent = Agent(
    name="Planner",
    model=active_model_config, # Use configured Google model
    description="Develops strategic plans and roadmaps based on delegated sub-tasks.",
    instruction=(
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "You are the Strategic Planner specialist.\n"
        "You will receive specific sub-tasks from the Team Coordinator related to planning, strategy, or process design.\n"
        "**When you receive a sub-task:**\n"
        " 1. Understand the specific planning requirement delegated to you.\n"
        " 2. Develop the requested plan, roadmap, or sequence of steps.\n"
        " 3. Identify any potential revision/branching points *specifically related to your plan* and note them.\n"
        " 4. Consider constraints or potential roadblocks relevant to your assigned task.\n"
        " 5. Formulate a clear and concise response containing the requested planning output.\n"
        " 6. Return ONLY the planning output as your response."
    ),
    tools=[] # Add planning-specific tools here if needed
)

researcher_agent = Agent(
    name="Researcher",
    model=researcher_model_config, # Use configured Google model
    description="Gathers and validates information based on delegated research sub-tasks.",
    instruction=(
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "You are the Information Gatherer specialist.\n"
        "You will receive specific sub-tasks from the Team Coordinator requiring information gathering or verification.\n"
        "**When you receive a sub-task:**\n"
        " 1. Identify the specific information requested in the delegated task.\n"
        " 2. **Actively use the `google_search` tool** to find relevant facts, data, or context from the web. Prioritize using this tool for external information.\n"
        " 3. Validate information where possible, potentially using search results.\n"
        " 4. Structure your findings clearly.\n"
        " 5. Note any significant information gaps encountered during your research.\n"
        " 6. Formulate a response containing the research findings relevant to the sub-task.\n"
        " 7. Return ONLY the research findings as your response."
    ),
    tools=[google_search] # Add google_search tool here
)

analyzer_agent = Agent(
    name="Analyzer",
    model=active_model_config, # Use configured Google model
    description="Performs analysis based on delegated analytical sub-tasks.",
    instruction=(
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "You are the Core Analyst specialist.\n"
        "You will receive specific sub-tasks from the Team Coordinator requiring analysis, pattern identification, or logical evaluation.\n"
        "**When you receive a sub-task:**\n"
        " 1. Understand the specific analytical requirement of the delegated task.\n"
        " 2. Perform the requested analysis (e.g., break down components, identify patterns, evaluate logic).\n"
        " 3. Generate concise insights based on your analysis.\n"
        " 4. Identify any significant logical inconsistencies or invalidated premises *within the scope of your sub-task*.\n"
        " 5. Formulate a response containing your analytical findings and insights.\n"
        " 6. Return ONLY the analysis and insights as your response."
    ),
    tools=[] # Add analytical tools if needed
)

critic_agent = Agent(
    name="Critic",
    model=active_model_config, # Use configured Google model
    description="Critically evaluates ideas or assumptions based on delegated critique sub-tasks.",
    instruction=(
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "You are the Quality Controller specialist.\n"
        "You will receive specific sub-tasks from the Team Coordinator requiring critique, evaluation of assumptions, or identification of flaws.\n"
        "**When you receive a sub-task:**\n"
        " 1. Understand the specific aspect requiring critique in the delegated task.\n"
        " 2. Critically evaluate the provided information or premise as requested.\n"
        " 3. Identify potential biases, flaws, or logical fallacies.\n"
        " 4. Suggest specific improvements or point out weaknesses constructively.\n"
        " 5. If your critique reveals significant flaws or outdated assumptions, highlight this clearly.\n"
        " 6. Formulate a response containing your critical evaluation and recommendations.\n"
        " 7. Return ONLY the critique and recommendations as your response."
    ),
    tools=[] # Add critique-related tools if needed
)

synthesizer_agent = Agent(
    name="Synthesizer",
    model=active_model_config, # Use configured Google model
    description="Integrates information or forms conclusions based on delegated synthesis sub-tasks.",
    instruction=(
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "You are the Integration Specialist.\n"
        "You will receive specific sub-tasks from the Team Coordinator requiring integration of information, synthesis of ideas, or formation of conclusions.\n"
        "**When you receive a sub-task:**\n"
        " 1. Understand the specific elements needing integration or synthesis.\n"
        " 2. Connect the provided elements, identify overarching themes, or draw conclusions as requested.\n"
        " 3. Distill complex inputs into clear, structured insights.\n"
        " 4. Formulate a response presenting the synthesized information or conclusions.\n"
        " 5. Return ONLY the synthesized output as your response."
    ),
    tools=[] # Add synthesis tools if needed
)

# --- Coordinator Agent Definition (Root Agent) ---

# Create AgentTool instances for each specialist
planner_tool = AgentTool(planner_agent)
researcher_tool = AgentTool(researcher_agent)
analyzer_tool = AgentTool(analyzer_agent)
critic_tool = AgentTool(critic_agent)
synthesizer_tool = AgentTool(synthesizer_agent)

root_agent = Agent(
    name="sequential_thinking_coordinator",
    model=active_model_config, # Use configured Google model
    description=(
        "Coordinates a team of specialist agents (Planner, Researcher, Analyzer, Critic, Synthesizer) "
        "to perform sequential thinking tasks using session state for history, allowing for revisions and branching."
    ),
    instruction=(
        "You are the Coordinator of a specialist team performing a sequential thinking task.\n"
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "Your available specialists are: Planner, Researcher, Analyzer, Critic, Synthesizer.\n"
        "Your available tools are: `sequentialthinking` (for validating thought metadata) and tools to call each specialist (`Planner`, `Researcher`, `Analyzer`, `Critic`, `Synthesizer`).\n\n"
        "**State Management:** You MUST use `session.state['thought_history']` (a list of dictionaries) to manage the structured history of the thinking process within this session. Initialize it as an empty list if it doesn't exist.\n\n"
        "**Your Workflow:**\n"
        "1.  **Initialize/Load State:** Access the current invocation context (`ctx`). Check if `ctx.session.state['thought_history']` exists. If not, initialize it: `ctx.session.state['thought_history'] = []`. Read the existing history from this state variable.\n"
        "2.  **Receive Input/Context:** Get the initial user problem or the result/context from the *last entry* in `ctx.session.state['thought_history']`.\n"
        "3.  **Plan Thought:** Based on the input and history, decide the content of the *current* thought (e.g., 'Analyze assumptions', 'Plan next steps', 'Revise thought #X', 'Branch to explore Y'). Determine the current `thoughtNumber` (usually `len(ctx.session.state['thought_history']) + 1`, adjust for revisions/branches if necessary).\n"
        "4.  **Log Thought Metadata:** Call the `sequentialthinking` tool with the planned `thought` content and all necessary metadata (`thoughtNumber`, `totalThoughts`, `nextThoughtNeeded`, `isRevision`, etc.).\n"
        "5.  **Analyze & Delegate:** Based *only* on the `thoughtContent` returned by `sequentialthinking`, determine the primary action (plan, research, analyze, critique, synthesize) and identify the **single most appropriate specialist** agent required for this specific thought step.\n"
        "6.  **Call Specialist Tool:** Use the corresponding AgentTool (e.g., `Planner` tool) to delegate the task. Pass the relevant `thoughtContent` and necessary context (potentially summarized from `ctx.session.state['thought_history']`) as the input prompt to the specialist tool.\n"
        "7.  **Receive & Record:** Get the response from the specialist agent tool. Create a dictionary containing the metadata from the `sequentialthinking` call and the specialist's response.\n"
        "8.  **Update State:** Append the dictionary created in the previous step to `ctx.session.state['thought_history']`.\n"
        "9.  **Decide Next Step:** Based on the latest entry in `ctx.session.state['thought_history']` and the overall goal:\n"
        "    *   If the `sequentialthinking` result indicated `nextThoughtNeeded` was true, plan the *next* thought and go back to step 3.\n"
        "    *   Consider if revision or branching is needed based on the specialist's output. If so, plan the revision/branch thought for the next step.\n"
        "    *   If `nextThoughtNeeded` was false, proceed to step 10.\n"
        "10. **Final Answer:** Synthesize the *entire sequence* of thoughts by processing the structured data in `ctx.session.state['thought_history']`. Provide a comprehensive final answer to the initial user request.\n\n"
        "**Important Rules:**\n"
        "*   Always call `sequentialthinking` *first* for each thought step.\n"
        "*   Delegate to **only one** specialist agent tool per thought step.\n"
        "*   Use the specialist's response (recorded in state) to inform the *next* thought.\n"
        "*   **Crucially, use `session.state['thought_history']` for reading and storing the structured history.** Access the context via `ctx.session.state`."
    ),
    # The sub_agents parameter is removed. Hierarchy is implied by tool usage.
    tools=[
        sequentialthinking,
        planner_tool,
        researcher_tool,
        analyzer_tool,
        critic_tool,
        synthesizer_tool
    ],
    # Enable state persistence if you want state to survive agent restarts *within the same session*,
    # but remember it's still tied to the SessionService implementation for true persistence.
    # enable_state_persistence=True,
) 