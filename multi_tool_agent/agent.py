import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      field_validator, model_validator)

# --- Logging Setup ---

def setup_logging() -> logging.Logger:
    """
    Set up application logging with both file and console handlers.
    Logs will be stored in the user's home directory under .sequential_thinking/logs.

    Returns:
        Logger instance configured with both handlers.
    """
    # Create logs directory in user's home
    home_dir = Path.home()
    log_dir = home_dir / ".sequential_thinking" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("sequential_thinking_adk") # Changed name slightly
    # Avoid adding handlers multiple times if the module is reloaded
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler with rotation
    log_file = log_dir / "sequential_thinking_adk.log"
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except PermissionError:
        print(f"Warning: Could not write to log file {log_file}. Check permissions.", file=sys.stderr)
        # Fallback or skip file logging if needed


    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()
logger.info("Sequential Thinking ADK Logger Initialized.")


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

# --- Utility for Formatting Thoughts (for Logging) ---

def format_thought_for_log(thought_data: ThoughtData) -> str:
    """Formats a thought for logging purposes, handling multi-byte characters."""
    prefix = ''
    context = ''
    branch_info_log = '' # Added for explicit branch tracking in log

    if thought_data.isRevision and thought_data.revisesThought is not None:
        prefix = '🔄 Revision'
        context = f' (revising thought {thought_data.revisesThought})'
    elif thought_data.branchFromThought is not None and thought_data.branchId is not None:
        prefix = '🌿 Branch'
        context = f' (from thought {thought_data.branchFromThought}, ID: {thought_data.branchId})'
        branch_info_log = f"Branch Details: ID='{thought_data.branchId}', originates from Thought #{thought_data.branchFromThought}"
    else:
        prefix = '💭 Thought'
        context = ''

    header = f"{prefix} {thought_data.thoughtNumber}/{thought_data.totalThoughts}{context}"

    # Helper to get visual width of a string (approximates multi-byte characters)
    def get_visual_width(s: str) -> int:
        width = 0
        for char in s:
            # Basic approximation: Wide characters (e.g., CJK) take 2 cells, others 1
            if 0x1100 <= ord(char) <= 0x115F or \
               0x2329 <= ord(char) <= 0x232A or \
               0x2E80 <= ord(char) <= 0x3247 or \
               0x3250 <= ord(char) <= 0x4DBF or \
               0x4E00 <= ord(char) <= 0xA4C6 or \
               0xA960 <= ord(char) <= 0xA97C or \
               0xAC00 <= ord(char) <= 0xD7A3 or \
               0xF900 <= ord(char) <= 0xFAFF or \
               0xFE10 <= ord(char) <= 0xFE19 or \
               0xFE30 <= ord(char) <= 0xFE6F or \
               0xFF00 <= ord(char) <= 0xFF60 or \
               0xFFE0 <= ord(char) <= 0xFFE6 or \
               0x1B000 <= ord(char) <= 0x1B001 or \
               0x1F200 <= ord(char) <= 0x1F251 or \
               0x1F300 <= ord(char) <= 0x1F64F or \
               0x1F680 <= ord(char) <= 0x1F6FF:
                width += 2
            else:
                width += 1
        return width

    header_width = get_visual_width(header)
    thought_width = get_visual_width(thought_data.thought)
    max_inner_width = max(header_width, thought_width)
    border_len = max_inner_width + 4 # Accounts for '│ ' and ' │'

    border = '─' * (border_len - 2) # Border width between corners

    # Wrap thought text correctly based on visual width
    thought_lines = []
    current_line = ""
    current_width = 0
    words = thought_data.thought.split()
    for i, word in enumerate(words):
        word_width = get_visual_width(word)
        space_width = 1 if current_line else 0

        if current_width + space_width + word_width <= max_inner_width:
            current_line += (" " if current_line else "") + word
            current_width += space_width + word_width
        else:
            thought_lines.append(current_line)
            current_line = word
            current_width = word_width

        # Add the last line if it exists
        if i == len(words) - 1 and current_line:
             thought_lines.append(current_line)


    # Format lines with padding
    formatted_header = f"│ {header}{' ' * (max_inner_width - header_width)} │"
    formatted_thought_lines = [
        f"│ {line}{' ' * (max_inner_width - get_visual_width(line))} │"
        for line in thought_lines
    ]

    # Include branch info in the log box if applicable
    formatted_branch_info = ''
    if branch_info_log:
        branch_info_width = get_visual_width(branch_info_log)
        padding = ' ' * (max_inner_width - branch_info_width)
        formatted_branch_info = f"\n│ {branch_info_log}{padding} │\n├{'─' * (border_len - 2)}┤" # Corrected newline escape

    return f"""
┌{border}┐
{formatted_header}
├{border}┤
{''.join(formatted_thought_lines)}
{formatted_branch_info} # Insert branch info line here if it exists
└{border}┘"""


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
            # print(f"Info: Extended totalThoughts to {current_thought_data.totalThoughts}") # Replaced logger

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
        # print(f"Validation Error processing tool call: {e}", file=sys.stderr) # Replaced logger
        return {
            "error": f"Input validation failed: {e.errors()}",
            "status": "validation_error"
        }
    except Exception as e:
        # import traceback
        # print(f"Error processing sequentialthinking tool call: {e}\n{traceback.format_exc()}", file=sys.stderr) # Replaced logger
        return {
            "error": f"An unexpected error occurred processing thought metadata: {str(e)}",
            "status": "failed"
        }

# --- Specialist Agent Definitions ---
# Note: For simplicity, these agents currently have no specific tools assigned.
# They rely solely on their instructions and the LLM's capabilities.
# You can add FunctionTool, APIHubToolset, etc., to them as needed.

planner_agent = Agent(
    name="Planner",
    model="gemini-2.0-flash-exp", # Use a consistent model or specify as needed
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
    model="gemini-2.0-flash-exp",
    description="Gathers and validates information based on delegated research sub-tasks.",
    instruction=(
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "You are the Information Gatherer specialist.\n"
        "You will receive specific sub-tasks from the Team Coordinator requiring information gathering or verification.\n"
        "**When you receive a sub-task:**\n"
        " 1. Identify the specific information requested in the delegated task.\n"
        " 2. Use your available tools (if any) or internal knowledge to find relevant facts, data, or context.\n"
        " 3. Validate information where possible.\n"
        " 4. Structure your findings clearly.\n"
        " 5. Note any significant information gaps encountered.\n"
        " 6. Formulate a response containing the research findings relevant to the sub-task.\n"
        " 7. Return ONLY the research findings as your response."
    ),
    tools=[] # Add ExaTools or other search tools here
)

analyzer_agent = Agent(
    name="Analyzer",
    model="gemini-2.0-flash-exp",
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
    model="gemini-2.0-flash-exp",
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
    model="gemini-2.0-flash-exp",
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
    model="gemini-2.0-flash-exp", # Or Gemini 1.5 Pro for more complex reasoning
    description=(
        "Coordinates a team of specialist agents (Planner, Researcher, Analyzer, Critic, Synthesizer) "
        "to perform sequential thinking tasks, allowing for revisions and branching."
    ),
    instruction=(
        "You are the Coordinator of a specialist team performing a sequential thinking task.\n"
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "Your available specialists are: Planner, Researcher, Analyzer, Critic, Synthesizer.\n"
        "Your available tools are: `sequentialthinking` (for managing thought metadata) and tools to call each specialist (e.g., `Planner`, `Researcher`).\n\n"
        "**Your Workflow:**\n"
        "1.  **Receive Input:** Get the initial user problem or the result from a previous thought step.\n"
        "2.  **Plan Thought:** Decide the content of the *current* thought (e.g., 'Analyze assumptions', 'Plan next steps', 'Revise thought #X', 'Branch to explore Y').\n"
        "3.  **Log Thought Metadata:** Call the `sequentialthinking` tool with the planned `thought` content and all necessary metadata (`thoughtNumber`, `totalThoughts`, `nextThoughtNeeded`, `isRevision`, etc.). This validates and records the step structure.\n"
        "4.  **Analyze & Delegate:** Based *only* on the `thoughtContent` returned by `sequentialthinking`, determine the primary action (plan, research, analyze, critique, synthesize) and identify the **single most appropriate specialist** agent required for this specific thought step. If multiple aspects are involved, break it down or prioritize the core action for this step.\n"
        "5.  **Call Specialist:** Use the corresponding AgentTool (e.g., `Planner` tool to call the planner agent) to delegate the task. Pass the relevant `thoughtContent` and any necessary prior context as the input prompt to the specialist tool.\n"
        "6.  **Receive & Integrate:** Get the response from the specialist agent.\n"
        "7.  **Synthesize Step Result:** Combine the specialist's response with the context of the current thought step. This synthesis forms the basis for the *next* thought.\n"
        "8.  **Decide Next Step:** Based on the synthesized result and the overall goal:\n"
        "    *   If more thinking is needed (`nextThoughtNeeded` was true from `sequentialthinking` tool), plan the *next* thought (increment `thoughtNumber`) and go back to step 2.\n"
        "    *   Consider if revision or branching is needed based on the specialist's output (e.g., Critic found a flaw, Analyzer identified ambiguity). If so, plan the revision/branch thought for the next step.\n"
        "    *   If the process is complete (`nextThoughtNeeded` was false), proceed to step 9.\n"
        "9.  **Final Answer:** Synthesize the entire sequence of thoughts (including revisions and branches, based on the history you track in the conversation) and provide a comprehensive final answer to the initial user request.\n\n"
        "**Important Rules:**\n"
        "*   Always call `sequentialthinking` *first* for each thought step to log the metadata.\n"
        "*   Delegate to **only one** specialist agent per thought step, based on the primary action of that thought.\n"
        "*   Use the specialist agent's response to inform the *next* thought step's planning.\n"
        "*   Track the history of thoughts and specialist responses within the conversation context to build the final answer."
    ),
    tools=[
        sequentialthinking,
        planner_tool,
        researcher_tool,
        analyzer_tool,
        critic_tool,
        synthesizer_tool
    ],
    # enable_state_persistence=True, # Consider enabling for longer conversations
) 