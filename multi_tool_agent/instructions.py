from datetime import datetime
from typing import List

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