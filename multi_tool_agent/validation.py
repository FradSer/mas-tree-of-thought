from typing import Any, Dict, Optional

from pydantic import ValidationError
from google.adk.tools import FunctionTool

from .models import ThoughtData

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

# --- Tool Instantiation ---
validator_tool = FunctionTool(validate_thought_node_data) 