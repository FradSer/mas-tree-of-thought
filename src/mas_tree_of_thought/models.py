from typing import Any, Dict, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError, # This was not directly used in the snippet but good for context if model evolves
    field_validator,
)

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