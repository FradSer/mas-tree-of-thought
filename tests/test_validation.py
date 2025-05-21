import pytest
from mas_tree_of_thought.validation import validate_thought_node_data
# from pydantic import ValidationError # Not directly needed if validate_thought_node_data handles it
# from multi_tool_agent.models import ThoughtData # Not directly needed for these tests

def test_valid_root_node_creation():
    """Test valid input for a root node."""
    result = validate_thought_node_data(
        parentId=None,
        thoughtId="root_node_id",
        thought="This is the initial problem.",
        depth=0,
        status="active"
    )
    assert result["validation_status"] == "success"
    assert result["validatedThoughtId"] == "root_node_id"
    assert result["parentId"] is None
    assert result["depth"] == 0
    assert result["thoughtContent"] == "This is the initial problem."
    assert result["status"] == "active"
    assert "message" in result
    assert result["message"] == "ThoughtNode data is valid."

def test_valid_child_node_creation():
    """Test valid input for a child node."""
    result = validate_thought_node_data(
        parentId="parent_node_id",
        thoughtId="child_node_id",
        thought="This is a child thought.",
        depth=1,
        status="pending"
    )
    assert result["validation_status"] == "success"
    assert result["validatedThoughtId"] == "child_node_id"
    assert result["parentId"] == "parent_node_id"
    assert result["depth"] == 1
    assert result["thoughtContent"] == "This is a child thought."
    assert result["status"] == "pending"
    assert "message" in result
    assert result["message"] == "ThoughtNode data is valid."

def test_valid_input_with_evaluation_score():
    """Test that evaluationScore is correctly passed and included."""
    result = validate_thought_node_data(
        parentId="parent_node_id",
        thoughtId="child_node_id_score",
        thought="This thought has a score.",
        depth=2,
        status="evaluated",
        evaluationScore=0.85
    )
    assert result["validation_status"] == "success"
    assert result["validatedThoughtId"] == "child_node_id_score"
    assert result["parentId"] == "parent_node_id"
    assert result["depth"] == 2
    assert result["thoughtContent"] == "This thought has a score."
    assert result["status"] == "evaluated"
    assert result["evaluationScore"] == 0.85
    assert "message" in result
    assert result["message"] == "ThoughtNode data is valid."

def test_invalid_missing_thought_id():
    """Test invalid input: thoughtId is None."""
    result = validate_thought_node_data(
        parentId=None,
        thoughtId=None,
        thought="Thought with missing ID.",
        depth=0,
        status="active"
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    assert "Thought ID cannot be empty" in result["error"]

def test_invalid_empty_thought_id():
    """Test invalid input: thoughtId is an empty string."""
    result = validate_thought_node_data(
        parentId=None,
        thoughtId="",
        thought="Thought with empty ID.",
        depth=0,
        status="active"
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    assert "Thought ID cannot be empty" in result["error"]

def test_invalid_missing_thought_content():
    """Test invalid input: thought content is None."""
    result = validate_thought_node_data(
        parentId=None,
        thoughtId="thought_id_no_content",
        thought=None,
        depth=0,
        status="active"
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    # Pydantic's error message for a missing required field might be more specific
    assert "Input should be a valid string" in result["error"] or "thoughtContent" in result["error"].lower()


def test_invalid_empty_thought_content():
    """Test invalid input: thought content is an empty string."""
    result = validate_thought_node_data(
        parentId=None,
        thoughtId="thought_id_empty_content",
        thought="",
        depth=0,
        status="active"
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    assert "Thought content cannot be empty" in result["error"]

def test_invalid_missing_depth():
    """Test invalid input: depth is None."""
    result = validate_thought_node_data(
        parentId=None,
        thoughtId="thought_id_no_depth",
        thought="This thought has no depth.",
        depth=None,
        status="active"
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    # Pydantic's error message for a missing required field might be more specific
    assert "Input should be a valid integer" in result["error"] or "depth" in result["error"].lower()


def test_invalid_status_value():
    """Test invalid input: status is an unexpected value."""
    result = validate_thought_node_data(
        parentId=None,
        thoughtId="thought_id_invalid_status",
        thought="This thought has an invalid status.",
        depth=0,
        status="this_is_not_a_valid_status"
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    # Pydantic's error message for enum validation
    assert "Input should be 'active', 'pending', 'evaluated', 'rejected', or 'error'" in result["error"] or "status" in result["error"].lower()

def test_all_fields_missing_or_invalid():
    """Test with multiple invalid fields to see combined or first error."""
    result = validate_thought_node_data(
        parentId="parent", # parentId can be anything if other fields are invalid
        thoughtId=None,
        thought=None,
        depth="not_an_int", # Invalid type
        status="invalid_status",
        evaluationScore="not_a_float" # Invalid type
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    # The exact error message might depend on Pydantic's validation order or custom checks.
    # Checking for one of the expected errors is usually sufficient.
    # For example, if thoughtId is checked first:
    assert "Thought ID cannot be empty" in result["error"] or "validation errors for ThoughtData" in result["error"]

def test_invalid_depth_type():
    """Test invalid input: depth is not an integer."""
    result = validate_thought_node_data(
        parentId=None,
        thoughtId="thought_id_invalid_depth_type",
        thought="This thought has invalid depth type.",
        depth="zero", # string instead of int
        status="active"
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    assert "Input should be a valid integer" in result["error"] or "depth" in result["error"].lower()

def test_invalid_evaluation_score_type():
    """Test invalid input: evaluationScore is not a float/None."""
    result = validate_thought_node_data(
        parentId="parent",
        thoughtId="thought_id_invalid_score_type",
        thought="This thought has invalid score type.",
        depth=1,
        status="evaluated",
        evaluationScore="high" # string instead of float/int
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    assert "Input should be a valid number" in result["error"] or "evaluationScore" in result["error"].lower()

def test_status_must_be_evaluated_if_score_is_present():
    """Test that status must be 'evaluated' if evaluationScore is present."""
    result = validate_thought_node_data(
        parentId="parent",
        thoughtId="thought_id_score_status_mismatch",
        thought="Score present, status not evaluated.",
        depth=1,
        status="active", # Not 'evaluated'
        evaluationScore=0.75
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    assert "If evaluationScore is provided, status must be 'evaluated'" in result["error"]

def test_score_not_allowed_for_active_status():
    """Test that evaluationScore is not allowed if status is 'active'."""
    result = validate_thought_node_data(
        parentId="parent",
        thoughtId="active_with_score",
        thought="Active thought with score.",
        depth=1,
        status="active",
        evaluationScore=0.5
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    assert "If evaluationScore is provided, status must be 'evaluated'" in result["error"]

def test_score_not_allowed_for_pending_status():
    """Test that evaluationScore is not allowed if status is 'pending'."""
    result = validate_thought_node_data(
        parentId="parent",
        thoughtId="pending_with_score",
        thought="Pending thought with score.",
        depth=1,
        status="pending",
        evaluationScore=0.5
    )
    assert result["validation_status"] == "validation_error"
    assert "error" in result
    assert "If evaluationScore is provided, status must be 'evaluated'" in result["error"]

def test_valid_evaluated_node_without_score():
    """ Test that an 'evaluated' node can exist without a score (score is Optional) """
    result = validate_thought_node_data(
        parentId="parent123",
        thoughtId="eval_no_score",
        thought="Evaluated but no score yet.",
        depth=1,
        status="evaluated",
        evaluationScore=None # Explicitly None
    )
    assert result["validation_status"] == "success"
    assert result["status"] == "evaluated"
    assert result.get("evaluationScore") is None # or not "evaluationScore" in result
    assert result["message"] == "ThoughtNode data is valid."

def test_valid_evaluated_node_with_zero_score():
    """ Test that an 'evaluated' node can have a zero score. """
    result = validate_thought_node_data(
        parentId="parent123",
        thoughtId="eval_zero_score",
        thought="Evaluated with zero score.",
        depth=1,
        status="evaluated",
        evaluationScore=0.0
    )
    assert result["validation_status"] == "success"
    assert result["status"] == "evaluated"
    assert result["evaluationScore"] == 0.0
    assert result["message"] == "ThoughtNode data is valid."

print("Successfully created tests/test_validation.py") # Placeholder for successful file creation
