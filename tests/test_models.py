import pytest
from pydantic import ValidationError
from multi_tool_agent.models import ThoughtData

# Valid Data Scenarios
def test_thought_data_valid_root_node():
    data = {
        "thoughtId": "root_node_01",
        "thought": "This is the initial problem statement or question.",
        "depth": 0
    }
    try:
        model = ThoughtData(**data)
        assert model.thoughtId == "root_node_01"
        assert model.thought == "This is the initial problem statement or question."
        assert model.depth == 0
        assert model.parentId is None  # Default value
        assert model.status == "active"  # Default value
        assert model.evaluationScore is None  # Default value
    except ValidationError as e:
        pytest.fail(f"Valid data for root node raised ValidationError: {e}")

def test_thought_data_valid_child_node():
    data = {
        "parentId": "root_node_01",
        "thoughtId": "child_node_02",
        "thought": "This is a follow-up thought or a sub-problem.",
        "depth": 1,
        "status": "pending",
        "evaluationScore": 7.5
    }
    try:
        model = ThoughtData(**data)
        assert model.parentId == "root_node_01"
        assert model.thoughtId == "child_node_02"
        assert model.thought == "This is a follow-up thought or a sub-problem."
        assert model.depth == 1
        assert model.status == "pending"
        assert model.evaluationScore == 7.5
    except ValidationError as e:
        pytest.fail(f"Valid data for child node raised ValidationError: {e}")

def test_thought_data_valid_all_optional_fields_set():
    data = {
        "parentId": "parent_abc",
        "thoughtId": "node_xyz",
        "thought": "A comprehensive thought with all fields.",
        "depth": 2,
        "status": "evaluated",
        "evaluationScore": 9.2
    }
    model = ThoughtData(**data) # Expect no error
    assert model.parentId == "parent_abc"
    assert model.thoughtId == "node_xyz"
    assert model.thought == "A comprehensive thought with all fields."
    assert model.depth == 2
    assert model.status == "evaluated"
    assert model.evaluationScore == 9.2

def test_thought_data_status_default_and_custom():
    # Test default status
    model_default_status = ThoughtData(thoughtId="id_def_stat", thought="Thought content", depth=0)
    assert model_default_status.status == "active"

    # Test custom valid status
    model_custom_status = ThoughtData(thoughtId="id_cust_stat", thought="Thought content", depth=0, status="rejected")
    assert model_custom_status.status == "rejected"

def test_thought_data_evaluation_score_none_and_float():
    # Test default evaluationScore (None)
    model_none_score = ThoughtData(thoughtId="id_none_score", thought="Thought content", depth=0)
    assert model_none_score.evaluationScore is None

    # Test float evaluationScore
    model_float_score = ThoughtData(thoughtId="id_float_score", thought="Thought content", depth=0, evaluationScore=0.0)
    assert model_float_score.evaluationScore == 0.0

    model_float_score_2 = ThoughtData(thoughtId="id_float_score_2", thought="Thought content", depth=0, evaluationScore=10.0)
    assert model_float_score_2.evaluationScore == 10.0


# Invalid Data Scenarios (expect ValidationError)
def test_thought_data_missing_thought_id():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(thought="A thought without an ID", depth=0)
    assert any(err['type'] == 'missing' and err['loc'] == ('thoughtId',) for err in excinfo.value.errors())

def test_thought_data_missing_thought_content():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(thoughtId="id_no_content", depth=0)
    assert any(err['type'] == 'missing' and err['loc'] == ('thought',) for err in excinfo.value.errors())

def test_thought_data_missing_depth():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(thoughtId="id_no_depth", thought="A thought without depth")
    assert any(err['type'] == 'missing' and err['loc'] == ('depth',) for err in excinfo.value.errors())

def test_thought_data_empty_thought_content():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(thoughtId="id_empty_thought", thought="", depth=0)
    # Check for 'string_too_short' or similar, depending on Pydantic version and exact validation
    assert any(err['type'] == 'string_too_short' and err['loc'] == ('thought',) for err in excinfo.value.errors())

def test_thought_data_negative_depth():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(thoughtId="id_neg_depth", thought="A thought with negative depth", depth=-1)
    assert any(err['type'] == 'greater_than_equal' and err['loc'] == ('depth',) for err in excinfo.value.errors())

def test_thought_data_invalid_parent_id_type():
    # This should be caught by the custom validator `validate_parent_id_format` which raises ValueError
    # Pydantic then wraps this in a ValidationError.
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(parentId=12345, thoughtId="id_inv_parent_type", thought="A thought", depth=0)
    # The error from the custom validator will be of type 'value_error'
    assert any(err['type'] == 'value_error' and "parentId must be a string if set" in err['msg'] for err in excinfo.value.errors())


def test_thought_data_invalid_evaluation_score_type():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(thoughtId="id_inv_score", thought="A thought", depth=0, evaluationScore="not_a_float")
    assert any(err['type'] == 'float_type' for err in excinfo.value.errors())


def test_thought_data_invalid_status_type():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(thoughtId="id_inv_status", thought="A thought", depth=0, status=123) # status must be a string
    assert any(err['type'] == 'string_type' for err in excinfo.value.errors())


def test_thought_data_invalid_depth_type():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(thoughtId="id_inv_depth_type", thought="A thought", depth="zero") # depth must be an int
    assert any(err['type'] == 'int_type' for err in excinfo.value.errors())

# Extra Fields
def test_thought_data_extra_field():
    with pytest.raises(ValidationError) as excinfo:
        ThoughtData(
            thoughtId="id_extra",
            thought="A thought with an extra field",
            depth=0,
            unexpected_field="this_should_not_be_allowed"
        )
    assert any(err['type'] == 'extra_forbidden' for err in excinfo.value.errors())

# Assignment Validation (validate_assignment=True)
def test_thought_data_assignment_validation_depth_negative():
    model = ThoughtData(thoughtId="id_assign_depth", thought="Valid thought", depth=1)
    with pytest.raises(ValidationError) as excinfo:
        model.depth = -5  # Attempt to assign invalid value
    assert any(err['type'] == 'greater_than_equal' and err['loc'] == ('depth',) for err in excinfo.value.errors())

def test_thought_data_assignment_validation_thought_empty():
    model = ThoughtData(thoughtId="id_assign_thought", thought="Valid thought", depth=1)
    with pytest.raises(ValidationError) as excinfo:
        model.thought = ""  # Attempt to assign invalid empty string
    assert any(err['type'] == 'string_too_short' and err['loc'] == ('thought',) for err in excinfo.value.errors())

def test_thought_data_assignment_validation_parent_id_invalid_type():
    model = ThoughtData(thoughtId="id_assign_parent", thought="Valid thought", depth=1, parentId="valid_parent")
    with pytest.raises(ValidationError) as excinfo:
        model.parentId = 123 # Attempt to assign invalid type
    # This should also be caught by the custom validator
    assert any(err['type'] == 'value_error' and "parentId must be a string if set" in err['msg'] for err in excinfo.value.errors())


def test_thought_data_assignment_validation_status_invalid_type():
    model = ThoughtData(thoughtId="id_assign_status", thought="Valid thought", depth=1)
    with pytest.raises(ValidationError) as excinfo:
        model.status = [] # Attempt to assign invalid type
    assert any(err['type'] == 'string_type' for err in excinfo.value.errors())

def test_thought_data_assignment_validation_evaluation_score_invalid_type():
    model = ThoughtData(thoughtId="id_assign_score", thought="Valid thought", depth=1)
    with pytest.raises(ValidationError) as excinfo:
        model.evaluationScore = "high_score" # Attempt to assign invalid type
    assert any(err['type'] == 'float_type' for err in excinfo.value.errors())

def test_thought_data_assignment_valid():
    model = ThoughtData(thoughtId="id_assign_valid", thought="Initial thought", depth=0)
    try:
        model.thought = "Updated valid thought"
        model.depth = 1
        model.status = "pending"
        model.parentId = "new_parent"
        model.evaluationScore = 5.0
        assert model.thought == "Updated valid thought"
        assert model.depth == 1
        assert model.status == "pending"
        assert model.parentId == "new_parent"
        assert model.evaluationScore == 5.0
    except ValidationError as e:
        pytest.fail(f"Valid assignment raised ValidationError: {e}")

print("Successfully created tests/test_models.py")
