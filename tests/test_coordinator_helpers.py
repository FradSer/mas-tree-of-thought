import pytest
from unittest.mock import MagicMock
from mas_tree_of_thought.coordinator import ToTCoordinator
from mas_tree_of_thought.validation import validate_thought_node_data # For validator mock

@pytest.fixture
def coordinator_instance():
    """Provides a ToTCoordinator instance with mocked dependencies."""
    # Mock the validator function if it's called directly or via a helper in __init__
    # For the purpose of these helper tests, a simple mock is sufficient.
    mock_validator = MagicMock(return_value={"validation_status": "success", "message": "Mocked validation."})

    return ToTCoordinator(
        name="TestCoordinator",
        planner=MagicMock(),
        researcher=MagicMock(),
        analyzer=MagicMock(),
        critic=MagicMock(),
        synthesizer=MagicMock(),
        validator=mock_validator, # Pass the mocked validator
        model="gemini-2.0-flash"  # Using a string model ID as it's simpler
    )

# Tests for _extract_score
@pytest.mark.parametrize("text_input, expected_score", [
    ("Evaluation Score: 7/10", 7.0),
    ("Some text before Evaluation Score: 8/10 and after.", 8.0),
    ("Evaluation Score: 10/10", 10.0),
    ("Evaluation Score: 0/10", 0.0),
    ("Evaluation Score: 7.5/10", 7.5),
    ("Evaluation Score: 5.0/10", 5.0),
    ("EVALUATION SCORE: 6/10", 6.0), # Test case-insensitivity of the prefix
    ("evaluation score: 3/10", 3.0), # Test case-insensitivity
    ("   Evaluation Score: 9/10   ", 9.0), # Test with leading/trailing spaces in text
    ("Evaluation Score: 7/10. Some other text.", 7.0),
])
def test_extract_score_valid(coordinator_instance, text_input, expected_score):
    assert coordinator_instance._extract_score(text_input) == expected_score

@pytest.mark.parametrize("text_input", [
    "Evaluation Score: 11/10",
    "Evaluation Score: -1/10",
    "Evaluation Score: 10.1/10",
    "Evaluation Score: -0.5/10",
])
def test_extract_score_invalid_range(coordinator_instance, text_input):
    assert coordinator_instance._extract_score(text_input) is None

@pytest.mark.parametrize("text_input", [
    "No score here.",
    "",
    "Evaluation Score: /10",
    "Evaluation Score: 7/",
    "Evaluation Score: /",
    "Completely different text."
])
def test_extract_score_missing_pattern_or_incomplete(coordinator_instance, text_input):
    assert coordinator_instance._extract_score(text_input) is None

@pytest.mark.parametrize("text_input", [
    "Evaluation Score: seven/10",
    "Evaluation Score: 7 /10",   # Space before slash
    "Evaluation Score: 7/ 10",   # Space after slash
    "Evaluation Score: 7 / 10", # Spaces around slash
    "Evaluation Score: 7.5.5/10", # Invalid float
    "Evaluation Score: 7a/10",
    "Evaluation Score: 7/10a",
    "Evaluation Score: not_a_number/10",
])
def test_extract_score_malformed(coordinator_instance, text_input):
    assert coordinator_instance._extract_score(text_input) is None

@pytest.mark.parametrize("text_input", [
    None,
    123,
    [],
    {},
])
def test_extract_score_not_string(coordinator_instance, text_input):
    assert coordinator_instance._extract_score(text_input) is None

# Tests for _extract_termination_recommendation
@pytest.mark.parametrize("text_input, expected_recommendation", [
    ("Termination Recommendation: True", True),
    ("Termination Recommendation: False", False),
    ("Termination Recommendation: true", True), # Case-insensitivity for value
    ("Termination Recommendation: false", False),# Case-insensitivity for value
    ("TERMINATION RECOMMENDATION: True", True), # Case-insensitivity for prefix
    ("termination recommendation: False", False),# Case-insensitivity for prefix
    ("  Termination Recommendation: True  ", True), # Leading/trailing spaces in text
    ("Termination Recommendation:  True", True),   # Space after colon
    ("Some text before. Termination Recommendation: False. Some text after.", False),
])
def test_extract_termination_recommendation_valid(coordinator_instance, text_input, expected_recommendation):
    assert coordinator_instance._extract_termination_recommendation(text_input) is expected_recommendation

@pytest.mark.parametrize("text_input", [
    "No recommendation here.",
    "",
    "Termination Recommendation:",
    "Termination Recommendation: ",
    "Completely different text."
])
def test_extract_termination_recommendation_missing_or_incomplete(coordinator_instance, text_input):
    # As per current implementation, missing pattern returns False
    assert coordinator_instance._extract_termination_recommendation(text_input) is False

@pytest.mark.parametrize("text_input", [
    "Termination Recommendation: Yes",
    "Termination Recommendation: No",
    "Termination Recommendation: T",
    "Termination Recommendation: F",
    "Termination Recommendation: truefalse",
    "Termination Recommendation: 1",
    "Termination Recommendation: 0",
    "Termination Recommendation: maybe",
])
def test_extract_termination_recommendation_malformed(coordinator_instance, text_input):
    # Malformed values should result in False
    assert coordinator_instance._extract_termination_recommendation(text_input) is False

@pytest.mark.parametrize("text_input", [
    None,
    123,
    [],
    {},
])
def test_extract_termination_recommendation_not_string(coordinator_instance, text_input):
    # Non-string inputs should result in False
    assert coordinator_instance._extract_termination_recommendation(text_input) is False

print("Successfully created tests/test_coordinator_helpers.py")
