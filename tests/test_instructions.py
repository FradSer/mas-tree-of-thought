import pytest
from unittest.mock import patch
from datetime import datetime

from multi_tool_agent.instructions import _create_agent_instruction

@pytest.fixture
def mock_datetime_now():
    # A fixed datetime object to ensure reproducible tests
    fixed_datetime = datetime(2024, 7, 27, 10, 30, 0)
    # Patching datetime specifically within the module where it's used
    with patch('multi_tool_agent.instructions.datetime') as mock_datetime_module:
        mock_datetime_module.now.return_value = fixed_datetime
        yield fixed_datetime.strftime('%Y-%m-%d %H:%M:%S')

def test_create_basic_instruction(mock_datetime_now):
    specialist_name = "Test Specialist"
    core_task = "This is the core task.\nIt has multiple lines."
    expected_datetime_str = mock_datetime_now

    instruction = _create_agent_instruction(specialist_name, core_task)

    expected_parts = [
        f"Current date and time: {expected_datetime_str}",
        f"You are the {specialist_name} specialist.",
        "You receive specific sub-tasks from the Coordinator.",
        "**Your Task:**",
        core_task  # core_task itself might contain \n
    ]
    expected_instruction = "\n".join(expected_parts)
    
    assert instruction == expected_instruction

def test_instruction_with_extra_guidance(mock_datetime_now):
    specialist_name = "Guided Specialist"
    core_task = "Main task here."
    extra_guidance = [
        "Guidance line 1.",
        "Guidance line 2.\nWith an internal newline."
    ]
    expected_datetime_str = mock_datetime_now

    instruction = _create_agent_instruction(specialist_name, core_task, extra_guidance)

    expected_base_parts = [
        f"Current date and time: {expected_datetime_str}",
        f"You are the {specialist_name} specialist.",
        "You receive specific sub-tasks from the Coordinator.",
        "**Your Task:**",
        core_task
    ]
    # All parts, including extra_guidance, are joined by "\n"
    full_expected_parts = expected_base_parts + extra_guidance
    expected_instruction = "\n".join(full_expected_parts)

    assert instruction == expected_instruction

def test_instruction_empty_core_task(mock_datetime_now):
    specialist_name = "Minimalist Specialist"
    core_task = ""  # Empty core task
    expected_datetime_str = mock_datetime_now

    instruction = _create_agent_instruction(specialist_name, core_task)

    expected_parts = [
        f"Current date and time: {expected_datetime_str}",
        f"You are the {specialist_name} specialist.",
        "You receive specific sub-tasks from the Coordinator.",
        "**Your Task:**",
        core_task  # This will be an empty string
    ]
    expected_instruction = "\n".join(expected_parts)
    
    assert instruction == expected_instruction
    # Check that the line for core_task is indeed empty or just results in an extra newline if core_task is empty
    # If core_task is "", then "**Your Task:**\n" is expected
    # The join will result in "...\n**Your Task:**\n\nGuidance..." if core_task is empty and guidance follows.
    # If core_task is "" and it's the last element, it's "...\n**Your Task:**\n"
    # Let's verify the exact output for an empty core_task at the end of the base instruction list:
    # parts = ["A", "B", ""] -> "A\nB\n"
    # The list is `base_instruction + [core_task]`. If core_task is "", then `base_instruction + [""]`.
    # `"\n".join(["Current...", "You are...", "Sub-tasks...", "**Your Task:**", ""])`
    # This will result in a trailing newline after "**Your Task:**", which is correct.
    assert instruction.endswith("**Your Task:**\n")


def test_instruction_empty_extra_guidance_list(mock_datetime_now):
    specialist_name = "Standard Specialist"
    core_task = "Standard core task."
    extra_guidance = []  # Empty list for extra guidance
    expected_datetime_str = mock_datetime_now

    instruction = _create_agent_instruction(specialist_name, core_task, extra_guidance)

    expected_parts = [
        f"Current date and time: {expected_datetime_str}",
        f"You are the {specialist_name} specialist.",
        "You receive specific sub-tasks from the Coordinator.",
        "**Your Task:**",
        core_task
    ]
    # Since extra_guidance is empty, it adds nothing to the parts list before joining
    expected_instruction = "\n".join(expected_parts)
    
    assert instruction == expected_instruction

def test_instruction_no_extra_guidance_param_uses_default_none(mock_datetime_now):
    specialist_name = "Default Specialist"
    core_task = "Default core task."
    # extra_guidance parameter is not provided, so it should use its default value (None)
    expected_datetime_str = mock_datetime_now

    instruction = _create_agent_instruction(specialist_name, core_task) # No extra_guidance here

    expected_parts = [
        f"Current date and time: {expected_datetime_str}",
        f"You are the {specialist_name} specialist.",
        "You receive specific sub-tasks from the Coordinator.",
        "**Your Task:**",
        core_task
    ]
    expected_instruction = "\n".join(expected_parts)
    
    assert instruction == expected_instruction

def test_core_task_with_trailing_newline(mock_datetime_now):
    specialist_name = "Newline Specialist"
    core_task = "This core task ends with a newline.\n" # Core task itself has a trailing newline
    expected_datetime_str = mock_datetime_now

    instruction = _create_agent_instruction(specialist_name, core_task)

    expected_parts = [
        f"Current date and time: {expected_datetime_str}",
        f"You are the {specialist_name} specialist.",
        "You receive specific sub-tasks from the Coordinator.",
        "**Your Task:**",
        core_task  # "This core task ends with a newline.\n"
    ]
    # The join will add newlines BETWEEN these parts.
    # So, the core_task's own trailing newline will be preserved.
    # Example: join(["A", "B\n"]) -> "A\nB\n"
    expected_instruction = "\n".join(expected_parts)
    
    assert instruction == expected_instruction
    # Specifically, after "**Your Task:**\n", the core_task starts, and it includes its own newline.
    # So, it should look like: ...\n**Your Task:**\nThis core task ends with a newline.\n
    assert instruction.endswith("This core task ends with a newline.\n")

def test_extra_guidance_with_trailing_newlines(mock_datetime_now):
    specialist_name = "Guidance Newline Specialist"
    core_task = "Core task."
    extra_guidance = [
        "Guidance line 1.\n", # Trailing newline in a guidance line
        "Guidance line 2."
    ]
    expected_datetime_str = mock_datetime_now

    instruction = _create_agent_instruction(specialist_name, core_task, extra_guidance)

    expected_base_parts = [
        f"Current date and time: {expected_datetime_str}",
        f"You are the {specialist_name} specialist.",
        "You receive specific sub-tasks from the Coordinator.",
        "**Your Task:**",
        core_task
    ]
    full_expected_parts = expected_base_parts + extra_guidance
    # Example: join(["A", "B\n", "C"]) -> "A\nB\n\nC"
    expected_instruction = "\n".join(full_expected_parts)
    
    assert instruction == expected_instruction
    assert f"{core_task}\nGuidance line 1.\n\nGuidance line 2." in instruction

print("Successfully created tests/test_instructions.py")
