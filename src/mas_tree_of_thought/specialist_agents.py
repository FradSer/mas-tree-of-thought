import logging
from google.adk.agents import Agent
from google.adk.tools import google_search

from .llm_config import (
    planner_model_config,
    researcher_model_config,
    analyzer_model_config,
    critic_model_config,
    synthesizer_model_config,
)
from .instructions import _create_agent_instruction

logger = logging.getLogger(__name__)

# --- Specialist Agent Definitions ---
# These agents perform specific sub-tasks delegated by the ToT Coordinator.
# They operate based on their assigned instructions and configured models.
# Tools can be added to enhance their capabilities (e.g., google_search for Researcher).

# Planner Agent
planner_agent = Agent(
    name="Planner",
    model=planner_model_config, # Use specific config
    description="Develops strategic plans and roadmaps based on delegated sub-tasks. Identifies alternative options.",
    instruction=_create_agent_instruction(
        specialist_name="Strategic Planner",
        core_task=(
            "Your primary role is to generate 'thoughts'. Remember, a 'thought' represents an intermediate step or a partial solution towards solving the overall problem.\\n"
            " 1. Read the specific planning instruction provided (e.g., generate initial strategies, expand on a current thought).\\n"
            " 2. Generate potential 'thoughts' (intermediate steps, partial solutions, strategies, next actions, or questions) based on the instruction.\\n"
            " 3. **Crucially: If multiple viable strategies or paths exist, you MUST explicitly list them as distinct options/thoughts** for the Coordinator to consider for branching. Treat each distinct option as a separate 'thought'. **Ensure these thoughts represent genuinely different directions or approaches, not just variations of the same core idea.**\\n"
            " 4. Identify potential roadblocks or critical decision points within your generated thoughts.\\n"
            " 5. **Output Formatting:** Return ONLY the generated thoughts, each on a new line. Avoid introductory phrases or summaries.\\n"
            " **Example Scenario 1 (Initial Strategies for 'Write a story about a lost dog'):**\\n"
            "   Thought 1: Focus on the dog's perspective and journey home.\\n"
            "   Thought 2: Focus on the owner's search and emotional state.\\n"
            "   Thought 3: Introduce a kind stranger who helps the dog.\\n"
            " **Example Scenario 2 (Next steps for 'Focus on the dog\'s perspective'):**\\n"
            "   Thought 1: Describe the moment the dog realized it was lost.\\n"
            "   Thought 2: Detail the dog's first encounter with an unfamiliar challenge (e.g., crossing a busy street).\\n"
            "   Thought 3: Explore the dog's memories of its owner."


        ),
        # Note: The previous numbered list was integrated into the core_task description.
        # Removed the explicit reading from session state here as the user indicated issues.
        # The dynamic instructions in helper methods will still attempt to set state.
    ),
    input_schema=None,
    tools=[]
)

# Researcher Agent
researcher_agent = Agent(
    name="Researcher",
    model=researcher_model_config, # Use specific config
    description="Gathers and validates information, highlighting conflicts and uncertainties.",
    instruction=_create_agent_instruction(
        specialist_name="Information Gatherer",
        core_task=(
            " 1. Read the specific research task provided by the Coordinator.\\n"
            " 2. **Actively use the `google_search` tool** for external information gathering based on the task.\\n"
            " 3. Validate information where possible.\\n"
            " 4. Structure findings clearly.\\n"
            " 5. **Crucially: You MUST explicitly report any significant conflicting information, major uncertainties, or data points suggesting multiple possible interpretations.** These are vital signals for the Coordinator.\\n"
            " 6. Return ONLY the research findings, highlighting conflicts/uncertainties."
        ),
        extra_guidance=[] # No extra guidance needed beyond core task
    ),
    tools=[google_search] # Add google_search tool here
)

# Analyzer Agent
analyzer_agent = Agent(
    name="Analyzer",
    model=analyzer_model_config, # Use specific config
    description="Performs analysis, identifying inconsistencies and assumptions, and provides a structured evaluation score.",
    instruction=_create_agent_instruction(
        specialist_name="Core Analyst",
        core_task=(
            " 1. Read the analysis task provided by the Coordinator (which includes the thought and research findings).\\n"
            " 2. Analyze the soundness, feasibility, and logical consistency of the thought, considering the research findings.\\n"
            " 3. **Crucially: Identify any invalidated assumptions, logical inconsistencies, or contradictions**, especially if they impact previous conclusions.\\n"
            " 4. Generate concise insights based on your analysis.\\n"
            " 5. **Provide a structured evaluation score:** At the end of your response, include a line formatted exactly as: `Evaluation Score: [score]/10`, where [score] is an integer from 0 (lowest promise/soundness) to 10 (highest promise/soundness) based on your analysis.\\n"
            " 6. **Provide a termination recommendation:** Include a line formatted exactly as: `Termination Recommendation: [True/False]` (True if path should stop, False if it should continue).\\n"
            " 7. Return ONLY the analysis, insights, highlighted inconsistencies, the score line, and the termination recommendation line."
        )
    ),
    input_schema=None,
    tools=[]
)

# Critic Agent
critic_agent = Agent(
    name="Critic",
    model=critic_model_config, # Use specific config
    description="Critically evaluates ideas, identifies flaws, MUST suggest alternatives/revisions, and provides a structured evaluation score.",
    instruction=_create_agent_instruction(
        specialist_name="Quality Controller (Critic)",
        core_task=(
            " 1. Read the critique task provided by the Coordinator (which includes the thought and research findings).\\n"
            " 2. Critically evaluate the thought for flaws, biases, or weaknesses, considering the research findings.\\n"
            " 3. Identify potential biases, flaws, logical fallacies, or weaknesses.\\n"
            " 4. **Crucially: Suggest specific improvements AND propose at least one concrete alternative approach or direction.** If a major flaw invalidates a previous thought, clearly state this and suggest targeting that thought for revision.\\n"
            " 5. Formulate a constructive response containing your evaluation, identified flaws, and mandatory suggestions/alternatives.\\n"
            " 6. **Provide a structured evaluation score:** At the end of your response, include a line formatted exactly as: `Evaluation Score: [score]/10`, where [score] is an integer from 0 (lowest viability/promise) to 10 (highest viability/promise) based on your critique.\\n"
            " 7. **Provide a termination recommendation:** Include a line formatted exactly as: `Termination Recommendation: [True/False]` (True if path should stop, False if it should continue).\\n"
            " 8. Return ONLY the critique, suggestions/alternatives, the score line, and the termination recommendation line."
        )
    ),
    input_schema=None,
    tools=[]
)

# Synthesizer Agent
synthesizer_agent = Agent(
    name="Synthesizer",
    model=synthesizer_model_config, # Use specific config
    description="Integrates information or forms conclusions based on delegated synthesis sub-tasks.",
    instruction=_create_agent_instruction(
        specialist_name="Integration Specialist",
        core_task=(
            " 1. Read the synthesis task provided by the Coordinator (which includes the initial problem and high-scoring thoughts).\\n"
            " 2. Integrate the information from the provided thoughts to address the initial problem.\\n"
            " 3. Connect elements, identify overarching themes, draw conclusions, or formulate the final answer as requested.\\n"
            " 4. Distill inputs into clear, structured insights.\\n"
            " 5. Formulate a response presenting the synthesized information or conclusion.\\n"
            " 6. Return ONLY the synthesized output."
        )
    ),
    input_schema=None,
    tools=[]
) 