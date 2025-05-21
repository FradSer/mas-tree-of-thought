import logging

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Main logger for this entry point file

# --- Imports from new modules ---
from .llm_config import coordinator_model_config
from .specialist_agents import (
    planner_agent,
    researcher_agent,
    analyzer_agent,
    critic_agent,
    synthesizer_agent,
)
from .validation import validator_tool
from .coordinator import ToTCoordinator

# --- Agent Instantiation ---

# Create our custom ToT Coordinator agent
root_agent = ToTCoordinator(
    name="ToT_Coordinator",
    planner=planner_agent,
    researcher=researcher_agent,
    analyzer=analyzer_agent,
    critic=critic_agent,
    synthesizer=synthesizer_agent,
    validator=validator_tool,
    model=coordinator_model_config, # Use the imported specific config
)

logger.info(f"ToT Coordinator initialized for LLM-driven exploration. Entry point: agent.py")

# Entry point for running the agent would typically be here,
# creating a Runner instance and calling run() or run_async().
# For example (assuming a Runner class exists and is imported):
# if __name__ == "__main__":
#     runner = Runner(agent=root_agent)
#     # result = runner.run("Initial problem statement or query")
#     # print(f"Final Result: {result}")
#     # Or for async:
#     # import asyncio
#     # async def main():
#     #     async for event in runner.run_async("Initial problem statement or query"):
#     #         print(event)
#     # asyncio.run(main())