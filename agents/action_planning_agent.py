"""
Test Script for ActionPlanningAgent
Author: Sanjeet Singh
Purpose: Demonstrates the extraction of actionable steps from a high-level prompt using the ActionPlanningAgent.
"""

from openai import OpenAI
from workflow_agents.base_agents import ActionPlanningAgent
import os
from dotenv import load_dotenv

# ---------------- Load OpenAI API key ----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env missing OPENAI_API_KEY")

# ---------------- Initialize OpenAI client ----------------
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=api_key
)

# ---------------- Knowledge for the agent ----------------
knowledge = """
# Fried Egg
1. Heat pan with oil or butter
2. Crack egg into pan
3. Cook until white is set
4. Season with salt and pepper
5. Serve

# Scrambled Eggs
1. Crack eggs into a bowl
2. Beat eggs
3. Heat pan
4. Pour eggs into pan
5. Stir until just set
6. Season and serve
"""

# ---------------- Instantiate the Action Planning Agent ----------------
agent = ActionPlanningAgent(client=client, knowledge_context=knowledge)

# ---------------- Define a prompt ----------------
prompt = "One morning I wanted to have scrambled eggs"

# ---------------- Extract actionable steps ----------------
steps = agent.extract_steps_from_prompt(prompt)

# ---------------- Print results ----------------
print("=== Action Planning Agent Test ===")
print(f"Prompt: {prompt}")
print("\nExtracted Steps:")
for i, step in enumerate(steps, 1):
    print(f"{i}. {step}")
