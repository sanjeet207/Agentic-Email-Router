"""
Test Script for AugmentedPromptAgent
Author: Sanjeet Singh
Purpose: Demonstrates persona-guided responses using the AugmentedPromptAgent.
"""

from openai import OpenAI
from workflow_agents.base_agents import AugmentedPromptAgent
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

# ---------------- Instantiate the Augmented Prompt Agent ----------------
persona = "a helpful assistant; always answer politely"
agent = AugmentedPromptAgent(client=client, persona=persona)

# ---------------- Define a prompt ----------------
prompt = "What is the capital of France?"

# ---------------- Get the agent's response ----------------
response = agent.respond(prompt)

# ---------------- Print results ----------------
print("=== Augmented Prompt Agent Test ===")
print(f"Prompt: {prompt}")
print(f"Agent Response: {response}")
print("\nExplanation:")
print("This agent follows the provided persona to guide response style.")
print("Persona ensures that the tone and phrasing of the response is consistent with instructions.")
