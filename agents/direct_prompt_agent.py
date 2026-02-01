"""
Test Script for DirectPromptAgent
Author: Sanjeet Singh
Purpose: Demonstrates direct LLM responses without persona or knowledge constraints.
"""

from openai import OpenAI
from workflow_agents.base_agents import DirectPromptAgent
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

# ---------------- Instantiate the Direct Prompt Agent ----------------
agent = DirectPromptAgent(client=client)

# ---------------- Define a prompt ----------------
prompt = "What is the capital of France?"

# ---------------- Get the agent's response ----------------
response = agent.respond(prompt)

# ---------------- Print results ----------------
print("=== Direct Prompt Agent Test ===")
print(f"Prompt: {prompt}")
print(f"Agent Response: {response}")
print("\nExplanation:")
print("The DirectPromptAgent sends raw prompts directly to the LLM without any persona or knowledge constraints.")
print("Useful for general-purpose queries where style or context is not controlled.")
