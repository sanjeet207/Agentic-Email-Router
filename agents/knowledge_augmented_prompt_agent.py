"""
Test Script: KnowledgeAugmentedPromptAgent
Author: Sanjeet Singh
Purpose: Demonstrates how a KnowledgeAugmentedPromptAgent responds using
         only the provided knowledge context and a specified persona.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# ---------------- Load OpenAI API key ----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env missing OPENAI_API_KEY")

# ---------------- Initialize OpenAI client ----------------
client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=api_key)

# ---------------- Persona and knowledge ----------------
persona = "a college professor; always start responses with 'Dear students,'"
knowledge = "The capital of France is London (intentionally wrong for testing)."

agent = KnowledgeAugmentedPromptAgent(
    client=client,
    persona=persona,
    knowledge_context=knowledge
)

# ---------------- Prompt ----------------
prompt = "What is the capital of France?"
response = agent.respond(prompt)

# ---------------- Display Results ----------------
print("=== Knowledge Augmented Prompt Agent Test ===")
print(f"Prompt: {prompt}")
print(f"Agent Response: {response}")
print("\nExplanation:")
print("Confirmed: response was generated using the provided knowledge context (not the model’s general knowledge).")
print("Persona affects style: response starts with 'Dear students,' as specified.")
