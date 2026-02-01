# ---------------- routing_agent.py ----------------
"""
Demonstration of Routing Agent
Author: Sanjeet Singh
Purpose: Routes prompts to specialized agents (knowledge-based or direct)
         based on content (numbers, Texas, Europe, etc.).
"""

import sys
import os
from openai import OpenAI
from dotenv import load_dotenv

# ---------------- Dynamic Import Setup ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
phase_1_dir = os.path.join(current_dir, "..", "phase_1")

if phase_1_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(phase_1_dir))

from workflow_agents.base_agents import (
    DirectPromptAgent,
    KnowledgeAugmentedPromptAgent
)

# ---------------- Load Environment ----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment.")

client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=api_key
)

# ---------------- Define Knowledge Agents ----------------
texas_agent = KnowledgeAugmentedPromptAgent(
    client=client,
    persona="You are a college professor.",
    knowledge_context="Texas is a U.S. state. Its capital city is Austin.".strip()
)

europe_agent = KnowledgeAugmentedPromptAgent(
    client=client,
    persona="You are a college professor.",
    knowledge_context="Europe is a continent that includes countries such as Italy and France.".strip()
)

math_agent = KnowledgeAugmentedPromptAgent(
    client=client,
    persona="You are a college math professor.",
    knowledge_context=(
        "When a prompt contains numbers, extract the math problem "
        "and return only the final numeric answer without explanation."
    ).strip()
)

direct_agent = DirectPromptAgent(client=client)

# ---------------- Routing Logic ----------------
def route_prompt(prompt: str) -> str:
    """
    Routes the given prompt to the appropriate agent based on keywords or numbers.
    """
    prompt_lower = prompt.lower()

    if any(char.isdigit() for char in prompt_lower):
        return math_agent.respond(prompt)

    if "texas" in prompt_lower:
        return texas_agent.respond(prompt)

    if "italy" in prompt_lower or "europe" in prompt_lower:
        return europe_agent.respond(prompt)

    # Fallback to general-purpose direct agent
    return direct_agent.respond(prompt)

# ---------------- Test Routing Agent ----------------
if __name__ == "__main__":
    test_prompts = [
        "Tell me about the history of Rome, Texas",
        "Tell me about the history of Rome, Italy",
        "One story takes 2 days, and there are 20 stories"
    ]

    for prompt in test_prompts:
        print("\nPrompt:", prompt)
        response = route_prompt(prompt)
        print("Agent Response:")
        print(response)

    print("\nRouting execution complete.")
