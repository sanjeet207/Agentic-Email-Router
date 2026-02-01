# ---------------- knowledge_augmented_prompt_agent_demo.py ----------------
"""
Demonstration of KnowledgeAugmentedPromptAgent
Author: Sanjeet Singh
Purpose: Shows how the agent answers using only the supplied knowledge context
         and follows a specified persona.
"""

from openai import OpenAI
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# ---------------- Load environment variables ----------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment.")

# ---------------- Initialize OpenAI client ----------------
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=openai_api_key
)

# ---------------- Define persona and knowledge ----------------
persona = "You are a college professor; your answer always starts with: Dear students,"

knowledge_text = """
In the historic city of Boston, Clara, a marine biologist and science communicator, began each morning analyzing sonar data to track whale migration patterns along the Atlantic coast.
She spent her afternoons in a university lab, researching CRISPR-based gene editing to restore coral reefs damaged by ocean acidification and warming.

Clara created a podcast called "Crosscurrents", a show that explored the intersection of science, culture, and ethics.
Each week, she interviewed researchers, engineers, artists, and activists—from marine ecologists and AI ethicists to digital archivists preserving endangered languages.
""".strip()

# ---------------- Instantiate the KnowledgeAugmentedPromptAgent ----------------
knowledge_agent = KnowledgeAugmentedPromptAgent(
    client=client,
    persona=persona,
    knowledge_context=knowledge_text
)

# ---------------- Define prompt ----------------
prompt = "What is the podcast that Clara hosts about?"
print("Prompt:", prompt)

# ---------------- Get response ----------------
response = knowledge_agent.respond(prompt)

# ---------------- Print the agent's response ----------------
print("\nAgent Response:")
print(response)

# ---------------- Explanation ----------------
print("\nExplanation:")
print("The agent uses the supplied knowledge (about Clara and her podcast) to answer the prompt.")
print("It also follows the persona, starting responses with 'Dear students,' as specified.")
