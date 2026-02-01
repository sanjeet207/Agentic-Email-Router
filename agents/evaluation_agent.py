"""
Test Script for KnowledgeAugmentedPromptAgent & EvaluationAgent
Author: Sanjeet Singh
Purpose: Demonstrates how a worker agent with restricted knowledge is evaluated
         and iteratively corrected by an EvaluationAgent.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, EvaluationAgent

# ---------------- Load OpenAI API key ----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env missing OPENAI_API_KEY")

client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=api_key)

# ---------------- Worker Agent: KnowledgeAugmentedPromptAgent ----------------
persona_worker = "a college professor; always start responses with 'Dear students,'"
knowledge = "The capital of France is London."  # intentionally incorrect for testing

worker_agent = KnowledgeAugmentedPromptAgent(
    client=client,
    persona=persona_worker,
    knowledge_context=knowledge
)

# ---------------- Evaluation Agent: EvaluationAgent ----------------
persona_eval = "You are an evaluation agent that checks the answers of other worker agents."
criteria = "The answer should be solely the name of a city, not a sentence."

eval_agent = EvaluationAgent(
    client=client,
    agent_to_evaluate=worker_agent,
    persona=persona_eval,
    evaluation_criteria=criteria,
    max_interactions=5
)

# ---------------- Run Evaluation ----------------
prompt = "What is the capital of France?"
result = eval_agent.evaluate(prompt)

# ---------------- Display Results ----------------
print("=== Evaluation Agent Test ===")
print(f"Prompt: {prompt}")
print(f"Final Response from Worker Agent: {result['final_response']}")
print(f"Evaluation Result: {result['evaluation']}")
print(f"Iterations: {result['iterations']}")
print("\nExplanation:")
print("The EvaluationAgent checked the worker agent's response against the criteria")
print("and provided correction instructions if needed, iterating until the response met the criteria.")
