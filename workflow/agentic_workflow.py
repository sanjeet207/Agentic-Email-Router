"""
Agentic Workflow - Multi-Agent Project Planner
Author: Sanjeet Singh
Description:
This script demonstrates a multi-agent AI workflow using OpenAI GPT-based agents.
It includes:
- Action Planning Agent: extracts actionable steps from a high-level goal
- Knowledge-Augmented Prompt Agents: generate outputs based on persona and domain knowledge
- Evaluation Agents: validate and iteratively correct agent outputs
- Routing Agent: assigns workflow steps to the most relevant agent

Technologies: Python, OpenAI API, dotenv, NumPy
Purpose: Portfolio-ready demonstration of agent orchestration and AI-driven workflow automation.
"""

import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- Load environment variables ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment.")

# ---------------- Add Phase 1 path for agent imports ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
phase_1_path = os.path.normpath(os.path.join(current_dir, "../phase_1"))
sys.path.insert(0, phase_1_path)

# ---------------- Import agent classes ----------------
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent
)

# ---------------- Initialize OpenAI client ----------------
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=OPENAI_API_KEY
)

# ---------------- Load Product Specification ----------------
spec_path = os.path.join(current_dir, "Product-Spec-Email-Router.txt")
with open(spec_path, "r") as f:
    product_spec = f.read()

# ---------------- Action Planning Agent ----------------
knowledge_action_planning = (
    "You are a project planning AI. Extract clear, actionable workflow steps "
    "from a high-level project prompt."
)

action_planner = ActionPlanningAgent(
    client=client,
    knowledge_context=knowledge_action_planning
)

workflow_prompt = "Generate a complete project plan for building the Email Router."
workflow_steps = action_planner.extract_steps_from_prompt(workflow_prompt)

# ---------------- Knowledge-Augmented Agents ----------------
def create_knowledge_agent(persona: str, knowledge_text: str) -> KnowledgeAugmentedPromptAgent:
    return KnowledgeAugmentedPromptAgent(
        client=client,
        persona=persona,
        knowledge=knowledge_text
    )

# Product Manager Agent
persona_pm = "You are a Product Manager AI responsible for defining user stories."
knowledge_pm = "Create user stories from this product spec:\n" + product_spec
product_manager_agent = create_knowledge_agent(persona_pm, knowledge_pm)

# Program Manager Agent
persona_prog = "You are a Program Manager AI responsible for defining product features."
knowledge_prog = (
    "Define product features in this format:\n"
    "Feature Name: ...\nDescription: ...\nKey Functionality: ...\nUser Benefit: ...\n" + product_spec
)
program_manager_agent = create_knowledge_agent(persona_prog, knowledge_prog)

# Development Engineer Agent
persona_dev = "You are a Development Engineer AI responsible for detailed engineering tasks."
knowledge_dev = (
    "Define engineering tasks in this format:\n"
    "Task ID: ...\nTask Title: ...\nRelated User Story: ...\nDescription: ...\n"
    "Acceptance Criteria: ...\nEstimated Effort: ...\nDependencies: ...\n" + product_spec
)
development_engineer_agent = create_knowledge_agent(persona_dev, knowledge_dev)

# ---------------- Evaluation Agents ----------------
def create_eval_agent(agent, persona: str, criteria: str) -> EvaluationAgent:
    return EvaluationAgent(
        client=client,
        persona=persona,
        evaluation_criteria=criteria,
        agent_to_evaluate=agent
    )

# Evaluation for User Stories
pm_eval_criteria = (
    "The answer should be user stories strictly following the structure:\n"
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
pm_eval = create_eval_agent(
    product_manager_agent,
    "Evaluation agent checking user stories",
    pm_eval_criteria
)

# Evaluation for Features
program_eval_criteria = (
    "Each feature must strictly follow this structure:\n"
    "Feature Name: [Name]\nDescription: [Brief explanation]\n"
    "Key Functionality: [Capabilities]\nUser Benefit: [Value]"
)
program_eval = create_eval_agent(
    program_manager_agent,
    "Evaluation agent checking product features",
    program_eval_criteria
)

# Evaluation for Engineering Tasks
dev_eval_criteria = (
    "Each engineering task must strictly follow this structure:\n"
    "Task ID: [Unique ID]\nTask Title: [Brief description]\n"
    "Related User Story: [Reference]\nDescription: [Detailed explanation]\n"
    "Acceptance Criteria: [Completion criteria]\nEstimated Effort: [Time estimate]\n"
    "Dependencies: [Prerequisite tasks]"
)
dev_eval = create_eval_agent(
    development_engineer_agent,
    "Evaluation agent checking engineering tasks",
    dev_eval_criteria
)

# ---------------- Helper Functions ----------------
def evaluate_until_valid(step, eval_agent):
    """Evaluate a step until it passes validation criteria."""
    response = eval_agent.evaluate(step)["final_response"]
    while "reject" in response.lower() or "correction" in response.lower():
        response = eval_agent.evaluate(step)["final_response"]
    return response

# Mapping of agents for routing
agent_mapping = {
    "Product Manager": pm_eval,
    "Program Manager": program_eval,
    "Development Engineer": dev_eval
}

# Routing function
def route_step(step: str) -> dict:
    step_lower = step.lower()
    if "user story" in step_lower or "training" in step_lower:
        agent_name = "Product Manager"
    elif "feature" in step_lower or "integration" in step_lower or "architecture" in step_lower:
        agent_name = "Program Manager"
    else:
        agent_name = "Development Engineer"
    
    output_text = evaluate_until_valid(step, agent_mapping[agent_name])
    return {"agent": agent_name, "output": output_text}

# ---------------- Execute Workflow ----------------
workflow_results = []
for step in workflow_steps:
    result = route_step(step)
    workflow_results.append(result)

# ---------------- Save Results to JSON ----------------
output_file = os.path.join(current_dir, "phase_2_workflow_output.json")
with open(output_file, "w") as f:
    json.dump(workflow_results, f, indent=4)

# ---------------- Print Summary ----------------
print("\n✅ Phase 2 Agentic Workflow Completed")
print(f"Total Steps Processed: {len(workflow_results)}")
print(f"Workflow output saved to: {output_file}\n")
print("Sample Output:")
print(json.dumps(workflow_results[:3], indent=4))  # Show first 3 steps as preview
