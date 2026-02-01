"""
Agentic Workflow Agents - Base Module

This module defines various AI agents used for an agentic workflow:
- DirectPromptAgent: sends raw prompts to the LLM
- AugmentedPromptAgent: uses persona prompts
- KnowledgeAugmentedPromptAgent: uses only provided knowledge
- EvaluationAgent: evaluates and iteratively corrects other agents
- RoutingAgent: routes prompts to the most relevant agent using embeddings
- ActionPlanningAgent: extracts actionable steps from high-level prompts

Author: Sanjeet Singh
"""

import numpy as np
from openai import OpenAI
from typing import List, Callable, Dict


# =====================================================
# Direct Prompt Agent
# =====================================================
class DirectPromptAgent:
    """
    Sends a raw user prompt directly to the LLM and returns the response.
    Ideal for general-purpose queries without persona or knowledge constraints.
    """

    def __init__(self, client: OpenAI):
        self.client = client

    def respond(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()


# =====================================================
# Augmented Prompt Agent
# =====================================================
class AugmentedPromptAgent:
    """
    Uses a persona system prompt but no external knowledge.
    Guides the model to respond in the style of the given persona.
    """

    def __init__(self, client: OpenAI, persona: str):
        self.client = client
        self.persona = persona

    def respond(self, prompt: str) -> str:
        system_prompt = (
            f"You are {self.persona}.\n"
            "Forget all previous context."
        )

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()


# =====================================================
# Knowledge Augmented Prompt Agent
# =====================================================
class KnowledgeAugmentedPromptAgent:
    """
    Responds to prompts using only the provided knowledge context.
    Persona guides response style.
    """

    def __init__(self, client: OpenAI, persona: str, knowledge_context: str):
        self.client = client
        self.persona = persona
        self.knowledge_context = knowledge_context

    def respond(self, prompt: str) -> str:
        system_prompt = (
            f"You are {self.persona}.\n"
            "Use ONLY the knowledge below.\n"
            "Do NOT use outside knowledge.\n\n"
            f"KNOWLEDGE:\n{self.knowledge_context}"
        )

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()


# =====================================================
# Evaluation Agent
# =====================================================
class EvaluationAgent:
    """
    Evaluates another agent's output and iteratively applies corrections
    until it meets the evaluation criteria or max interactions are reached.
    """

    def __init__(
        self,
        client: OpenAI,
        persona: str,
        evaluation_criteria: str,
        agent_to_evaluate,
        max_interactions: int = 3
    ):
        self.client = client
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions

    def evaluate(self, task: str) -> dict:
        prompt = task
        final_response = None
        evaluation_result = None

        for i in range(self.max_interactions):
            agent_output = self.agent_to_evaluate.respond(prompt)

            evaluation_prompt = (
                f"{self.persona}\n\n"
                f"Evaluate the following response:\n{agent_output}\n\n"
                f"Against these criteria:\n{self.evaluation_criteria}\n\n"
                "Respond with Yes or No, followed by a brief explanation."
            )

            evaluation_result = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0
            ).choices[0].message.content.strip()

            final_response = agent_output

            if evaluation_result.lower().startswith("yes"):
                break

            correction_prompt = (
                f"The response failed evaluation.\n\n"
                f"Response:\n{agent_output}\n\n"
                f"Feedback:\n{evaluation_result}\n\n"
                "Provide clear correction instructions."
            )

            correction_instructions = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=0
            ).choices[0].message.content.strip()

            prompt = f"{task}\n\nApply these corrections:\n{correction_instructions}"

        return {
            "final_response": final_response,
            "evaluation": evaluation_result,
            "iterations": i + 1
        }


# =====================================================
# Routing Agent
# =====================================================
class RoutingAgent:
    """
    Routes input to the most relevant agent using vector embeddings
    and cosine similarity.
    """

    def __init__(self, client: OpenAI, agents: List[Dict[str, Callable]]):
        self.client = client
        self.agents = agents

    def _embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding

    def route(self, prompt: str) -> str:
        prompt_embedding = self._embed(prompt)

        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_embedding = self._embed(agent["description"])
            similarity = np.dot(prompt_embedding, agent_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(agent_embedding)
            )

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        return best_agent["func"](prompt)


# =====================================================
# Action Planning Agent
# =====================================================
class ActionPlanningAgent:
    """
    Extracts a numbered list of actionable steps from a high-level goal prompt.
    """

    def __init__(self, client: OpenAI, knowledge_context: str):
        self.client = client
        self.knowledge_context = knowledge_context

    def extract_steps_from_prompt(self, prompt: str) -> List[str]:
        system_prompt = (
            "You are an action planning agent.\n"
            "Extract a numbered list of actionable steps.\n"
            "Use ONLY the provided knowledge.\n\n"
            f"KNOWLEDGE:\n{self.knowledge_context}"
        )

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        lines = response.choices[0].message.content.split("\n")
        return [line.strip() for line in lines if line.strip()]
