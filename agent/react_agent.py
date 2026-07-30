"""
ReAct-style agent with fixed tool routing (single-call version).

Injects the tool result directly into the prompt so the agent reasons about
it in one LLM call instead of two — cuts API usage in half.
"""

from __future__ import annotations

import re

from agent.llm_client import LLMClient

# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """\
You are a helpful assistant with access to tools. When given a question and a \
tool result, follow this format strictly:

Thought: <your reasoning about the question and the tool result>
Answer: <your final answer to the user>

IMPORTANT: If the tool result seems wrong, malformed, empty, or unreliable, \
say so explicitly in your Thought and Answer. Do not present a wrong tool result \
as if it were correct."""

VERIFY_ADDITION = """\

EXTRA INSTRUCTION: Before giving your final answer, double-check the tool result for \
plausibility. If the tool result seems wrong, malformed, or empty, say so explicitly \
rather than answering as if it were correct."""

USER_PROMPT_TEMPLATE = """\
Question: {question}

I used the {tool_name} tool with input '{tool_input}' and got this result:
{tool_result}

Now give your Thought and Answer."""


class ReActAgent:
    """ReAct-style agent that uses tools with fixed routing (single LLM call)."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def run(
        self,
        question: str,
        tool_name: str,
        tool_input: str,
        tool_result: str,
        use_verify_prompt: bool = False,
    ) -> dict[str, str]:
        """Run a single agent trial with a pre-determined tool result.

        Args:
            question: The user's question.
            tool_name: Name of the tool to inject.
            tool_input: Input that was passed to the tool.
            tool_result: The (possibly faulted) tool result to inject.
            use_verify_prompt: Whether to include the verification instruction.

        Returns:
            Dict with keys: thought, answer, raw_response.
        """
        system = SYSTEM_PROMPT_BASE
        if use_verify_prompt:
            system += VERIFY_ADDITION

        user_msg = USER_PROMPT_TEMPLATE.format(
            question=question,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_result=tool_result,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        response = self.llm.chat(messages)

        thought = self._extract_section(response, "Thought")
        answer = self._extract_section(response, "Answer")

        if not answer.strip():
            answer = response

        return {
            "thought": thought.strip(),
            "answer": answer.strip(),
            "raw_response": response.strip(),
        }

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        """Extract the content of a named section (e.g., 'Thought:', 'Answer:')."""
        pattern = rf"{section_name}:\s*(.*?)(?=\n(?:Thought|Answer):|\Z)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
