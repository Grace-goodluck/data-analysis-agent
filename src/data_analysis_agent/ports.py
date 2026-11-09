"""The one outside service the agent depends on, as an interface.

The real implementation lives in gemini_client.py; tests pass a fake,
so no test touches the network or Gemini.
"""

from typing import Protocol


class LLMError(Exception):
    """A Gemini call failed (network, quota, empty reply...)."""


class LLM(Protocol):
    model: str

    def generate(self, prompt: str) -> str:
        """Return the model's reply text, or raise LLMError."""
        ...
