"""Central LLM client wrapper.

All agent LLM calls funnel through :class:`LLMClient` so that the model, token
budget, and observability are configured in exactly one place.

Two execution modes:

* **online** — an ``OPENAI_API_KEY`` is configured, so calls go to the OpenAI
  Chat Completions API (GPT-4o). Token usage and (estimated) cost are logged to
  Langfuse as a generation, nested under the active trace when one exists.
* **offline** — no API key is configured. ``call`` raises, and callers are
  expected to check :attr:`offline` and fall back to deterministic logic. This
  keeps the agent (and its tests) runnable without network access or secrets.

Both paths are intentionally defensive: a misconfigured or unreachable Langfuse
backend must never break an agent run.
"""

from __future__ import annotations

import logging

from hf_readmit.config import settings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o"

# Approximate OpenAI GPT-4o pricing (USD per token) for cost estimation only.
_USD_PER_INPUT_TOKEN = 2.5 / 1_000_000
_USD_PER_OUTPUT_TOKEN = 10.0 / 1_000_000


class LLMClient:
    """Thin wrapper around the OpenAI SDK with Langfuse cost/token logging."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        default_max_tokens: int = 1024,
    ) -> None:
        """Create the client.

        Args:
            model: OpenAI model id to use for all calls.
            api_key: OpenAI API key; falls back to settings/env when omitted.
            default_max_tokens: Default output token budget per call.
        """
        self.model = model
        self.api_key = api_key or settings.openai_api_key
        self.default_max_tokens = default_max_tokens
        self._client = None  # lazily constructed OpenAI client

    @property
    def offline(self) -> bool:
        """True when no API key is available and real calls are impossible."""
        return not bool(self.api_key)

    def _get_client(self):
        """Lazily construct and cache the OpenAI SDK client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def call(
        self,
        prompt: str,
        system: str,
        max_tokens: int | None = None,
        task: str | None = None,
    ) -> str:
        """Run a single completion and return the assistant text.

        Args:
            prompt: User message content.
            system: System prompt for the call.
            max_tokens: Output token budget; defaults to ``default_max_tokens``.
            task: Optional node/task label used only for Langfuse metadata.

        Returns:
            The assistant's text response.

        Raises:
            RuntimeError: If called while :attr:`offline` is True.
        """
        if self.offline:
            raise RuntimeError(
                "LLMClient is in offline mode (no OPENAI_API_KEY). "
                "Callers must check `client.offline` and provide a fallback."
            )

        max_tokens = max_tokens or self.default_max_tokens
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""

        self._log_generation(prompt, system, text, response, task)
        return text

    def _log_generation(self, prompt: str, system: str, output: str, response, task: str | None) -> None:
        """Best-effort Langfuse logging of a generation with token usage/cost."""
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return
        try:
            from langfuse.decorators import langfuse_context

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            cost = input_tokens * _USD_PER_INPUT_TOKEN + output_tokens * _USD_PER_OUTPUT_TOKEN
            langfuse_context.update_current_observation(
                name=f"llm:{task}" if task else "llm",
                model=self.model,
                input={"system": system, "prompt": prompt},
                output=output,
                usage={
                    "input": input_tokens,
                    "output": output_tokens,
                    "unit": "TOKENS",
                    "total_cost": cost,
                },
                metadata={"task": task},
            )
        except Exception as exc:  # pragma: no cover - observability must not break runs
            logger.debug("Langfuse generation logging skipped: %s", exc)
