"""Utility helpers for interacting with OpenAI Assistants (web-enabled)."""

import json
import time
from typing import Dict, Callable, Any, List, Optional

import streamlit as st

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - environment specific
    raise RuntimeError(
        "openai package is required for assistant_helper. Install openai>=1.0"
    ) from exc


DEFAULT_ASSISTANT_KEY = "wealth_master_gpt5"
DEFAULT_ASSISTANT_MODEL_PRIMARY = "gpt-4o"
DEFAULT_ASSISTANT_MODEL_FALLBACK = "gpt-4o-mini"  # gpt-4o-mini as fallback
DEFAULT_ASSISTANT_INSTRUCTIONS = (
    "You are the Wealth Manager AI core assistant."
    " Use Moneycontrol, NSE/BSE knowledge, AMFI data, and all provided context"
    " (holdings, transactions, historical prices, PDFs, news) to answer precisely."
    " Always cite tickers, dates, and numbers when available."
    " If live market data is needed, prefer Moneycontrol scraping results provided"
    " by the application; when unavailable, make the best estimate using recent"
    " context and clearly label it as an estimate."
)


class AssistantRunner:
    """Creates and runs an OpenAI Assistant."""

    def __init__(
        self,
        assistant_key: str,
        instructions: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        auto_create: bool = True,
    ) -> None:
        self._assistant_id_key = f"assistant_id__{assistant_key}"
        self._client = OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
        self._assistant_id: Optional[str] = st.session_state.get(self._assistant_id_key)
        self._model = model or DEFAULT_ASSISTANT_MODEL_PRIMARY
        self._instructions = instructions
        self._tools = tools or []

        if auto_create and not self._assistant_id:
            last_error: Optional[Exception] = None
            preferred_models = [self._model]
            if DEFAULT_ASSISTANT_MODEL_FALLBACK not in preferred_models:
                preferred_models.append(DEFAULT_ASSISTANT_MODEL_FALLBACK)

            for candidate_model in preferred_models:
                try:
                    assistant = self._client.beta.assistants.create(
                        model=candidate_model,
                        instructions=self._instructions,
                        tools=self._tools,
                    )
                except Exception as exc:  # pragma: no cover - depends on OpenAI response
                    last_error = exc
                    continue
                else:
                    self._assistant_id = assistant.id
                    self._model = candidate_model
                    st.session_state[self._assistant_id_key] = self._assistant_id
                    last_error = None
                    break

            if not self._assistant_id:
                # Re-raise the last error to make failure explicit
                raise last_error  # type: ignore[misc]

    @property
    def assistant_id(self) -> str:
        if not self._assistant_id:
            raise RuntimeError("Assistant was not initialised.")
        return self._assistant_id

    def run(
        self,
        user_message: str,
        function_map: Optional[Dict[str, Callable[..., Any]]] = None,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Run assistant conversation and return final assistant text."""

        function_map = function_map or {}

        thread = self._client.beta.threads.create()

        if extra_messages:
            for message in extra_messages:
                self._client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role=message.get("role", "user"),
                    content=message.get("content", ""),
                )

        self._client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message,
        )

        run = self._client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant_id,
        )

        while True:
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for call in tool_calls:
                    func_name = call.function.name
                    func_args = json.loads(call.function.arguments or "{}")
                    handler = function_map.get(func_name)

                    if handler is None:
                        output = json.dumps({"error": f"Unknown function: {func_name}"})
                    else:
                        try:
                            result = handler(**func_args)
                        except Exception as exc:  # pragma: no cover - runtime safety
                            output = json.dumps({"error": str(exc)})
                        else:
                            output = result if isinstance(result, str) else json.dumps(result)

                    tool_outputs.append({
                        "tool_call_id": call.id,
                        "output": output,
                    })

                run = self._client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs,
                )
                continue

            if run.status in {"completed", "failed", "cancelled"}:
                break

            time.sleep(1.5)
            run = self._client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )

        if run.status != "completed":
            raise RuntimeError(f"Assistant run failed: {run.status}")

        messages = self._client.beta.threads.messages.list(thread_id=thread.id)
        response_chunks: List[str] = []

        for message in messages.data:
            if message.role != "assistant":
                continue
            for part in message.content:
                if part.type == "text":
                    response_chunks.append(part.text.value)

        return "\n".join(response_chunks).strip()


def get_shared_assistant(tools: Optional[List[Dict[str, Any]]] = None) -> AssistantRunner:
    """Return the shared Wealth Manager assistant instance (cached per session).

    Tools are part of the assistant signature; different tool sets create unique assistants
    while keeping the same base instructions/model.
    """
    tools = tools or []
    tools_key = json.dumps(tools, sort_keys=True)
    session_key = f"_shared_assistant_runner::{tools_key}"
    runner = st.session_state.get(session_key)
    if runner is None:
        runner = AssistantRunner(
            assistant_key=f"{DEFAULT_ASSISTANT_KEY}::{hash(tools_key)}",
            instructions=DEFAULT_ASSISTANT_INSTRUCTIONS,
            tools=tools,
        )
        st.session_state[session_key] = runner
    return runner


def run_gpt5_completion(
    messages: List[Dict[str, str]],
    *,
    model: str = DEFAULT_ASSISTANT_MODEL_PRIMARY,
    fallback_model: Optional[str] = DEFAULT_ASSISTANT_MODEL_FALLBACK,
    **kwargs: Any,
) -> str:
    """Call the Chat Completions API (GPT-5 preferred) with automatic fallback.

    Args:
        messages: Sequence of chat messages (role/content dictionaries).
        model: Primary model to request (defaults to GPT-5).
        fallback_model: Secondary model if the primary is unsupported or fails.
        **kwargs: Extra keyword arguments passed to ``chat.completions.create``.

    Returns:
        The assistant's response content as a string.

    Raises:
        Exception: When both the primary and fallback models fail.
    """

    client = OpenAI(api_key=st.secrets["api_keys"]["open_ai"])
    last_error: Optional[Exception] = None

    candidate_models = [model]
    if fallback_model and fallback_model not in candidate_models:
        candidate_models.append(fallback_model)

    for candidate in candidate_models:
        try:
            response = client.chat.completions.create(
                model=candidate,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on OpenAI runtime
            last_error = exc
            continue

        choice = response.choices[0]
        if choice.message and choice.message.content:
            return choice.message.content
        last_error = RuntimeError("Chat completion returned empty content")

    if last_error:
        raise last_error
    raise RuntimeError("Chat completion failed without an explicit error")
