import time
import hashlib
import threading
import logging
from typing import Dict, Tuple, Any
from models.client import ModelError

logger = logging.getLogger(__name__)

class ServerUnavailableError(ModelError):
    """The local vLLM endpoint remained unreachable after connection retries."""

class VLLMClient:
    """
    Client for talking to a vLLM server via the OpenAI API.
    Replaces the mock LLMClient with actual cluster generation capabilities.
    """
    def __init__(
        self, 
        model_name: str = "Qwen2.5-72B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_retries: int = 3,
        openai_max_retries: int = None,
        retry_delay: float = 2.0,
        timeout: float = 300.0,
        enable_cache: bool = True
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.openai_max_retries = openai_max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache: Dict[str, str] = {}
        self._thread_local = threading.local()

    def _get_cache_key(
        self, system_prompt: str, user_prompt: str, model: str,
        temperature: float, top_p: float, top_k: int,
        repetition_penalty: float, seed: int, max_tokens: int,
    ) -> str:
        key_content = (
            f"{system_prompt}|{user_prompt}|{model}|{temperature}|{top_p}|"
            f"{top_k}|{repetition_penalty}|{seed}|{max_tokens}"
        )
        return hashlib.sha256(key_content.encode('utf-8')).hexdigest()

    @property
    def client(self):
        # Each thread gets its own OpenAI client
        if not hasattr(self._thread_local, "openai_client"):
            from openai import OpenAI
            self._thread_local.openai_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=(
                    self.max_retries if self.openai_max_retries is None
                    else self.openai_max_retries
                ),
            )
        return self._thread_local.openai_client

    def generate(
        self, system_prompt, user_prompt: str, model: str = None,
        temperature: float = 0.0, max_tokens: int = 512,
        response_format: Dict[str, Any] = None, top_p: float = 1.0,
        top_k: int = 0, repetition_penalty: float = 1.05, seed: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        target_model = model or self.model_name
        metadata = {
            "model": target_model,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
            "max_tokens": max_tokens,
            "timestamp": time.time(),
        }

        cache_key = self._get_cache_key(
            str(system_prompt), user_prompt, target_model, temperature, top_p,
            top_k, repetition_penalty, seed, max_tokens,
        )
        if self.enable_cache and cache_key in self.cache:
            return self.cache[cache_key], metadata

        # Build messages — omit the system role entirely when no prompt is given.
        # This is the true no-system-prompt control condition.
        messages = []
        if system_prompt is not None and system_prompt != "":
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        last_error = None
        # enable_thinking=False is a Qwen3-specific chat_template kwarg that
        # suppresses chain-of-thought reasoning tokens. Only send it for true
        # Qwen3 variants (e.g. "Qwen3-", "QwQ-"); Qwen2.5 does NOT support
        # this parameter and its Jinja2 template will raise an error if sent.
        _is_qwen3 = "qwen3" in target_model.lower() or "qwq" in target_model.lower()
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if _is_qwen3 else {}

        for attempt in range(1, self.max_retries + 1):
            try:
                create_kwargs: Dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "max_tokens": max_tokens,
                }
                extra_body = dict(extra)
                extra_body.update({"top_k": top_k, "repetition_penalty": repetition_penalty})
                create_kwargs["extra_body"] = extra_body
                if response_format is not None:
                    create_kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**create_kwargs)
                choice = response.choices[0]
                text = choice.message.content or ""
                
                metadata["finish_reason"] = choice.finish_reason
                if response.usage:
                    metadata["prompt_tokens"] = response.usage.prompt_tokens
                    metadata["completion_tokens"] = response.usage.completion_tokens

                if self.enable_cache and text.strip():
                    self.cache[cache_key] = text

                return text, metadata

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "vLLM request failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        from openai import APIConnectionError, APITimeoutError

        error_type = ModelError
        if isinstance(last_error, APIConnectionError) and not isinstance(last_error, APITimeoutError):
            error_type = ServerUnavailableError
        raise error_type(
            f"vLLM generation failed after {self.max_retries} attempts: {last_error}"
        ) from last_error
