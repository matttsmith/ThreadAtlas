"""Shared error types for LLM backends."""


class LLMError(RuntimeError):
    """Raised when the LLM call itself fails (timeout, non-zero exit, HTTP error)."""


class LLMNotConfiguredError(LLMError):
    """Raised when a task is requested but the LLM is not enabled for it."""
