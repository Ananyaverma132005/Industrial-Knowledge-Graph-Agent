"""Mistral via Ollama (ChatOllama). Availability is re-checked every 30 s so the app picks the
LLM up as soon as `ollama serve` is running, and degrades gracefully when it isn't."""
import time

import httpx

import config


class LLMProvider:
    def __init__(self):
        self._llm = None
        self._checked_at = 0.0
        self._available = False
        self.status = "not checked"

    def _check(self) -> bool:
        if config.USE_LLM == "false":
            self.status = "disabled (USE_LLM=false)"
            return False
        try:
            tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=1.5).json()
            names = [m.get("name", "") for m in tags.get("models", [])]
            if not any(n.split(":")[0] == config.OLLAMA_MODEL.split(":")[0] for n in names):
                self.status = f"Ollama running but model '{config.OLLAMA_MODEL}' not pulled"
                return False
            self.status = f"{config.OLLAMA_MODEL} via Ollama"
            return True
        except Exception:  # noqa: BLE001
            self.status = "Ollama not reachable"
            return False

    def get(self):
        """Return a ChatOllama instance, or None when the LLM isn't available."""
        if time.time() - self._checked_at > 30:
            self._available = self._check()
            self._checked_at = time.time()
        if not self._available:
            return None
        if self._llm is None:
            from langchain_ollama import ChatOllama

            self._llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
        return self._llm
