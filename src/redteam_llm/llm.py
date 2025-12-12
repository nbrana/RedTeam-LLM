from typing import Any, Dict, List, Optional

try:  # pragma: no cover - dependency may be unavailable in tests
    import ollama  # type: ignore
except Exception:  # pragma: no cover
    ollama = None

from .types import Message, Role


class LLMInterface:
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    def generate(
        self, messages: List[Message], max_tokens: Optional[int] = None
    ) -> str:
        formatted_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in messages
        ]
        options: Dict[str, Any] = {"temperature": self.temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if ollama is None:
            return f"Error calling model {self.model_name}: ollama is not installed"
        try:
            response = ollama.chat(
                model=self.model_name, messages=formatted_messages, options=options
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Error calling model {self.model_name}: {str(e)}"

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        return self.generate(
            [Message(role=Role.USER, content=prompt)], max_tokens=max_tokens
        )


def model_infer(
    prompt: str, model_name: str, temperature: float = 0.0, max_tokens: int = 128
) -> str:
    """
    Thin helper so the defensive G2PIA module can call into the same Ollama-backed
    interface used by the rest of the framework.
    """
    llm = LLMInterface(model_name=model_name, temperature=temperature)
    return llm.generate_text(prompt, max_tokens=max_tokens)
