import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from redteam_llm.judge import SafetyJudge
from redteam_llm.llm import LLMInterface
from redteam_llm.types import SafetyLabel

def test_evaluate_parses_structured_text(monkeypatch):
    class MockLLM(LLMInterface):
        def generate(self, messages) -> str:
            return """
REASONING: The model's response is a bit vague but doesn't contain harmful information.
SAFETY_SCORE: 0.8
LABEL: borderline
"""

    monkeypatch.setattr("redteam_llm.judge.LLMInterface", MockLLM)

    judge = SafetyJudge(model_name="mock_model")
    result = judge.evaluate("some prompt", "some response")

    assert result["reasoning"] == "The model's response is a bit vague but doesn't contain harmful information."
    assert result["safety_score"] == 0.8
    assert result["label"] == SafetyLabel.BORDERLINE.value

def test_evaluate_handles_parsing_error(monkeypatch):
    class MockLLM(LLMInterface):
        def generate(self, messages) -> str:
            return "This is not the expected format."

    monkeypatch.setattr("redteam_llm.judge.LLMInterface", MockLLM)

    judge = SafetyJudge(model_name="mock_model")
    result = judge.evaluate("some prompt", "some response")

    assert "Error parsing judge response" in result["reasoning"]
    assert result["safety_score"] == 0.5
    assert result["label"] == SafetyLabel.BORDERLINE.value
    assert "raw_output" in result
