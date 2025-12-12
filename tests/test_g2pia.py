import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from redteam.modules import g2pia_defensive as g2pia  # noqa: E402


def test_extract_core_words_basic():
    result = g2pia.extract_core_words("The cat eats fish.")
    assert result["subject"].lower() == "cat"
    assert result["predicate"].startswith("eat")
    assert result["object"].lower() == "fish"


def test_candidate_core_pool_synonyms_and_numbers(monkeypatch):
    class DummyLemma:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

    class DummySyn:
        def __init__(self, names):
            self._names = names

        def lemmas(self):
            return [DummyLemma(n) for n in self._names]

    def fake_synsets(word):
        if word == "eat":
            return [DummySyn(["consume"])]
        if word == "apple":
            return [DummySyn(["fruit"])]
        return []

    monkeypatch.setattr(
        g2pia, "_WORDNET", type("WN", (), {"synsets": staticmethod(fake_synsets)})()
    )
    pool = g2pia.candidate_core_pool("eat", "apple")
    assert ("consume", "apple") in pool
    assert ("eat", "fruit") in pool
    assert any(obj.split()[-1].isdigit() for _, obj in pool)


def test_build_generation_prompt_contains_banner():
    prompt = g2pia.build_generation_prompt(
        {"subject": "agent", "predicate": "observe", "object": "pattern"}
    )
    assert g2pia.SAFETY_BANNER in prompt
    assert "one short" in prompt.lower()
    assert "number" in prompt.lower()


def test_check_constraints_custom_embeddings():
    def fake_word_embedding(word: str):
        return np.array([len(word), 1.0, 0.0])

    def fake_sentence_embedding(text: str):
        return np.array([len(text), 0.5, 0.5])

    scores = g2pia.check_constraints(
        "base text",
        "base text 123",
        epsilon=0.6,
        gamma=0.2,
        delta=1.0,
        victim_embedding_fn=fake_sentence_embedding,
        core_embedding_fn=fake_word_embedding,
    )
    assert scores["semantic_distance"] < 0.6
    assert scores["semantic_ok"]
    assert scores["cosine_ok"]


def test_evaluate_attack_set_respects_flag():
    dataset_path = ROOT / "tests" / "data" / "g2pia_synthetic.jsonl"
    dataset = g2pia.load_dataset(dataset_path)
    generator_called = {"count": 0}

    def math_infer(prompt: str) -> str:
        match = re.search(r"(\d+)\s*\+\s*(\d+)", prompt)
        if not match:
            return "0"
        a, b = match.groups()
        return str(int(a) + int(b))

    def stub_generator(**kwargs):
        generator_called["count"] += 1
        return ["injected"]

    report = g2pia.evaluate_attack_set(
        dataset=dataset,
        victim_model="stub",
        candidates_per_sample=2,
        allow_adversarial_sim=False,
        victim_infer_fn=math_infer,
        candidate_generator=stub_generator,
    )

    assert generator_called["count"] == 0
    assert report["metrics"]["clean_accuracy"] == pytest.approx(1.0)
    assert report["metrics"]["attack_success_rate"] == pytest.approx(0.0)
    assert report["metrics"]["attack_accuracy"] == pytest.approx(0.0)


def test_evaluate_attack_set_with_mock_victim():
    dataset = [("What is 1 + 1?", "2")]

    def victim(prompt: str) -> str:
        if "injection" in prompt:
            return "wrong"
        return "2"

    def generator(**kwargs):
        return ["this is an injection"]

    report = g2pia.evaluate_attack_set(
        dataset=dataset,
        victim_model="stub",
        candidates_per_sample=1,
        allow_adversarial_sim=True,
        victim_infer_fn=victim,
        candidate_generator=generator,
    )

    assert report["metrics"]["clean_accuracy"] == pytest.approx(1.0)
    assert report["metrics"]["attack_success_rate"] == pytest.approx(1.0)
    assert report["metrics"]["attack_accuracy"] == pytest.approx(0.0)
