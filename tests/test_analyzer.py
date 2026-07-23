"""
Unit tests for agent/analyzer.py — no live API calls.
Tests verify that new parameters (user_request, single_only, count, avoid_names)
correctly affect the prompt sent to the LLM.
"""
import json
from unittest.mock import patch

import pytest

from agent.analyzer import analyze


_VALID_RESPONSE = json.dumps({
    "niche": "ukiyo-e wall art",
    "market_observations": "high demand",
    "recurring_patterns": "cranes, mountains",
    "potential_opportunities": "seasonal sets",
    "poster_concepts": [{"name": "Mountain at Dusk"}],
})


def _mock_ask(response=_VALID_RESPONSE):
    return patch("agent.analyzer.ask", return_value=response)


def test_user_request_appears_in_prompt():
    captured = {}
    def fake_ask(prompt, **kwargs):
        captured["prompt"] = prompt
        return _VALID_RESPONSE
    with patch("agent.analyzer.ask", fake_ask):
        analyze([{"title": "Product 1"}], user_request="vintage botanical prints")
    assert "vintage botanical prints" in captured["prompt"]


def test_single_only_flag_appears_in_prompt():
    captured = {}
    def fake_ask(prompt, **kwargs):
        captured["prompt"] = prompt
        return _VALID_RESPONSE
    with patch("agent.analyzer.ask", fake_ask):
        analyze([{"title": "Product 1"}], single_only=True)
    assert "single" in captured["prompt"]


def test_count_appears_in_prompt():
    captured = {}
    def fake_ask(prompt, **kwargs):
        captured["prompt"] = prompt
        return _VALID_RESPONSE
    with patch("agent.analyzer.ask", fake_ask):
        analyze([{"title": "Product 1"}], count=5)
    assert "5" in captured["prompt"]


def test_avoid_names_appear_in_prompt():
    captured = {}
    def fake_ask(prompt, **kwargs):
        captured["prompt"] = prompt
        return _VALID_RESPONSE
    with patch("agent.analyzer.ask", fake_ask):
        analyze([{"title": "Product 1"}], avoid_names=["Mountain Crane", "Edo Fox"])
    assert "Mountain Crane" in captured["prompt"]
    assert "Edo Fox" in captured["prompt"]


def test_no_extra_instructions_without_params():
    captured = {}
    def fake_ask(prompt, **kwargs):
        captured["prompt"] = prompt
        return _VALID_RESPONSE
    with patch("agent.analyzer.ask", fake_ask):
        analyze([{"title": "Product 1"}])
    assert "user specifically wants" not in captured["prompt"]
    assert '"single_or_set" must be exactly "single"' not in captured["prompt"]
    assert "already used" not in captured["prompt"]


def test_missing_required_field_raises():
    bad = json.dumps({"niche": "x", "poster_concepts": []})
    with _mock_ask(bad):
        with pytest.raises(ValueError, match="missing fields"):
            analyze([])
