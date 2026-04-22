"""
tests/test_agent.py – Unit tests for the AutoStream agent.

Run: pytest tests/ -v
"""

import sys
import os
import re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.rag import retrieve, _flatten_kb, _load_kb
from agent.agent import AutoStreamAgent, Intent
from tools.lead_capture import mock_lead_capture, get_all_leads


# ══════════════════════════════════════════════════════════════════
# RAG tests (no API calls)
# ══════════════════════════════════════════════════════════════════

class TestRAG:
    def test_kb_loads(self):
        kb = _load_kb()
        assert "plans" in kb
        assert "policies" in kb
        assert len(kb["plans"]) == 2

    def test_flatten_produces_chunks(self):
        kb = _load_kb()
        chunks = _flatten_kb(kb)
        assert len(chunks) > 5
        assert all("text" in c and "source" in c for c in chunks)

    def test_retrieve_pricing_query(self):
        result = retrieve("what is the price of the Pro plan?")
        assert "Pro" in result or "79" in result

    def test_retrieve_refund_query(self):
        result = retrieve("refund policy")
        assert "refund" in result.lower() or "7 days" in result.lower()

    def test_retrieve_support_query(self):
        result = retrieve("24/7 support")
        assert "support" in result.lower()

    def test_retrieve_returns_string(self):
        result = retrieve("hello world random query xyz123")
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════
# Tool tests (no API calls)
# ══════════════════════════════════════════════════════════════════

class TestLeadCapture:
    def test_mock_lead_capture_returns_dict(self, capsys):
        result = mock_lead_capture("Test User", "test@example.com", "YouTube")
        assert result["name"] == "Test User"
        assert result["email"] == "test@example.com"
        assert result["platform"] == "YouTube"
        assert "captured_at" in result

    def test_mock_lead_capture_prints(self, capsys):
        mock_lead_capture("Jane Doe", "jane@example.com", "Instagram")
        captured = capsys.readouterr()
        assert "Jane Doe" in captured.out
        assert "jane@example.com" in captured.out

    def test_leads_accumulate(self):
        before = len(get_all_leads())
        mock_lead_capture("Another User", "another@example.com", "TikTok")
        assert len(get_all_leads()) == before + 1

    def test_email_normalised_lowercase(self, capsys):
        result = mock_lead_capture("Case Test", "UPPER@EXAMPLE.COM", "Twitter")
        assert result["email"] == "upper@example.com"


# ══════════════════════════════════════════════════════════════════
# Intent parsing tests (no API calls)
# ══════════════════════════════════════════════════════════════════

class TestIntentParsing:
    def test_parse_greeting(self):
        assert AutoStreamAgent._parse_intent("[INTENT:greeting] Hello!") == Intent.GREETING

    def test_parse_product_inquiry(self):
        assert AutoStreamAgent._parse_intent("[INTENT:product_inquiry] Here are our plans...") == Intent.PRODUCT_INQUIRY

    def test_parse_high_intent(self):
        assert AutoStreamAgent._parse_intent("[INTENT:high_intent] Great, let me get your details.") == Intent.HIGH_INTENT

    def test_parse_unknown_when_missing(self):
        assert AutoStreamAgent._parse_intent("No tag here at all.") == Intent.UNKNOWN

    def test_clean_reply_strips_intent_tag(self):
        raw = "[INTENT:greeting] Hi there, welcome to AutoStream!"
        clean = AutoStreamAgent._clean_reply(raw)
        assert "[INTENT:" not in clean
        assert "Hi there" in clean

    def test_clean_reply_strips_lead_ready(self):
        raw = "[INTENT:high_intent] [LEAD_READY] All done!"
        clean = AutoStreamAgent._clean_reply(raw)
        assert "[LEAD_READY]" not in clean
        assert "All done" in clean


# ══════════════════════════════════════════════════════════════════
# Lead field extraction tests (no API calls)
# ══════════════════════════════════════════════════════════════════

class TestLeadExtraction:
    def _make_agent_in_collection(self) -> AutoStreamAgent:
        agent = AutoStreamAgent.__new__(AutoStreamAgent)
        agent.state = {
            "conversation_history": [],
            "current_intent": Intent.HIGH_INTENT,
            "lead_info": {},
            "lead_captured": False,
            "collecting_lead": True,
        }
        return agent

    def test_extracts_name(self):
        agent = self._make_agent_in_collection()
        agent._extract_lead_field("Jordan Lee")
        assert agent.state["lead_info"].get("name") == "Jordan Lee"

    def test_extracts_email(self):
        agent = self._make_agent_in_collection()
        agent.state["lead_info"]["name"] = "Jordan"
        agent._extract_lead_field("jordan@example.com")
        assert agent.state["lead_info"].get("email") == "jordan@example.com"

    def test_extracts_email_from_sentence(self):
        agent = self._make_agent_in_collection()
        agent.state["lead_info"]["name"] = "Jordan"
        agent._extract_lead_field("Sure, my email is jordan@example.com, thanks!")
        assert agent.state["lead_info"].get("email") == "jordan@example.com"

    def test_extracts_platform_keyword(self):
        agent = self._make_agent_in_collection()
        agent.state["lead_info"] = {"name": "Jordan", "email": "j@e.com"}
        agent._extract_lead_field("I mainly use YouTube for my content")
        assert agent.state["lead_info"].get("platform") == "Youtube"

    def test_extracts_platform_freetext(self):
        agent = self._make_agent_in_collection()
        agent.state["lead_info"] = {"name": "Jordan", "email": "j@e.com"}
        agent._extract_lead_field("Rumble")
        assert agent.state["lead_info"].get("platform") == "Rumble"
