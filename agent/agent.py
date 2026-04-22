"""
agent.py – AutoStream Conversational AI Agent

Architecture:
  - State machine with 5 nodes (LangGraph-style TypedDict state)
  - Google Gemini 1.5 Flash as the LLM backbone (free tier)
  - RAG retrieval injected into each LLM call
  - Intent detection → lead qualification → tool execution flow
  - Full conversation memory retained across turns
"""

import os
import re
import sys
import json
from enum import Enum
from typing import TypedDict, Optional, Annotated
from groq import Groq
# ── Local imports ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent.rag import retrieve, get_full_kb_summary
from tools.lead_capture import mock_lead_capture


# ══════════════════════════════════════════════════════════════════
# State Definition
# ══════════════════════════════════════════════════════════════════

class Intent(str, Enum):
    GREETING       = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    HIGH_INTENT    = "high_intent"
    UNKNOWN        = "unknown"


class LeadInfo(TypedDict, total=False):
    name: str
    email: str
    platform: str


class AgentState(TypedDict):
    """Immutable-style state that is rebuilt on every turn."""
    conversation_history: list[dict]   # Full message history for LLM context
    current_intent: Intent             # Most recent intent classification
    lead_info: LeadInfo                # Collected lead fields
    lead_captured: bool                # Whether mock_lead_capture was called
    collecting_lead: bool              # Whether we are mid-collection


# ══════════════════════════════════════════════════════════════════
# System Prompt
# ══════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are Alex, a friendly and knowledgeable sales assistant for AutoStream – an AI-powered automated video editing SaaS for content creators.

YOUR RESPONSIBILITIES:
1. Greet users warmly and answer product/pricing questions accurately using the knowledge base below.
2. Detect when a user is ready to sign up or shows strong interest (high intent).
3. When high intent is detected, collect their Name, Email, and Creator Platform (e.g. YouTube, Instagram, TikTok) ONE field at a time.
4. Confirm lead capture only after ALL THREE fields are collected.

INTENT CLASSIFICATION – include one of these tags at the START of every response:
  [INTENT:greeting]         – User is just saying hi or making small talk
  [INTENT:product_inquiry]  – User asks about features, pricing, or policies
  [INTENT:high_intent]      – User explicitly wants to sign up, try a plan, or is clearly ready to buy

LEAD COLLECTION RULES:
- Only start collecting lead info when intent is high_intent.
- Ask for ONE field at a time: first Name, then Email, then Platform.
- After collecting all three, output exactly: [LEAD_READY] on its own line before your confirmation message.
- Never make up or assume lead details.
- Do NOT trigger lead capture for product inquiries alone.

KNOWLEDGE BASE:
{kb_summary}

RESPONSE STYLE:
- Conversational, warm, concise (2–4 sentences per turn).
- Use the retrieved context below when answering factual questions.
- Never mention "intent tags" or internal mechanics to the user.
"""


# ══════════════════════════════════════════════════════════════════
# Agent Class
# ══════════════════════════════════════════════════════════════════

class AutoStreamAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("\n\n  ⚠️  GROQ_API_KEY is not set.\n  Get a free key at: https://console.groq.com\n")
        self.client = Groq(api_key=api_key)
        self.kb_summary = get_full_kb_summary()
        self.system_prompt = _SYSTEM_PROMPT.format(kb_summary=self.kb_summary)

        # Initial state
        self.state: AgentState = {
            "conversation_history": [],
            "current_intent": Intent.UNKNOWN,
            "lead_info": {},
            "lead_captured": False,
            "collecting_lead": False,
        }

    # ── Core: call the LLM ─────────────────────────────────────────
    def _call_llm(self, user_message: str, rag_context: str) -> str:
        augmented_user = (
            f"[Relevant knowledge base context]\n{rag_context}\n\n"
            f"[User message]\n{user_message}"
        )
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + self.state["conversation_history"]
            + [{"role": "user", "content": augmented_user}]
        )
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=512,
        )
        return response.choices[0].message.content

    # ── Parse intent from LLM reply ────────────────────────────────
    @staticmethod
    def _parse_intent(reply: str) -> Intent:
        match = re.search(r"\[INTENT:(\w+)\]", reply)
        if not match:
            return Intent.UNKNOWN
        tag = match.group(1).lower()
        mapping = {
            "greeting": Intent.GREETING,
            "product_inquiry": Intent.PRODUCT_INQUIRY,
            "high_intent": Intent.HIGH_INTENT,
        }
        return mapping.get(tag, Intent.UNKNOWN)

    # ── Strip internal tags from visible reply ─────────────────────
    @staticmethod
    def _clean_reply(reply: str) -> str:
        cleaned = re.sub(r"\[INTENT:\w+\]\s*", "", reply)
        cleaned = re.sub(r"\[LEAD_READY\]\s*", "", cleaned)
        return cleaned.strip()

    # ── Lead field extraction ──────────────────────────────────────
    def _extract_lead_field(self, user_message: str) -> None:
        """
        Progressively fill lead_info based on which field we're waiting for.
        Simple heuristic extraction – the LLM guides the user to give clean input.
        """
        info = self.state["lead_info"]

        if "name" not in info:
            candidate = user_message.strip()
            # Strip common lead-in phrases like "my name is / I'm / I am"
            candidate = re.sub(r"(?i)^(my name is|i'm|i am)\s+", "", candidate).strip()
            if len(candidate.split()) >= 1 and len(candidate) < 60:
                info["name"] = candidate
        elif "email" not in info:
            email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", user_message)
            if email_match:
                info["email"] = email_match.group(0)
        elif "platform" not in info:
            platforms = ["youtube", "instagram", "tiktok", "twitter", "x", "linkedin",
                         "facebook", "twitch", "snapchat", "pinterest"]
            lower = user_message.lower()
            for p in platforms:
                if p in lower:
                    info["platform"] = p.capitalize()
                    break
            if "platform" not in info:
                # Accept free text if no keyword matched
                info["platform"] = user_message.strip()

        self.state["lead_info"] = info

    # ── Main chat method ───────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        """
        Process one user turn and return the agent's response string.
        Updates self.state in place.
        """
        # 1. Retrieve relevant context via RAG
        rag_context = retrieve(user_message, top_k=3)

        # 2. If we're mid lead-collection, try to extract the next field
        if self.state["collecting_lead"] and not self.state["lead_captured"]:
            self._extract_lead_field(user_message)

        # 3. Call LLM
        raw_reply = self._call_llm(user_message, rag_context)

        # 4. Parse intent and update state
        intent = self._parse_intent(raw_reply)
        self.state["current_intent"] = intent

        if intent == Intent.HIGH_INTENT and not self.state["collecting_lead"]:
            self.state["collecting_lead"] = True

        # 5. Check if LLM signals all lead fields are collected
        lead_ready = "[LEAD_READY]" in raw_reply
        if lead_ready and not self.state["lead_captured"]:
            info = self.state["lead_info"]
            if all(k in info for k in ("name", "email", "platform")):
                mock_lead_capture(info["name"], info["email"], info["platform"])
                self.state["lead_captured"] = True

        # 6. Update conversation history (store original user msg, not augmented)
        self.state["conversation_history"].append({"role": "user", "content": user_message})
        self.state["conversation_history"].append({"role": "assistant", "content": raw_reply})

        # 7. Return cleaned reply (no internal tags)
        return self._clean_reply(raw_reply)

    # ── State introspection helper ─────────────────────────────────
    def get_state_summary(self) -> dict:
        return {
            "intent": self.state["current_intent"].value,
            "collecting_lead": self.state["collecting_lead"],
            "lead_info": dict(self.state["lead_info"]),
            "lead_captured": self.state["lead_captured"],
            "turns": len(self.state["conversation_history"]) // 2,
        }
