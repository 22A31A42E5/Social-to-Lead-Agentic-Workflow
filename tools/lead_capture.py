"""
tools.py – Tool definitions for the AutoStream agent.

Contains the mock_lead_capture function and any other callable tools
the agent may invoke during a conversation.
"""

import json
import datetime
from typing import Optional


# ──────────────────────────────────────────────
# In-memory "database" (replace with a real DB in production)
# ──────────────────────────────────────────────
_captured_leads: list[dict] = []


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulate persisting a qualified lead to a CRM / backend system.

    Args:
        name:     Full name of the prospect.
        email:    Contact email address.
        platform: Primary creator platform (e.g. YouTube, Instagram).

    Returns:
        A dict confirming the captured lead details.
    """
    lead = {
        "id": len(_captured_leads) + 1,
        "name": name.strip(),
        "email": email.strip().lower(),
        "platform": platform.strip(),
        "captured_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "new",
    }
    _captured_leads.append(lead)

    # ── Console confirmation (required by assignment) ──────────────
    print(f"\n{'='*55}")
    print(f"  ✅  Lead captured successfully!")
    print(f"      Name     : {lead['name']}")
    print(f"      Email    : {lead['email']}")
    print(f"      Platform : {lead['platform']}")
    print(f"      Time     : {lead['captured_at']}")
    print(f"{'='*55}\n")

    return lead


def get_all_leads() -> list[dict]:
    """Return all leads captured in this session (for testing / inspection)."""
    return _captured_leads.copy()
