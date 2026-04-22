"""
rag.py – Lightweight RAG pipeline for AutoStream knowledge retrieval.

Uses a JSON knowledge base + simple TF-IDF-style cosine similarity
so the project runs without a vector database dependency.
For production, swap _retrieve() with a proper vector store
(Chroma, Pinecone, FAISS, etc.).
"""

import json
import math
import re
from pathlib import Path
from typing import Optional

# ── Path resolution ────────────────────────────────────────────────
_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def _load_kb() -> dict:
    with open(_KB_PATH, "r") as f:
        return json.load(f)


def _flatten_kb(kb: dict) -> list[dict]:
    """Turn the nested JSON into a flat list of searchable chunks."""
    chunks: list[dict] = []

    # Company overview
    chunks.append({
        "id": "company_overview",
        "text": f"{kb['company']['name']}: {kb['company']['description']} – {kb['company']['tagline']}",
        "source": "company",
    })

    # Plans
    for plan in kb["plans"]:
        features_str = "; ".join(plan["features"])
        limitations_str = "; ".join(plan["limitations"]) if plan["limitations"] else "none"
        text = (
            f"{plan['name']}: ${plan['price_monthly']}/month. "
            f"Features: {features_str}. "
            f"Limitations: {limitations_str}."
        )
        chunks.append({"id": plan["name"].lower().replace(" ", "_"), "text": text, "source": "pricing"})

    # Policies
    for policy in kb["policies"]:
        chunks.append({
            "id": f"policy_{policy['topic'].lower().replace(' ', '_')}",
            "text": f"{policy['topic']}: {policy['details']}",
            "source": "policy",
        })

    # FAQ
    for faq in kb["faq"]:
        chunks.append({
            "id": f"faq_{hash(faq['question']) % 10000}",
            "text": f"Q: {faq['question']} A: {faq['answer']}",
            "source": "faq",
        })

    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    total = len(tokens) or 1
    return {k: v / total for k, v in freq.items()}


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    keys = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Return the top-k most relevant knowledge-base chunks as a
    formatted string ready to inject into the LLM prompt.
    """
    kb = _load_kb()
    chunks = _flatten_kb(kb)

    q_tf = _tf(_tokenize(query))
    scored = []
    for chunk in chunks:
        c_tf = _tf(_tokenize(chunk["text"]))
        score = _cosine(q_tf, c_tf)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if not top or top[0][0] == 0:
        return "No relevant information found in the knowledge base."

    result_parts = []
    for score, chunk in top:
        if score > 0:
            result_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")

    return "\n\n".join(result_parts) if result_parts else "No relevant information found."


def get_full_kb_summary() -> str:
    """Return a condensed summary of all knowledge for the system prompt."""
    kb = _load_kb()
    plans_summary = []
    for plan in kb["plans"]:
        features = ", ".join(plan["features"])
        plans_summary.append(f"- {plan['name']}: ${plan['price_monthly']}/mo | {features}")

    policies_summary = [f"- {p['topic']}: {p['details']}" for p in kb["policies"]]

    return (
        "=== AutoStream Knowledge Base ===\n\n"
        "PLANS:\n" + "\n".join(plans_summary) + "\n\n"
        "POLICIES:\n" + "\n".join(policies_summary)
    )
