# 🎬 AutoStream AI Agent

> A conversational AI agent that detects user intent, answers product questions via RAG, and captures qualified leads — built for the ServiceHive / Inflx ML Intern assignment.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [WhatsApp Deployment via Webhooks](#whatsapp-deployment-via-webhooks)
- [Running Tests](#running-tests)
- [Example Conversation](#example-conversation)

---

## Quick Start

### 1. Clone & enter the project

```bash
git clone https://github.com/<your-username>/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_YOUR_API_KEY
```

Get a free key (no credit card needed) at: https://console.groq.com

### 5. Run the agent

```bash
# Interactive mode
python main.py

# Auto-run the built-in demo conversation
python main.py --demo

# Show internal state after each turn (debug)
python main.py --demo --debug
```

---

## Project Structure

```
autostream-agent/
├── main.py                          # CLI entry point
├── requirements.txt
├── README.md
├── .env                             # API key (not committed to git)
├── knowledge_base/
│   └── autostream_kb.json           # Plans, policies, FAQ
├── agent/
│   ├── agent.py                     # Core agent + state machine
│   └── rag.py                       # RAG retrieval pipeline
├── tools/
│   └── lead_capture.py              # mock_lead_capture() tool
└── tests/
    └── test_agent.py                # 21 unit tests (pytest)
```

---

## Architecture

### Why a custom state machine (LangGraph-compatible) instead of full LangGraph / AutoGen?

The assignment approves LangGraph / AutoGen as orchestration frameworks and LLaMA / Groq as the LLM. Rather than adding the overhead of a full framework for a 5–6 turn workflow, we implemented a **lightweight custom state machine using Python `TypedDict`** — the same pattern LangGraph uses internally. The architecture maps 1-to-1 to LangGraph nodes and is directly portable if needed. This keeps the dependency footprint minimal while fully satisfying the state management requirement.

The LLM backbone is **LLaMA 3.3 70B served via Groq**, which provides a completely free API tier with fast inference — no credit card required.

### State Management

The agent maintains an `AgentState` TypedDict with five fields, updated on every turn:

| Field | Type | Purpose |
|---|---|---|
| `conversation_history` | `list[dict]` | Full message log sent to LLM on every call — retains memory across all turns |
| `current_intent` | `Intent` enum | Parsed from LLM's `[INTENT:…]` tag each turn |
| `lead_info` | `dict` | Progressively filled: `name → email → platform` |
| `collecting_lead` | `bool` | Set to `True` once high-intent is detected |
| `lead_captured` | `bool` | Set to `True` after `mock_lead_capture()` is called |

### Turn Lifecycle

```
User message
     │
     ▼
 RAG retrieval ──► top-3 cosine-similar KB chunks
     │
     ▼
 LLM call (Groq – LLaMA 3.3 70B) ──► raw reply with [INTENT:…] tag
     │
     ├── Parse intent → update state.current_intent
     ├── If HIGH_INTENT → set collecting_lead = True
     ├── If collecting_lead → extract next lead field from user msg
     ├── If [LEAD_READY] in reply → call mock_lead_capture()
     │
     ▼
 Strip internal tags → return clean reply to user
```

### RAG Pipeline

Rather than requiring a vector database, the RAG layer uses **cosine similarity on TF-IDF term vectors**, computed in pure Python with zero extra dependencies. The knowledge base is a structured JSON file (`knowledge_base/autostream_kb.json`) flattened into searchable text chunks at query time. In production, swap `retrieve()` in `agent/rag.py` with a Chroma / Pinecone / FAISS call — the interface is identical.

### Lead Capture Guard

`mock_lead_capture()` is gated by **three independent checks** to ensure it never fires prematurely:

1. `collecting_lead` must be `True` (only set on HIGH_INTENT detection)
2. `[LEAD_READY]` must appear in the LLM's response (LLM self-confirms all fields collected)
3. All three fields (`name`, `email`, `platform`) must be present in `state.lead_info`

---

## WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, use the **WhatsApp Business API (Cloud API)** provided by Meta, integrated via a webhook endpoint.

### Architecture

```
WhatsApp User
     │  (sends message)
     ▼
Meta WhatsApp Cloud API
     │  (HTTP POST webhook event)
     ▼
Your Server  ←──── FastAPI / Flask app
     │
     ├── Verify webhook token (GET /webhook)
     ├── Parse incoming message body (POST /webhook)
     ├── Route to AutoStreamAgent.chat(user_message)
     │
     ▼
Meta Send Message API  ──► WhatsApp User (reply)
```

### Step-by-Step

1. **Create a Meta Developer App** at [developers.facebook.com](https://developers.facebook.com), enable the WhatsApp product, and get a phone number.

2. **Expose your server** (use ngrok in dev, a cloud VM in prod):
   ```bash
   ngrok http 8000
   ```

3. **Implement the webhook** (FastAPI example):

```python
from fastapi import FastAPI, Request
from dotenv import load_dotenv
import httpx, os

load_dotenv()

app = FastAPI()
agent_sessions: dict = {}   # phone_number → AutoStreamAgent instance

VERIFY_TOKEN = os.environ["WA_VERIFY_TOKEN"]
WA_TOKEN     = os.environ["WA_TOKEN"]
PHONE_ID     = os.environ["WA_PHONE_NUMBER_ID"]

@app.get("/webhook")
async def verify(hub_mode, hub_verify_token, hub_challenge):
    if hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return {"error": "invalid token"}, 403

@app.post("/webhook")
async def receive(request: Request):
    body = await request.json()
    msg  = body["entry"][0]["changes"][0]["value"]["messages"][0]
    from_number = msg["from"]
    text        = msg["text"]["body"]

    # Per-user agent session
    if from_number not in agent_sessions:
        agent_sessions[from_number] = AutoStreamAgent()
    reply = agent_sessions[from_number].chat(text)

    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v19.0/{PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {WA_TOKEN}"},
            json={"messaging_product": "whatsapp",
                  "to": from_number,
                  "text": {"body": reply}},
        )
    return {"status": "ok"}
```

4. **Register the webhook URL** in the Meta Developer Console under WhatsApp → Configuration → Webhook, subscribing to the `messages` field.

5. **State persistence**: For multi-user production, replace the in-memory `agent_sessions` dict with Redis (keyed by phone number), serialising `AgentState` as JSON.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 21 tests run without an API key — they test RAG retrieval, intent parsing, lead field extraction, and tool logic entirely in isolation.

---

## Example Conversation

```
You : Hi there!
Alex: Hi, it's great to meet you! Is there something I can help you with,
      or would you like to learn more about AutoStream?

You : Tell me about your pricing plans.
Alex: We have two plans at AutoStream:
      • Basic – $29/month: 10 videos/month, 720p export, basic auto-editing tools.
      • Pro   – $79/month: Unlimited videos, 4K export, AI captions, 24/7 support.
      Would you like to know more about either plan?

You : What's included in the Pro plan exactly?
Alex: The Pro plan includes unlimited video exports, 4K resolution, AI-powered captions,
      advanced auto-editing tools, premium templates, 24/7 priority support,
      custom branding, an analytics dashboard, and unlimited cloud storage.

You : Do you have a refund policy?
Alex: Yes — you can request a refund within 7 days of purchase and it will be processed
      within 5–7 business days. No refunds are issued after that window.

You : That sounds good, I want to try the Pro plan for my YouTube channel.
Alex: That's great! To get you started, could you please tell me your name?

You : Sailaja Patneedi
Alex: Thanks, Sailaja! Could you share your email address?

You : sailajap@example.com
Alex: Perfect. And which creator platform will you mainly be using — YouTube, Instagram, TikTok?

You : YouTube

═══════════════════════════════════════════════════════
  ✅  Lead captured successfully!
      Name     : Sailaja Patneedi
      Email    : sailajap@example.com
      Platform : Youtube
═══════════════════════════════════════════════════════

Alex: You're all set, Sailaja! Our team will reach out to sailajap@example.com shortly.
      Welcome to AutoStream! 🚀
```

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM | Groq – LLaMA 3.3 70B | Free tier, fast inference, no credit card needed |
| State | Custom TypedDict (LangGraph-compatible) | Zero extra deps; directly portable to LangGraph |
| RAG | TF-IDF cosine similarity (pure Python) | No vector DB required; swap-in ready for Chroma/FAISS |
| Knowledge Base | JSON | Easy to edit; maps cleanly to structured RAG chunks |
| Tests | pytest | 21 isolated unit tests, no API key needed |
