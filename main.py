"""
main.py – CLI entry point for the AutoStream AI Agent.

Run:
    python main.py

Optional flags:
    --demo      Run the built-in demo conversation automatically
    --debug     Print internal state after each turn
"""
from dotenv import load_dotenv
load_dotenv()
import sys
import os
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from agent.agent import AutoStreamAgent


# ══════════════════════════════════════════════════════════════════
# Demo conversation (matches the assignment's expected flow)
# ══════════════════════════════════════════════════════════════════

DEMO_SCRIPT = [
    "Hi there!",
    "Tell me about your pricing plans.",
    "What's included in the Pro plan exactly?",
    "Do you have a refund policy?",
    "That sounds good, I want to try the Pro plan for my YouTube channel.",
    "My name is Jordan Lee",
    "jordan.lee@example.com",
    "YouTube",
]


def run_demo(agent: AutoStreamAgent, debug: bool = False) -> None:
    print("\n" + "═" * 60)
    print("  🎬  AutoStream AI Agent  —  DEMO MODE")
    print("═" * 60 + "\n")

    for user_msg in DEMO_SCRIPT:
        print(f"  You : {user_msg}")
        response = agent.chat(user_msg)
        print(f"  Alex: {response}\n")
        if debug:
            print(f"  [State] {agent.get_state_summary()}\n")

    print("═" * 60)
    print("  Demo complete.")
    print("═" * 60 + "\n")


def run_interactive(agent: AutoStreamAgent, debug: bool = False) -> None:
    print("\n" + "═" * 60)
    print("  🎬  AutoStream AI Agent")
    print("  Type 'quit' or 'exit' to end the session.")
    print("  Type '/state' to inspect internal state (debug).")
    print("═" * 60 + "\n")

    while True:
        try:
            user_input = input("  You : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye! 👋\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\n  Goodbye! 👋\n")
            break

        if user_input.lower() == "/state":
            import json
            print(f"\n  [State] {json.dumps(agent.get_state_summary(), indent=2)}\n")
            continue

        response = agent.chat(user_input)
        print(f"\n  Alex: {response}\n")

        if debug:
            print(f"  [State] {agent.get_state_summary()}\n")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AutoStream Conversational AI Agent")
    parser.add_argument("--demo",  action="store_true", help="Run the built-in demo script")
    parser.add_argument("--debug", action="store_true", help="Print internal state after each turn")
    args = parser.parse_args()

    # Verify API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n  ⚠️  GROQ_API_KEY is not set.")
        print("  Get a free key at: https://console.groq.com\n")
        print("  Then add it to your .env file:\n")
        print("      GROQ_API_KEY=gsk_...\n")
        sys.exit(1)

    agent = AutoStreamAgent()

    if args.demo:
        run_demo(agent, debug=args.debug)
    else:
        run_interactive(agent, debug=args.debug)


if __name__ == "__main__":
    main()
