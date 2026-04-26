"""Interactive CLI for the Adaptive Travel Companion.

Usage:
    python -m trav_planner_assistant
    # or
    uv run python cli.py
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from graph.graph import graph
from graph.state import empty_state


SESSION_ID = "cli-session-001"
CONFIG = {"configurable": {"thread_id": SESSION_ID}}


def main():
    print("Adaptive Travel Companion")
    print("Type your message and press Enter. Ctrl+C to exit.\n")

    # Seed initial state into checkpointer
    initial = empty_state(SESSION_ID)
    graph.update_state(CONFIG, initial)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
        )

        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            print(f"\nAssistant: {last.content}\n")


if __name__ == "__main__":
    main()
