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

from langchain_core.messages import HumanMessage, SystemMessage
from graph.graph import graph
from graph.state import empty_state
from workers import queue

SESSION_ID = "cli-session-001"
CONFIG = {"configurable": {"thread_id": SESSION_ID}}
WAKE_SIGNAL_PATH = ".wake_signal"


def _check_disruption_wake_signal() -> None:
    """If the Live Data Worker wrote a wake signal, inject queued disruptions into the graph."""
    if not os.path.exists(WAKE_SIGNAL_PATH):
        return
    os.remove(WAKE_SIGNAL_PATH)
    pending = queue.dequeue_pending()
    if not pending:
        return
    graph.update_state(CONFIG, {"disruption_queue": [e.model_dump() for e in pending]})
    result = graph.invoke(
        {"messages": [SystemMessage(content="A disruption has been detected on your itinerary.")]},
        config=CONFIG,
    )
    messages = result.get("messages", [])
    if messages:
        print(f"\nAssistant: {messages[-1].content}\n")
    for e in pending:
        queue.mark_processed(e.event_key)


def main():
    print("Adaptive Travel Companion")
    print("Type your message and press Enter. Ctrl+C to exit.\n")

    current_state = graph.get_state(CONFIG)
    if not current_state.values:
        initial = empty_state(SESSION_ID)
        graph.update_state(CONFIG, initial)

    while True:
        _check_disruption_wake_signal()

        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        prev_itinerary = graph.get_state(CONFIG).values.get("itinerary")

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
        )

        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            print(f"\nAssistant: {last.content}\n")

        itinerary = result.get("itinerary")
        if itinerary and itinerary != prev_itinerary:
            print("-" * 20)
            print(f"--- {itinerary.days}-DAY ITINERARY ---")
            for stop in itinerary.stops:
                print(f"[{stop.type.upper()}] {stop.name}")
            print("-" * 20 + "\n")


if __name__ == "__main__":
    main()
