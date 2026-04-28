from __future__ import annotations
import json
import re
from datetime import datetime
from anthropic import Anthropic
from langchain_core.messages import AIMessage
from graph.state import TripState
from models.itinerary import ItineraryVersion, Stop, StopType
from decimal import Decimal

_client = Anthropic()

_SYSTEM_PROMPT = """Role: You are an expert travel itinerary builder for the Adaptive Travel Companion.

End Goal: Compose a structured, logical day-by-day itinerary based on the user's constraint profile and the provided destination research context.

Instructions:
- You must ONLY use the options provided in the research context.
- Ensure the sequence of stops makes logical and geographical sense (e.g., meals between activities, realistic daily pacing).
- Each stop must have a unique `id`, `type`, `name`, and optionally `doc_id` mapping back to the research context.
- You must create a Directed Acyclic Graph (DAG) by defining `dag_edges` which represent chronological dependencies `["from_stop_id", "to_stop_id"]`.

Steps (Approach this step-by-step):
1. Review the requested number of days and constraints.
2. Select appropriate options from the provided research context, ensuring a good balance of activities and meals.
3. Group the selected stops into logical daily sequences.
4. Define chronological dependencies between the stops to form the `dag_edges`.

Narrowing / Output Format:
Provide your step-by-step reasoning inside <reasoning> tags.
Then, output the final itinerary as a valid JSON block enclosed in <itinerary> tags EXACTLY matching this structure:
<itinerary>
{
  "stops": [
    {
      "id": "stop1",
      "type": "meal",
      "name": "Restaurant Name",
      "doc_id": "doc_001",
      "depends_on": []
    }
  ],
  "dag_edges": [
    ["stop1", "stop2"]
  ],
  "days": 3
}
</itinerary>
"""

def _extract_days_from_messages(messages) -> int:
    for msg in reversed(messages):
        if msg.type == "human":
            match = re.search(r'(\d+)[ -]day', msg.content.lower())
            if match:
                return int(match.group(1))
    return 1

def _validate_itinerary(stops, profile, budget_ledger):
    return {"status": "valid", "issues": []}

def _format_itinerary_summary(itinerary: ItineraryVersion) -> str:
    return f"I've built a {itinerary.days}-day itinerary with {len(itinerary.stops)} stops for you."

def run_itinerary_builder(state: TripState) -> dict:
    if not state.get("profile"):
        return {}
        
    profile_ver = state["profile"]
    profile = profile_ver.profile
    
    rag_context = state.get("rag_context", {})
    rag_results = []
    for results in rag_context.values():
        rag_results.extend(results)
        
    days = _extract_days_from_messages(state.get("messages", []))
    
    context_str = json.dumps([
        {
            "doc_id": r.get("doc_id"), 
            "name": r.get("name"), 
            "category": r.get("category"), 
            "description": r.get("description")
        } for r in rag_results
    ])
    
    user_prompt = f"Please build a {days}-day itinerary using these options: {context_str}"
    
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    reply_content = response.content[0].text
    
    match = re.search(r"<itinerary>(.*?)</itinerary>", reply_content, re.DOTALL)
    if not match:
        return {"messages": [AIMessage(content="I'm sorry, I couldn't build the itinerary. Let's try again.")]}
        
    try:
        data = json.loads(match.group(1).strip())
        parsed_stops = data.get("stops", [])
        edges = data.get("dag_edges", [])
        
        stops = []
        for s in parsed_stops:
            stop = Stop(
                id=s["id"],
                type=StopType(s["type"]),
                name=s["name"],
                doc_id=s.get("doc_id"),
                depends_on=s.get("depends_on", [])
            )
            
            if stop.doc_id:
                orig_doc = next((r for r in rag_results if r.get("doc_id") == stop.doc_id), None)
                if orig_doc:
                    cat = orig_doc.get("category")
                    alts = [r for r in rag_results if r.get("category") == cat and r.get("doc_id") != stop.doc_id][:3]
                    for alt in alts:
                        alt_stop = Stop(
                            id=f"{stop.id}_alt_{alt.get('doc_id')}",
                            type=stop.type,
                            name=alt.get("name"),
                            doc_id=alt.get("doc_id")
                        )
                        stop.fallback_alternatives.append(alt_stop)
                        
            stops.append(stop)
            
        validation_report = _validate_itinerary(stops, profile, state.get("budget_ledger"))
        
        itinerary = ItineraryVersion(
            version_id=(state.get("itinerary_history", []) and state["itinerary_history"][-1].version_id + 1) or 1,
            created_at=datetime.utcnow(),
            stops=stops,
            dag_edges=edges,
            validation_report=validation_report,
            profile_version_id=profile_ver.version_id,
            days=days,
        )
        
        cleared_context = {k: [] for k in rag_context.keys()}
        
        return {
            "itinerary": itinerary,
            "itinerary_history": [itinerary],
            "rag_context": cleared_context,
            "messages": [AIMessage(content=_format_itinerary_summary(itinerary))],
            "state_version": state.get("state_version", 0) + 1,
        }
        
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error building itinerary: {e}")]}

def itinerary_builder_node(state: TripState) -> dict:
    return run_itinerary_builder(state)
