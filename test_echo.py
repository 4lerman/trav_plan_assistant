import asyncio
from graph.graph import orchestrator_reply_node
from langchain_core.messages import HumanMessage
state = {"messages": [HumanMessage(content="so, what is the plan?")]}
print(orchestrator_reply_node(state))
