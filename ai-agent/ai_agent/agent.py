"""LangGraph ReAct agent for OpenEMR clinical assistant."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from ai_agent.tools.find_appointments import find_appointments

SYSTEM_PROMPT = """\
You are a clinical assistant for an OpenEMR electronic health record system.

Your role:
- Help staff look up appointments, patient information, and scheduling details.
- Provide clear, concise, clinically appropriate responses.
- Never fabricate data — only report what the tools return.
- When a query is ambiguous, ask clarifying questions before searching.

Available tools:
- find_appointments: Search for appointments by patient name, date, provider, \
status, or patient ID. Use this whenever the user asks about appointments or \
scheduling.

Guidelines:
- Present appointment results in a clear, readable format.
- Include relevant details: patient name, date, time, provider, status.
- If no results are found, say so clearly and suggest refining the search.
- If multiple patients match a name, present the options and ask the user to \
clarify.
- Respect patient privacy — only share information relevant to the query.\
"""


class AgentState(TypedDict):
    """State flowing through the agent graph."""

    messages: Annotated[list, add_messages]
    user_id: str
    error: str | None


# -- tools ---------------------------------------------------------------------

tools = [find_appointments]

# -- model ---------------------------------------------------------------------

model = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
model_with_tools = model.bind_tools(tools)

# -- nodes ---------------------------------------------------------------------


def call_llm(state: AgentState) -> dict[str, Any]:
    """Invoke the LLM with the system prompt and conversation history."""
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    response = model_with_tools.invoke([system_msg] + state["messages"])
    return {"messages": [response]}


def route(state: AgentState) -> str:
    """Route to tools if the last message has tool calls, otherwise end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# -- graph construction --------------------------------------------------------

builder = StateGraph(AgentState)
builder.add_node("agent", call_llm)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
