"""Tests for the LangGraph ReAct agent structure."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_agent.agent import (
    SYSTEM_PROMPT,
    AgentState,
    call_llm,
    graph,
    route,
    tools,
)


# -- graph structure -----------------------------------------------------------


def test_graph_compiles():
    assert graph is not None
    assert type(graph).__name__ == "CompiledStateGraph"


def test_graph_has_expected_nodes():
    node_names = set(graph.nodes.keys())
    assert "agent" in node_names
    assert "tools" in node_names


def test_tools_list_contains_find_appointments():
    tool_names = [t.name for t in tools]
    assert "find_appointments" in tool_names


def test_state_has_expected_keys():
    keys = list(AgentState.__annotations__.keys())
    assert "messages" in keys
    assert "user_id" in keys
    assert "error" in keys


# -- system prompt -------------------------------------------------------------


def test_system_prompt_mentions_role():
    assert "clinical assistant" in SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_tools():
    assert "find_appointments" in SYSTEM_PROMPT


def test_system_prompt_warns_against_fabrication():
    assert "fabricate" in SYSTEM_PROMPT.lower()


# -- route function ------------------------------------------------------------


def test_route_returns_end_for_plain_message():
    state: AgentState = {
        "messages": [AIMessage(content="Hello")],
        "user_id": "test",
        "error": None,
    }
    assert route(state) == "__end__"


def test_route_returns_tools_for_tool_calls():
    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "find_appointments", "args": {}, "id": "call_1", "type": "tool_call"}
        ],
    )
    state: AgentState = {
        "messages": [msg],
        "user_id": "test",
        "error": None,
    }
    assert route(state) == "tools"


def test_route_returns_end_for_empty_tool_calls():
    msg = AIMessage(content="Done", tool_calls=[])
    state: AgentState = {
        "messages": [msg],
        "user_id": "test",
        "error": None,
    }
    assert route(state) == "__end__"
