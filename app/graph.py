import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.store.memory import InMemoryStore

from app.agents.clarification import clarification_agent
from app.agents.context import context_agent
from app.agents.fulfillment import fulfillment_agent
from app.agents.research import research_agent
from app.agents.styling import styling_agent
from app.state import AgentState


def router(state: AgentState):
    """
    Decides the next node based on the 'next_step' key in the state.
    """
    next_step = state.get("next_step")
    # print(f"Routing to: {next_step}")  # Debugging
    if next_step == "research_agent":
        return "research_agent"
    elif next_step == "clarification_agent":
        return "clarification_agent"
    elif next_step == "END":
        return END
    else:
        return END


def create_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("context_agent", context_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("styling_agent", styling_agent)
    workflow.add_node("fulfillment_agent", fulfillment_agent)
    workflow.add_node("clarification_agent", clarification_agent)

    # Set entry point
    workflow.set_entry_point("context_agent")

    # Add edges
    # All nodes can potentially route to any other node or END via the router (and context_agent logic)

    workflow.add_conditional_edges(
        "context_agent",
        router,
        {
            "research_agent": "research_agent",
            "clarification_agent": "clarification_agent",
            END: END,
        },
    )

    # Agents return to context_agent (planner) or router to decide next step?
    # For now, let's have them go back to context_agent to reassess/plan.
    # OR, if we trust they update state correctly, we can route directly.
    # But usually "Hub and Spoke" means they go back to the Hub (Context).

    workflow.add_edge("research_agent", "styling_agent")
    workflow.add_edge("clarification_agent", "context_agent")

    sqllite_url = os.getenv("SQLLITE_URL", "")

    # 1. Establish the connection object manually
    #    check_same_thread=False is often needed for dev servers/tests
    conn = sqlite3.connect("database.db", check_same_thread=False)

    # 2. Pass the connection object directly to the SqliteSaver constructor
    #    This avoids reliance on 'from_conn_string' or 'from_url' which vary by version.
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory, store=InMemoryStore())
