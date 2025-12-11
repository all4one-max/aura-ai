from langgraph.graph import END, StateGraph
from langgraph.store.memory import InMemoryStore

from app.agents.clarification import clarification_agent
from app.agents.context import context_agent
from app.agents.ranking import ranking_agent
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
    elif next_step == "END":
        return END
    else:
        return END


def create_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("context_agent", context_agent)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("styling_agent", styling_agent)
    workflow.add_node("ranking_agent", ranking_agent)
    workflow.add_node("clarification_agent", clarification_agent)

    # Set entry point
    workflow.set_entry_point("context_agent")

    # Add edges
    workflow.add_conditional_edges(
        "context_agent",
        router,
        {
            "research_agent": "research_agent",
            "clarification_agent": "clarification_agent",
            END: END,
        },
    )

    workflow.add_edge("research_agent", "styling_agent")
    workflow.add_edge("styling_agent", "ranking_agent")
    workflow.add_edge("ranking_agent", END)
    # Clarification agent ends the flow - user needs to send another message
    # When user sends next message, it will start from context_agent again (state is persisted)
    workflow.add_edge("clarification_agent", END)

    return workflow.compile(checkpointer=checkpointer, store=InMemoryStore())
