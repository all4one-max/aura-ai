from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.agents.context import context_agent
from app.agents.research import research_agent
from app.agents.styling import styling_agent
from app.agents.fulfillment import fulfillment_agent

def router(state: AgentState):
    """
    Decides the next node based on the 'next_step' key in the state.
    """
    next_step = state.get("next_step")
    if next_step == "research_agent":
        return "research_agent"
    elif next_step == "styling_agent":
        return "styling_agent"
    elif next_step == "fulfillment_agent":
        return "fulfillment_agent"
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

    # Set entry point
    workflow.set_entry_point("context_agent")

    # Add edges
    # For this simple linear flow, we route from one agent to the router which decides the next.
    # Actually, in LangGraph, we can use conditional edges.
    
    workflow.add_conditional_edges(
        "context_agent",
        router,
        {
            "research_agent": "research_agent",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "research_agent",
        router,
        {
            "styling_agent": "styling_agent",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "styling_agent",
        router,
        {
            "fulfillment_agent": "fulfillment_agent",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "fulfillment_agent",
        router,
        {
            END: END
        }
    )

    return workflow.compile()
