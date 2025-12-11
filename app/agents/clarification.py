from typing import Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore

from app.state import AgentState


async def clarification_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    """
    Clarification agent that asks the user for missing information.
    
    The clarification question is already in the last message from context_agent.
    This agent simply returns that question and routes back to context_agent
    so that when the user responds, context_agent can process the new information.
    """
    print("Clarification Agent")
    messages = state["messages"]

    # The clarification question is in the last message from context_agent
    clarification_question = messages[-1].content if messages else "I need more information to help you."

    # Return the clarification question
    # The graph will route back to context_agent after this
    # When user sends next message, context_agent will process it and update the query
    return {
        "messages": [AIMessage(content=clarification_question)],
        "current_agent": "clarification_agent",
        "next_step": "END",  # End here, wait for user's next message
    }
