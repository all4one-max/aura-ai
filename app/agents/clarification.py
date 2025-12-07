from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore
from langgraph.types import interrupt

from app.state import AgentState


def clarification_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    print("Clarification Agent")
    messages = state["messages"]

    clarification_question = messages[-1].content

    data = interrupt(
        {
            "messages": [AIMessage(content=clarification_question)],
            "current_agent": "clarification_agent",
        }
    )

    return {
        "messages": [HumanMessage(content=data.content)],
        "current_agent": "clarification_agent",
    }
