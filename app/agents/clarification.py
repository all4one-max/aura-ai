from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.stores import BaseStore
from langgraph.types import interrupt

from app.state import AgentState


def clarification_agent(
    state: AgentState, config: RunnableConfig, *, store: Optional[BaseStore] = None
):
    messages = state["messages"]

    clarification_question = messages[-1].content

    data = interrupt(
        {
            "messages": [AIMessage(content=clarification_question)],
            "current_agent": "clarification_agent",
        }
    )
    # append last human message with the new clarification information
    # concatenate all human messages

    """
    0 =
HumanMessage(content='Recommend me some shirts', additional_kwargs={}, response_metadata={}, id='f2b19c51-66c3-4f0c-ae21-eb671d4e79d5')
1 =
AIMessage(content='To help you find the perfect shirts, could you please specify the occasion or context for which you need them?', additional_kwargs={}, response_metadata={}, id='12ec5946-7bbc-4cc5-b048-230cfe9cfd04')
2 =
HumanMessage(content='Recommend me some shirts Recommend me some shirts', additional_kwargs={}, response_metadata={}, id='5e1eb444-b2a6-4639-b6e6-6253f2388fcc')
3 =
AIMessage(content='To help you find the perfect shirts, could you please specify the occasion or context for which you need them?', additional_kwargs={}, response_metadata={}, id='11e78280-d178-429f-9bca-51001dc94644')
4 =
HumanMessage(content='Recommend me some shirts Recommend me some shirts Recommend me some shirts the occasion is beach wedding', additional_kwargs={}, response_metadata={}, id='243b9645-10f5-4410-aaac-60b880efbc0f')
len() =
5

basically we do need to fix this concatenation logic may be will put it in context agent
    """
    final_data = ""
    for message in messages:
        if message.type == "human":
            final_data += message.content + " "
    final_data += data.content
    return {
        "messages": [HumanMessage(content=final_data)],
        "current_agent": "clarification_agent",
    }
