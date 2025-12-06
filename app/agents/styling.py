from app.state import AgentState
from langchain_core.messages import AIMessage

def styling_agent(state: AgentState):
    """
    Visualizes the product on the user.
    """
    # Mock selection of the first item
    search_results = state.get("search_results", [])
    if search_results:
        selected_item = search_results[0]
        return {
            "messages": [AIMessage(content=f"Here is how the {selected_item['name']} looks on you.")],
            "selected_item": selected_item,
            "next_step": "fulfillment_agent"
        }
    
    return {
        "messages": [AIMessage(content="I couldn't find any items to style.")],
        "next_step": "END"
    }
