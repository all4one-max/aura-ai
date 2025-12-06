from app.state import AgentState
from langchain_core.messages import AIMessage

def fulfillment_agent(state: AgentState):
    """
    Handles purchase.
    """
    selected_item = state.get("selected_item")
    if selected_item:
        return {
            "messages": [AIMessage(content=f"Ordering {selected_item['name']} for ${selected_item['price']}. Order placed!")],
            "next_step": "END"
        }
        
    return {
        "messages": [AIMessage(content="Nothing to purchase.")],
        "next_step": "END"
    }
