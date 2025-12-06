from app.state import AgentState
from langchain_core.messages import AIMessage

def research_agent(state: AgentState):
    """
    Searches for products.
    """
    # Mock search results
    search_results = [
        {"id": 1, "name": "Blue Silk Dress", "price": 150, "url": "http://example.com/dress1"},
        {"id": 2, "name": "Navy Cocktail Dress", "price": 180, "url": "http://example.com/dress2"}
    ]
    
    return {
        "messages": [AIMessage(content="I found a few dresses that match your criteria.")],
        "search_results": search_results,
        "next_step": "styling_agent"
    }
