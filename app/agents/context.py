from app.state import AgentState
from langchain_core.messages import AIMessage

def context_agent(state: AgentState):
    """
    Analyzes user intent and profile.
    For this scaffold, it just acknowledges the user and passes to Research.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Mock logic: If no user profile, set one.
    if not state.get("user_profile"):
        user_profile = {
            "body_type": "Hourglass",
            "skin_tone": "Cool Winter",
            "preferences": ["Blue", "Silk"],
            "sizes": {"US": "6"}
        }
        return {
            "messages": [AIMessage(content="I've analyzed your profile. You have a Cool Winter undertone.")],
            "user_profile": user_profile,
            "next_step": "research_agent"
        }
    
    return {
        "messages": [AIMessage(content="Context analyzed.")],
        "next_step": "research_agent"
    }
