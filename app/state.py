from typing import TypedDict, List, Optional, Any
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

class UserProfile(TypedDict):
    body_type: Optional[str]
    skin_tone: Optional[str]
    preferences: List[str]
    sizes: Optional[dict]

class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_profile: Optional[UserProfile]
    search_results: List[Any]
    selected_item: Optional[Any]
    next_step: Optional[str]
    current_agent: Optional[str]
