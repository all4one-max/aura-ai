"""
Utility to sync agent state to AgentStateTable after each agent completes.
This replaces LangGraph's checkpoint system with our own state management.
"""
from typing import Optional
from app.state import AgentState
from app.dao.agent_state_dao import sync_agent_state_from_checkpoint


async def sync_state_after_agent(
    state: AgentState,
    thread_id: str,
    user_id: str,
    request_id: Optional[str] = None,
) -> None:
    """
    Sync agent state to AgentStateTable after an agent completes.
    This is called after each agent step to persist state.
    
    Args:
        state: Current AgentState from the agent
        thread_id: Chat/conversation identifier
        user_id: User identifier
        request_id: Optional request identifier
    """
    try:
        await sync_agent_state_from_checkpoint(
            thread_id=thread_id,
            user_id=user_id,
            state=state,
            request_id=request_id,
        )
    except Exception as e:
        # Don't fail the request if state sync fails
        print(f"⚠️  Warning: Failed to sync agent state: {e}")
        import traceback
        traceback.print_exc()

