"""Customer history lookup tool."""

import json
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from Models.chat_history import recent_customer_messages
from Tools.cache import cached_tool_result


@tool
def check_customer_history(state: Annotated[dict, InjectedState]) -> str:
    """Read the customer's last 10 earlier messages across sessions."""
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    if not user_id or not session_id:
        return "Customer history is unavailable: user_id and session_id are required."
    latest_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    exclude_session_id = session_id if latest_user and latest_user.id else None
    exclude_message_id = latest_user.id if latest_user else None

    def lookup() -> str:
        history = recent_customer_messages(
            user_id,
            limit=10,
            exclude_session_id=exclude_session_id,
            exclude_message_id=exclude_message_id,
        )
        if not history:
            return "No earlier customer chat messages were found."
        return json.dumps(history, ensure_ascii=False)

    # Without a stable message ID, a later ticket could reuse another ticket's result.
    if not exclude_message_id:
        return lookup()
    return cached_tool_result(
        "check_customer_history",
        {
            "user_id": user_id,
            "session_id": session_id,
            "exclude_session_id": exclude_session_id,
            "exclude_message_id": exclude_message_id,
            "limit": 10,
        },
        lookup,
    )
