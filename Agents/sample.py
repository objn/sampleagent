"""Support ticket workflow with optional, situation-driven tool calls."""

import json
import os
from typing import Annotated, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode
from pydantic import BaseModel, Field

from Models.chat_history import recent_customer_messages, save_chat_message
from Prompts.classify import classify_prompt
from Prompts.extract import extract_prompt
from Prompts.guardrail import guardrail_prompt
from Prompts.history import history_relevance_prompt
from Prompts.plan import plan_prompt
from Prompts.respond import decide_prompt, respond_prompt
from RAG.knowledge import search_knowledge


class TicketState(MessagesState, total=False):
    user_id: str
    session_id: str
    urgency: str
    urgency_reason: str
    needs_human_review: bool
    product: str
    issue_type: str
    sentiment: Literal["Positive", "Neutral", "Negative"]
    issue_summary: str
    action: str
    decision_reason: str
    specialist_team: str
    draft_response: str
    response_text: str
    specialist_chat_session: dict
    mock_email: dict
    status: str
    relevant_customer_history: dict[str, list[dict]]
    guardrail_category: Literal["safe", "prompt_injection", "nsfw"]
    guardrail_language: Literal["th", "en"]


class UrgencyResult(BaseModel):
    urgency: Literal["critical", "high", "medium", "low"]
    reason: str
    needs_human_review: bool


class TicketDetails(BaseModel):
    product: str = Field(description="Product name, or 'unknown' if absent")
    issue_type: Literal["billing", "technical", "account", "feature_request", "other"]
    sentiment: Literal["Positive", "Neutral", "Negative"]
    summary: str


class TicketDecision(BaseModel):
    action: Literal["auto_respond", "route_specialist", "escalate_human"]
    reason: str
    specialist_team: str = Field(description="Team name, or empty string if not routing")


class HistoryRelevance(BaseModel):
    relevant_ids: list[int] = Field(description="Database IDs of messages relevant to the latest customer message")


class GuardrailResult(BaseModel):
    category: Literal["safe", "prompt_injection", "nsfw"]
    language: Literal["th", "en"]


@tool
def search_uploaded_knowledge(query: str, category: str | None = None) -> str:
    """Search uploaded support documents when a product answer or policy needs evidence."""
    matches = search_knowledge(query, category=category)
    if not matches:
        return "No uploaded documents matched the question."
    return "\n\n".join(
        f"Source: {match['filename']} (category: {match['category']})"
        + (f", page {match['page_number']}" if match["page_number"] else "")
        + f"\n{match['content']}"
        for match in matches
    )


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
    history = recent_customer_messages(
        user_id,
        limit=10,
        exclude_session_id=session_id if latest_user and latest_user.id else None,
        exclude_message_id=latest_user.id if latest_user else None,
    )
    if not history:
        return "No earlier customer chat messages were found."
    return json.dumps(history, ensure_ascii=False)


TOOLS = [search_uploaded_knowledge, check_customer_history]


def _model() -> ChatOpenAI:
    return ChatOpenAI(model=os.getenv("TICKET_MODEL", "gpt-4.1-mini"))


def _ticket_text(state: TicketState) -> str:
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        raise ValueError("A ticket needs at least one human message")
    return "\n".join(str(m.content) for m in user_messages)


def _latest_user_index(state: TicketState) -> int:
    return max(
        i for i, message in enumerate(state["messages"])
        if isinstance(message, HumanMessage)
    )


def _visible_tool_messages(state: TicketState) -> list:
    """Hide unreviewed history from every downstream model prompt."""
    reviewed = state.get("relevant_customer_history", {})
    visible = []
    for message in state["messages"][_latest_user_index(state) + 1:]:
        if isinstance(message, ToolMessage) and message.name == "check_customer_history":
            history = reviewed.get(message.tool_call_id, [])
            content = (
                json.dumps(history, ensure_ascii=False)
                if history else "No relevant customer history for the latest message."
            )
            visible.append(message.model_copy(update={"content": content}))
        else:
            visible.append(message)
    return visible


def _action_id(state: TicketState, kind: str) -> str:
    latest_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if state.get("user_id") and state.get("session_id") and latest_user and latest_user.id:
        value = f"{state['user_id']}:{state['session_id']}:{latest_user.id}:{kind}"
        return str(uuid5(NAMESPACE_URL, value))
    return str(uuid4())


def _tool_results(state: TicketState) -> str:
    results = [
        {"tool": m.name, "content": m.content, "status": m.status}
        for m in _visible_tool_messages(state)
        if isinstance(m, ToolMessage)
    ]
    return json.dumps(results, ensure_ascii=False) if results else "No tools were used."


def record_user_message(state: TicketState, config: RunnableConfig) -> dict:
    """Persist user messages, including message-only LangGraph Studio runs."""
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_messages:
        raise ValueError("A ticket needs at least one human message")
    configurable = config.get("configurable", {})
    session_id = (
        state.get("session_id")
        or configurable.get("thread_id")
        or f"session:{user_messages[0].id or uuid4()}"
    )
    user_id = state.get("user_id") or configurable.get("user_id") or f"guest:{session_id}"
    session_id, user_id = str(session_id), str(user_id)
    for message in user_messages:
        save_chat_message(
            user_id=user_id,
            session_id=session_id,
            message_id=message.id,
            role="user",
            content=str(message.content),
        )
    return {"user_id": user_id, "session_id": session_id, "relevant_customer_history": {}}


def check_guardrail(state: TicketState) -> dict:
    """Inspect the newest user message before any ticket processing."""
    latest_user = state["messages"][_latest_user_index(state)]
    result = (guardrail_prompt | _model().with_structured_output(GuardrailResult)).invoke({
        "latest_message": str(latest_user.content),
    })
    return {"guardrail_category": result.category, "guardrail_language": result.language}


def route_after_guardrail(state: TicketState) -> str:
    return "classify" if state["guardrail_category"] == "safe" else "reject_request"


def reject_request(state: TicketState) -> dict:
    """Use fixed replies so unsafe input cannot steer the refusal text."""
    category = state["guardrail_category"]
    language = state["guardrail_language"]
    responses = {
        "prompt_injection": {
            "th": "ขออภัย เราไม่สามารถช่วยทำตามคำขอนี้ได้ เพราะข้อความมีคำสั่งให้เปลี่ยนหรือข้ามแนวทางการทำงานของระบบ กรุณาส่งคำถามเกี่ยวกับปัญหาที่ต้องการให้ช่วยโดยตรง",
            "en": "Sorry, I can't help with that request because it attempts to change or bypass this support system's instructions. Please describe the support issue you need help with directly.",
        },
        "nsfw": {
            "th": "ขออภัย เราไม่สามารถช่วยในเรื่องนี้ได้ เพราะข้อความมีเนื้อหาทางเพศที่โจ่งแจ้งหรือความรุนแรงที่มีรายละเอียดชัดเจน กรุณาส่งคำถามเกี่ยวกับการใช้งานหรือปัญหาที่ต้องการให้ช่วยโดยไม่ใส่รายละเอียดดังกล่าว",
            "en": "Sorry, I can't help with that request because it contains explicit sexual or graphic violent content. Please describe your support issue without those details.",
        },
    }
    return {
        "action": "guardrail_rejected",
        "decision_reason": category,
        "response_text": responses[category][language],
        "status": "guardrail_rejected",
    }


def classify_ticket(state: TicketState) -> dict:
    result = (classify_prompt | _model().with_structured_output(UrgencyResult)).invoke(
        {"text": _ticket_text(state)}
    )
    return {
        "urgency": result.urgency,
        "urgency_reason": result.reason,
        "needs_human_review": result.needs_human_review,
    }


def extract_details(state: TicketState) -> dict:
    result = (extract_prompt | _model().with_structured_output(TicketDetails)).invoke(
        {"text": _ticket_text(state)}
    )
    return {
        "product": result.product,
        "issue_type": result.issue_type,
        "sentiment": result.sentiment,
        "issue_summary": result.summary,
    }


def assess_history_relevance(state: TicketState) -> dict:
    """Keep only prior messages related to the latest customer question."""
    tool_message = state["messages"][-1]
    if not isinstance(tool_message, ToolMessage) or tool_message.name != "check_customer_history":
        raise ValueError("Expected a customer history tool result")

    try:
        history = json.loads(str(tool_message.content))
    except (TypeError, ValueError):
        history = []
    if not isinstance(history, list):
        history = []
    valid_history = [row for row in history if isinstance(row, dict) and isinstance(row.get("id"), int)]
    selected = []
    if valid_history:
        latest_user = state["messages"][_latest_user_index(state)]
        result = (history_relevance_prompt | _model().with_structured_output(HistoryRelevance)).invoke({
            "latest_message": str(latest_user.content),
            "product": state["product"],
            "issue_type": state["issue_type"],
            "history": json.dumps(valid_history, ensure_ascii=False),
        })
        relevant_ids = set(result.relevant_ids)
        selected = [row for row in valid_history if row["id"] in relevant_ids]
    reviewed = dict(state.get("relevant_customer_history", {}))
    reviewed[tool_message.tool_call_id] = selected
    return {"relevant_customer_history": reviewed}


def plan_next_step(state: TicketState) -> dict:
    # At most two tool calls. Each is optional and chosen by the model.
    tool_count = sum(isinstance(m, ToolMessage) for m in _visible_tool_messages(state))
    model = _model()
    if tool_count < 2:
        model = model.bind_tools(TOOLS, parallel_tool_calls=False)
    prompt = plan_prompt.invoke({
        "ticket": _ticket_text(state),
        "urgency": state["urgency"],
        "product": state["product"],
        "issue_type": state["issue_type"],
        "sentiment": state["sentiment"],
        "summary": state["issue_summary"],
        "history_available": bool(state.get("user_id") and state.get("session_id")),
        "tool_history": _visible_tool_messages(state),
    })
    return {"messages": [model.invoke(prompt)]}


def route_after_plan(state: TicketState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return {
            "search_uploaded_knowledge": "knowledge_base",
            "check_customer_history": "customer_history",
        }[last.tool_calls[0]["name"]]
    return "decide"


def decide_action(state: TicketState) -> dict:
    result = (decide_prompt | _model().with_structured_output(TicketDecision)).invoke({
        "ticket": _ticket_text(state),
        "urgency": state["urgency"],
        "urgency_reason": state["urgency_reason"],
        "needs_human_review": state["needs_human_review"],
        "product": state["product"],
        "issue_type": state["issue_type"],
        "sentiment": state["sentiment"],
        "summary": state["issue_summary"],
        "tool_results": _tool_results(state),
    })
    action = result.action
    reason = result.reason
    if state["urgency"] == "critical" or state["needs_human_review"]:
        action = "escalate_human"
        reason = "Critical urgency or uncertain risk requires human review. " + reason
    elif state["urgency"] == "high" and action == "auto_respond":
        action = "route_specialist"
        reason = "High urgency requires a specialist. " + reason
    elif action == "auto_respond" and state["issue_type"] != "other":
        has_knowledge = any(
            isinstance(m, ToolMessage)
            and m.name == "search_uploaded_knowledge"
            and m.status == "success"
            and "No uploaded documents matched" not in str(m.content)
            for m in state["messages"]
        )
        if not has_knowledge:
            action = "route_specialist"
            reason = "No matching knowledge source supports an automatic answer. " + reason
    default_team = {
        "billing": "billing",
        "technical": "technical",
        "account": "account",
        "feature_request": "product",
        "other": "support",
    }[state["issue_type"]]
    proposed_team = result.specialist_team.strip().lower()
    team = proposed_team if proposed_team in {
        "billing", "technical", "account", "product", "support"
    } else default_team
    return {
        "action": action,
        "decision_reason": reason,
        "specialist_team": team if action != "auto_respond" else "",
    }


def draft_auto_response(state: TicketState) -> dict:
    response = (respond_prompt | _model()).invoke({
        "ticket": _ticket_text(state),
        "summary": state["issue_summary"],
        "tool_results": _tool_results(state),
    })
    text = str(response.content)
    return {
        "draft_response": text,
        "response_text": text,
        "status": "answered",
    }


def mark_specialist_route(state: TicketState) -> dict:
    team = state["specialist_team"]
    chat_id = _action_id(state, "specialist_chat")
    agent_id = f"mock-{team}-agent-01"
    return {
        "specialist_chat_session": {
            "id": chat_id,
            "team": team,
            "assigned_agent_id": agent_id,
            "status": "simulated_open",
        },
        "response_text": (
            f"[Mockup] เราเปิดห้องแชตหมายเลข {chat_id} ให้ทีม {team} แล้ว "
            f"และมอบหมายให้เจ้าหน้าที่ {agent_id} ดูแลเรื่องนี้ เจ้าหน้าที่จะติดต่อคุณทางห้องแชต"
        ),
        "status": "mock_specialist_session_opened",
    }


def mark_human_escalation(state: TicketState) -> dict:
    team = state["specialist_team"]
    email_id = _action_id(state, "escalation_email")
    return {
        "mock_email": {
            "id": email_id,
            "to_department": team,
            "subject": f"Support ticket escalation: {state['issue_summary']}",
            "status": "simulated_sent",
        },
        "response_text": (
            f"[Mockup] เราได้ส่งอีเมลแจ้งฝ่าย {team} ให้ตรวจสอบเรื่องนี้แล้ว "
            "เจ้าหน้าที่จะติดต่อกลับโดยเร็วที่สุด"
        ),
        "status": "mock_escalation_sent",
    }


def record_agent_response(state: TicketState) -> dict:
    """Expose the final reply as an AI message and keep it in chat history."""
    response = AIMessage(
        content=state["response_text"],
        id=_action_id(state, f"reply:{state['action']}"),
    )
    save_chat_message(
        user_id=state["user_id"],
        session_id=state["session_id"],
        message_id=response.id,
        role="agent",
        content=state["response_text"],
    )
    return {"messages": [response]}


builder = StateGraph(TicketState)
builder.add_node("record_user_message", record_user_message)
builder.add_node("guardrail", check_guardrail)
builder.add_node("reject_request", reject_request)
builder.add_node("classify", classify_ticket)
builder.add_node("extract", extract_details)
builder.add_node("plan", plan_next_step)
builder.add_node("knowledge_base", ToolNode([search_uploaded_knowledge]))
builder.add_node("customer_history", ToolNode([check_customer_history]))
builder.add_node("assess_history_relevance", assess_history_relevance)
builder.add_node("decide", decide_action)
builder.add_node("auto_respond", draft_auto_response)
builder.add_node("route_specialist", mark_specialist_route)
builder.add_node("escalate_human", mark_human_escalation)
builder.add_node("record_agent_response", record_agent_response)

builder.add_edge(START, "record_user_message")
builder.add_edge("record_user_message", "guardrail")
builder.add_conditional_edges("guardrail", route_after_guardrail, {
    "classify": "classify",
    "reject_request": "reject_request",
})
builder.add_edge("reject_request", "record_agent_response")
builder.add_edge("classify", "extract")
builder.add_edge("extract", "plan")
builder.add_conditional_edges("plan", route_after_plan, {
    "knowledge_base": "knowledge_base",
    "customer_history": "customer_history",
    "decide": "decide",
})
builder.add_edge("knowledge_base", "plan")
builder.add_edge("customer_history", "assess_history_relevance")
builder.add_edge("assess_history_relevance", "plan")
builder.add_conditional_edges("decide", lambda state: state["action"], {
    "auto_respond": "auto_respond",
    "route_specialist": "route_specialist",
    "escalate_human": "escalate_human",
})
builder.add_edge("auto_respond", "record_agent_response")
builder.add_edge("route_specialist", "record_agent_response")
builder.add_edge("escalate_human", "record_agent_response")
builder.add_edge("record_agent_response", END)

agent = builder.compile()
