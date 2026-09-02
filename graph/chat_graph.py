import time
from typing import TypedDict, List, Dict

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langgraph.graph import StateGraph, START, END

from config import RECENT_MESSAGES
from models.llm import get_model
from memory.conversation_memory import update_summary


model = get_model()


class ChatState(TypedDict):
    user_input: str
    summary: str
    recent_messages: List[Dict[str, str]]
    response: str
    response_time: float
    total_time: float


def chat_node(state: ChatState) -> ChatState:

    start_time = time.time()

    state["recent_messages"].append({
        "role": "user",
        "content": state["user_input"]
    })

    system_prompt = f"""
You are a helpful AI assistant.

Follow these instructions:
- Answer the user's questions clearly and accurately.
- Use the conversation summary and recent messages as context.
- Do not invent facts.
- If you are unsure, say so.
- Maintain consistency with previous conversation.
- Follow the user's instructions and preferences when known.

Conversation summary:
{state["summary"]}
"""

    messages = [
        SystemMessage(content=system_prompt)
    ]

    for message in state["recent_messages"][-RECENT_MESSAGES:]:

        if message["role"] == "user":
            messages.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":
            messages.append(
                AIMessage(content=message["content"])
            )

    print("\nAssistant: ", end="", flush=True)

    response_text = ""

    for chunk in model.stream(messages):

        if chunk.content:
            print(chunk.content, end="", flush=True)
            response_text += chunk.content

    print()

    state["recent_messages"].append({
        "role": "assistant",
        "content": response_text
    })

    state["summary"] = update_summary(
        state["summary"],
        state["recent_messages"]
    )

    state["recent_messages"] = (
        state["recent_messages"][-RECENT_MESSAGES:]
    )

    end_time = time.time()

    state["response_time"] = end_time - start_time
    state["total_time"] += state["response_time"]
    state["response"] = response_text

    return state


def build_graph():

    graph_builder = StateGraph(ChatState)

    graph_builder.add_node("chat", chat_node)

    graph_builder.add_edge(START, "chat")
    graph_builder.add_edge("chat", END)

    return graph_builder.compile()