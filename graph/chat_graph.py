from pyexpat.errors import messages
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



def chat_node(state: ChatState, runtime) -> ChatState:

    start_time = time.time()

    state["recent_messages"].append({
        "role": "user",
        "content": state["user_input"]
    })

    system_prompt = f"""
    You are generating responses for a live voice conversation.

    Your response will be converted directly into speech, so write as a natural person would speak.

    You will be given:
    - A conversation summary containing older context.
    - Recent conversation messages containing the latest context.

    Use both the summary and recent messages to understand the conversation and maintain continuity.

    Voice response rules:
    - Respond naturally and conversationally.
    - Keep responses concise and easy to listen to.
    - Do not use markdown, bullets, stars, emojis, or other formatting.
    - Do not write things that are meant to be visually read, such as headings or lists.
    - Avoid unnecessary repetition.
    - Do not sound robotic or overly formal.
    - Use short, natural sentences.
    - Continue naturally from the user's last message rather than restarting the conversation.
    - If the user's speech-to-text contains spelling mistakes, transcription errors, or missing words, infer the intended meaning from the conversation context.
    - Do not mention or point out speech-to-text errors unless clarification is actually necessary.
    - If the user's request is unclear, ask a short clarification question instead of guessing.

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

    response = model.invoke(messages)
    response_text = response.content.strip()




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