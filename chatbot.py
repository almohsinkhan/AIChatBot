from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
import time

model = ChatOllama(
    model="gemma3:4b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

RECENT_TURNS = 4
RECENT_MESSAGES = RECENT_TURNS * 2


class ChatState(TypedDict):
    user_input: str
    summary: str
    recent_messages: List[Dict[str, str]]
    response: str
    response_time: float
    total_time: float


def update_summary(state: ChatState) -> str:
    messages = state["recent_messages"]

    if len(messages) <= RECENT_MESSAGES:
        return state["summary"]

    messages_to_summarize = messages[:-RECENT_MESSAGES]

    old_conversation = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in messages_to_summarize
    )

    summary_prompt = f"""
Update the conversation memory summary.

An existing summary may already contain information
from older conversation. Combine it with the newly
older conversation.

Preserve important:
- user facts
- preferences
- names
- numbers
- projects
- decisions
- corrections
- important technical details
- important conversation context

Remove unnecessary conversational details.

Do not invent information.

EXISTING SUMMARY:
--- BEGIN EXISTING SUMMARY ---
{state["summary"]}
--- END EXISTING SUMMARY ---

NEW OLDER CONVERSATION:
--- BEGIN NEW CONVERSATION ---
{old_conversation}
--- END NEW CONVERSATION ---

Return only the updated summary.
"""

    summary_response = model.invoke([
        SystemMessage(content=summary_prompt)
    ])

    return summary_response.content


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

    state["summary"] = update_summary(state)

    state["recent_messages"] = (
        state["recent_messages"][-RECENT_MESSAGES:]
    )

    end_time = time.time()

    state["response_time"] = end_time - start_time
    state["total_time"] += state["response_time"]
    state["response"] = response_text

    return state


graph_builder = StateGraph(ChatState)

graph_builder.add_node("chat", chat_node)

graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)

graph = graph_builder.compile()


summary = ""
recent_messages = []
total_time = 0.0


while True:

    print("You (type END on a new line to send):")

    lines = []

    while True:
        line = input()

        if line.strip() == "END":
            break

        lines.append(line)

    user_input = "\n".join(lines).strip()

    if user_input.lower() == "quit":
        break

    if not user_input:
        continue

    result = graph.invoke({
        "user_input": user_input,
        "summary": summary,
        "recent_messages": recent_messages,
        "response": "",
        "response_time": 0.0,
        "total_time": total_time
    })

    print()

    summary = result["summary"]
    recent_messages = result["recent_messages"]
    total_time = result["total_time"]