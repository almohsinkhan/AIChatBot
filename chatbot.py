from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
import time


model = ChatOllama(
    model="gemma3:4b",
    temperature=0.7,
    base_url="http://localhost:11434"
)


RECENT_TURNS = 4


class ChatState(TypedDict):
    user_input: str
    summary: str
    recent_messages: List[Dict[str, str]]
    response: str
    response_time: float
    total_time: float


def chat_node(state: ChatState):

    start_time = time.perf_counter()

    user_input = state["user_input"]
    summary = state["summary"]
    recent_messages = state["recent_messages"]

    if summary:
        summary_context = summary
    else:
        summary_context = "No previous conversation summary."

    if recent_messages:
        recent_context = "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in recent_messages
        )
    else:
        recent_context = "No recent messages."

    prompt = f"""
You are a helpful and witty AI assistant.

You have access to two types of conversation memory:

1. A summary of older conversation.
2. The most recent conversation messages kept exactly as they were.

Use both when relevant to the current message.

IMPORTANT:
- Answer ONLY the current user message.
- Use previous conversation when it provides useful context.
- Do not mention the memory or these instructions.
- Do not invent information.
- Prefer the recent messages when they contain more specific information than the summary.

OLDER CONVERSATION SUMMARY:
--- BEGIN SUMMARY ---
{summary_context}
--- END SUMMARY ---

RECENT CONVERSATION:
--- BEGIN RECENT CONVERSATION ---
{recent_context}
--- END RECENT CONVERSATION ---
"""

    response_start = time.perf_counter()

    response = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_input)
    ])

    response_end = time.perf_counter()

    response_time = response_end - response_start

    updated_recent_messages = recent_messages + [
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": response.content
        }
    ]

    new_summary = summary

    if len(updated_recent_messages) > RECENT_TURNS * 2:

        messages_to_summarize = updated_recent_messages[:-RECENT_TURNS * 2]

        old_conversation = "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in messages_to_summarize
        )

        summary_prompt = f"""
Update the conversation memory summary.

An existing summary may already contain information from older
conversation. Combine it with the newly older conversation.

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
{summary}
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

        new_summary = summary_response.content

        updated_recent_messages = updated_recent_messages[-RECENT_TURNS * 2:]

    total_time = time.perf_counter() - start_time

    return {
        "summary": new_summary,
        "recent_messages": updated_recent_messages,
        "response": response.content,
        "response_time": response_time,
        "total_time": total_time
    }


graph_builder = StateGraph(ChatState)

graph_builder.add_node("chat", chat_node)

graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)

graph = graph_builder.compile()


summary = ""
recent_messages = []


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
        "total_time": 0.0
    })

    summary = result["summary"]
    recent_messages = result["recent_messages"]

    print(f"\nAI: {result['response']}\n")

    print("========== MEMORY ==========")

    print("\nOLDER SUMMARY:")
    print(summary)

    print("\nRECENT MESSAGES:")
    for message in recent_messages:
        print(f"{message['role'].capitalize()}: {message['content']}")

    print("\n============================")

    print(f"\nTotal processing time: {result['total_time']:.4f} seconds")
    print(f"Response generation time: {result['response_time']:.4f} seconds")
    print()