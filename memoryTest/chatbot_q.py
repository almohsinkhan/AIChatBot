from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
import time


model = ChatOllama(
    model="gemma3:4b",
    temperature=0.7,
    base_url="http://localhost:11434"
)


class ChatState(TypedDict):
    user_input: str
    user_memory: List[str]
    response: str
    response_time: float
    total_time: float


def chat_node(state: ChatState):
    start_time = time.perf_counter()

    user_memory = state["user_memory"]
    user_input = state["user_input"]

    if user_memory:
        previous_messages = "\n".join(
            f"{i + 1}. {message}"
            for i, message in enumerate(user_memory)
        )
    else:
        previous_messages = "No previous messages."

    prompt = f"""
You are a helpful and witty AI assistant.

You have access to some previous messages written by the user.
These messages are MEMORY ONLY.

IMPORTANT:
- Do NOT answer the previous messages.
- Do NOT treat them as new questions.
- Do NOT continue answering them.
- Use them only if they provide useful context for the CURRENT message.
- Answer ONLY the CURRENT message.
- If the current message does not need previous context, ignore the memory.
- Do not mention the memory or this instruction in your answer.
- Do not invent information.

PREVIOUS USER MESSAGES:
{previous_messages}

CURRENT USER MESSAGE:
{user_input}

Now answer ONLY the current user message.
"""

    response_start = time.perf_counter()

    response = model.invoke([
        SystemMessage(content=prompt)
    ])

    response_end = time.perf_counter()

    user_memory = user_memory + [user_input]

    total_time = time.perf_counter() - start_time
    response_time = response_end - response_start

    return {
        "user_memory": user_memory,
        "response": response.content,
        "response_time": response_time,
        "total_time": total_time
    }


graph_builder = StateGraph(ChatState)

graph_builder.add_node("chat", chat_node)

graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)

graph = graph_builder.compile()


user_memory = []


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
        "user_memory": user_memory,
        "response": "",
        "response_time": 0.0,
        "total_time": 0.0
    })

    user_memory = result["user_memory"]

    print(f"\nAI: {result['response']}\n")

    print(f"Stored user messages: {len(user_memory)}")
    print(f"Total processing time: {result['total_time']:.4f} seconds")
    print(f"Response generation time: {result['response_time']:.4f} seconds")
    print()