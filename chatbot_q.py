from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
import time


model = ChatOllama(
    model="gemma3:4b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# Only user messages are stored
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

    start_time = time.time()

    # Format previous user messages
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

    response_start = time.time()

    response = model.invoke([
        SystemMessage(content=prompt)
    ])

    response_end = time.time()

    # Store ONLY current user message
    user_memory.append(user_input)

    print(f"\nAI: {response.content}\n")

    total_time = time.time() - start_time
    response_time = response_end - response_start

    print(f"Stored user messages: {len(user_memory)}")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Response generation time: {response_time:.4f} seconds")
    print()