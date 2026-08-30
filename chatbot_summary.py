from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time


# Main chatbot model
model = ChatOllama(
    model="gemma3:4b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# Conversation summary
conversation_summary = ""

# Keep only recent messages
recent_messages = []

# Number of recent turns before updating summary
SUMMARY_EVERY = 4


def update_summary(old_summary, messages):
    """Create a compact summary of the conversation."""

    conversation_text = ""

    for message in messages:
        role = "User" if isinstance(message, HumanMessage) else "Assistant"
        conversation_text += f"{role}: {message.content}\n"

    prompt = f"""
You maintain short-term memory for a chatbot.

Existing summary:
{old_summary}

New conversation:
{conversation_text}

Create an updated concise summary.

Rules:
- Keep important facts the user provided.
- Keep user preferences.
- Keep project names, numbers, names, decisions, and important context.
- Keep relationships between facts.
- Remove greetings, repetition, and unnecessary explanations.
- Do not invent information.
- If information was corrected or updated, keep the newest information.
- Keep the summary short.

Return ONLY the updated summary.
"""

    response = model.invoke([
        SystemMessage(content=prompt)
    ])

    return response.content.strip()


while True:

    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        break

    start_time = time.time()

    # Add current user message
    recent_messages.append(
        HumanMessage(content=user_input)
    )

    # Build context for chatbot
    context_messages = [
        SystemMessage(
            content=f"""
You are a helpful and witty AI assistant.

Conversation memory:
{conversation_summary}

Use the memory when relevant.
Do not invent facts that are not present in the memory or conversation.
"""
        )
    ]

    # Add recent conversation
    context_messages.extend(recent_messages)

    # Generate response
    response_start = time.time()

    response = model.invoke(context_messages)

    response_end = time.time()

    # Store AI response
    recent_messages.append(
        AIMessage(content=response.content)
    )

    print(f"AI: {response.content}")

    # Update summary after several messages
    if len(recent_messages) >= SUMMARY_EVERY * 2:

        summary_start = time.time()

        conversation_summary = update_summary(
            conversation_summary,
            recent_messages
        )

        summary_end = time.time()

        # Clear old conversation
        recent_messages = []

        print("\n--- Memory Updated ---")
        print(conversation_summary)
        print(
            f"Summary generation time: "
            f"{summary_end - summary_start:.4f} seconds"
        )

    total_time = time.time() - start_time

    print(f"\nTotal processing time: {total_time:.4f} seconds")
    print(
        f"Response generation time: "
        f"{response_end - response_start:.4f} seconds"
    )
    print()
