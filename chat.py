import time
from typing import List, Dict, Generator

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from config import RECENT_MESSAGES
from models.llm import get_model
from memory.conversation_memory import update_summary


model = get_model()


def build_messages(
    user_input: str,
    summary: str,
    recent_messages: List[Dict[str, str]],
):
    """
    Build the conversation context sent to the LLM.
    """

    system_prompt = f"""
You are a highly knowledgeable, direct, and natural-sounding human expert participating in a live voice conversation.

IDENTITY & TONE:
- You are speaking over a live audio call.
- Your tone is conversational, confident, and warm.
- Never refer to yourself as an AI, a language model, or an assistant. Speak exactly as a knowledgeable human would.
- Avoid robotic filler phrases (e.g., "That's fascinating", "I'd be happy to help", "Here is the information"). Just answer the prompt directly.

VOICE FORMATTING (CRITICAL):
- Output plain text only. Your content is being read aloud by a text-to-speech engine.
- NO MARKDOWN: Never use asterisks, hashtags, bolding, or bullet points.
- NO EMOJIS: Do not use any emojis or special characters.
- KEEP IT BRIEF: Limit responses to 1 to 3 short sentences. Humans do not speak in long monologues.
- READABILITY: Spell out numbers, symbols, and dates exactly as they are spoken (e.g., "one hundred dollars" instead of "$100").

MEMORY & CONTEXT:
- Known Information: {summary}
- Treat the known information as established context. Never ask for this information again or act surprised by it.
- If the user asks about their projects or history, reference the known information directly as shared context.

CONVERSATION DYNAMICS:
- Answer the core of the user's request immediately.
- Do not repeat the user's question back to them.
- Do not end your turn with a question unless you genuinely need a specific piece of missing information to proceed.
- The user's input comes from Speech-to-Text. Silently ignore obvious transcription errors and infer the intended meaning. Never mention the errors.
"""

    messages = [
        SystemMessage(content=system_prompt)
    ]

    for message in recent_messages[-RECENT_MESSAGES:]:
        if message["role"] == "user":
            messages.append(
                HumanMessage(content=message["content"])
            )

        elif message["role"] == "assistant":
            messages.append(
                AIMessage(content=message["content"])
            )

    messages.append(
        HumanMessage(content=user_input)
    )

    return messages


def chat_stream(
    user_input: str,
    summary: str,
    recent_messages: List[Dict[str, str]],
) -> Generator[str, None, None]:
    """
    Stream the LLM response chunk by chunk.

    The caller can send each chunk directly to the TTS system.
    """

    messages = build_messages(
        user_input,
        summary,
        recent_messages,
    )

    start_time = time.time()

    full_response = ""

    for chunk in model.stream(messages):

        text = chunk.content

        if not text:
            continue

        full_response += text

        yield text

    response_time = time.time() - start_time

    print(
        f"\n[LLM response time: {response_time:.2f}s]"
    )


def update_memory(
    user_input: str,
    response: str,
    summary: str,
    recent_messages: List[Dict[str, str]],
):
    """
    Update conversation memory after the assistant finishes responding.
    """

    # Add user message
    recent_messages.append({
        "role": "user",
        "content": user_input,
    })

    # Add assistant message
    recent_messages.append({
        "role": "assistant",
        "content": response,
    })

    # Update summary if enough messages exist
    if len(recent_messages) > RECENT_MESSAGES:

        summary = update_summary(
            summary,
            recent_messages,
        )

        # Keep only the newest messages
        recent_messages = (
            recent_messages[-RECENT_MESSAGES:]
        )

    return summary, recent_messages
