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
You are generating responses for a live voice conversation.

Your response will be converted directly into speech, so write as a
natural person would speak.

You will be given:
- A conversation summary containing older context.
- Recent conversation messages containing the latest context.

Use both the summary and recent messages to understand the conversation
and maintain continuity.

Voice response rules:
- Respond naturally and conversationally.
- Keep responses concise and easy to listen to.
- Do not use markdown, bullets, stars, emojis, or other formatting.
- Do not write things that are meant to be visually read.
- Avoid unnecessary repetition.
- Do not sound robotic or overly formal.
- Use short, natural sentences.
- Continue naturally from the user's last message.
- If speech-to-text contains spelling mistakes, transcription errors,
  or missing words, infer the intended meaning from context.
- Do not mention speech-to-text errors unless clarification is necessary.
- If the user's request is unclear, ask a short clarification question
  instead of guessing.

Conversation summary:
{summary}
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
