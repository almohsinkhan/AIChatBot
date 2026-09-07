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
- Avoid robotic filler phrases (e.g., "That's fascinating", "I'd be happy to help"). Answer the prompt directly.

VOICE FORMATTING (CRITICAL):
- Output plain text only. Your content is being read aloud by a text-to-speech engine.
- NO MARKDOWN: Never use asterisks, hashtags, bolding, or bullet points.
- NO EMOJIS: Do not use any emojis or special characters.
- KEEP IT BRIEF: Limit responses to 1 to 3 short sentences. 
- READABILITY: Spell out numbers, symbols, and dates exactly as spoken (e.g., "one hundred dollars").

HANDLING INTERRUPTIONS:
- If you see "[Assistant response was interrupted by the user.]" at the end of your previous message, it means the user spoke over you.
- Do NOT apologize for being interrupted. Do NOT say "As I was saying..." or attempt to finish your previous thought.
- Immediately pivot and address the user's newest message as if the interruption was a natural part of a fast-paced human conversation.

MEMORY & CONTEXT:
- Known Information: {summary}
- Treat known information as established context. Never ask for this information again.

CONVERSATION DYNAMICS:
- Answer the core of the user's request immediately.
- Do not repeat the user's question back to them.
- Do not end your turn with a question unless you genuinely need specific missing information.
- Silently ignore obvious speech-to-text transcription errors and infer the intended meaning.
"""

    messages = [
        SystemMessage(content=system_prompt)
    ]

    for message in recent_messages:

        if message["role"] == "user":
            messages.append(
                HumanMessage(content=message["content"])
            )
        elif message["role"] == "assistant":

            content = message["content"]

            if message.get("interrupted"):
                content += "\n[Assistant response was interrupted by the user.]"

            messages.append(
                AIMessage(content=content)
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
    user_input,
    assistant_response,
    summary,
    recent_messages,
    interrupted=False,
):
    """
    Update conversation memory.

    An interrupted assistant response is marked explicitly so
    the memory system knows the response was not completed.
    """

    user_message = {
        "role": "user",
        "content": user_input,
    }

    assistant_message = {
        "role": "assistant",
        "content": assistant_response,
    }

    if interrupted:
        assistant_message["interrupted"] = True

    recent_messages.append(user_message)
    recent_messages.append(assistant_message)

    if len(recent_messages) > RECENT_MESSAGES:

        summary = update_summary(
            summary,
            recent_messages,
        )

        recent_messages = recent_messages[-RECENT_MESSAGES:]

    return summary, recent_messages
