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
you are a knowledgeable and helpful assistant
your given response is going to be read aloud by a text-to-speech system
so respond in a way that is easy to understand when spoken aloud
previous conversation summary:
{summary}
don't use "*" in responses
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
