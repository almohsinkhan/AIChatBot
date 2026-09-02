from typing import List, Dict
from langchain_core.messages import SystemMessage

from config import RECENT_MESSAGES, MAX_SUMMARY_CHARS
from models.llm import get_model


model = get_model()


def update_summary(
    summary: str,
    recent_messages: List[Dict[str, str]]
) -> str:

    if len(recent_messages) <= RECENT_MESSAGES:
        return summary

    messages_to_summarize = recent_messages[:-RECENT_MESSAGES]

    old_conversation = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in messages_to_summarize
    )

    summary_prompt = f"""
Update the conversation memory summary.

The final summary MUST stay below {MAX_SUMMARY_CHARS} characters.

An existing summary may already contain information from older
conversation. Combine it with the newly older conversation.

Preserve the most important information:

- user facts
- preferences
- names
- numbers
- projects
- decisions
- corrections
- important technical details
- important conversation context

Prioritize information that will be useful in future conversations.

Remove:

- repetition
- greetings
- unnecessary conversational details
- redundant information
- information that is no longer useful

If the information cannot fit within {MAX_SUMMARY_CHARS} characters,
keep the most important information and remove lower-priority details.

Do not invent information.
Do not change facts.
Do not exceed {MAX_SUMMARY_CHARS} characters.

EXISTING SUMMARY:
--- BEGIN EXISTING SUMMARY ---
{summary}
--- END EXISTING SUMMARY ---

NEW OLDER CONVERSATION:
--- BEGIN NEW CONVERSATION ---
{old_conversation}
--- END NEW CONVERSATION ---

Return ONLY the updated summary.
"""

    response = model.invoke([
        SystemMessage(content=summary_prompt)
    ])

    return response.content