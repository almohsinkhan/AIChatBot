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
You are maintaining long-term memory for a personal AI voice assistant.

Your job is to update the existing conversation summary using the older
conversation provided below.

The summary is persistent memory. It will be given to the assistant in
future conversations, so preserve information that may be useful later.

IMPORTANT:
- Do not summarize merely by shortening the conversation.
- Extract and preserve meaningful facts and context.
- Never remove an important fact just because it was mentioned only once.
- Do not invent, assume, or reinterpret information.
- Do not include the assistant's questions unless they contain useful context.
- Prefer factual information about the user and their ongoing work.

Preserve especially:

1. User information
   - name
   - preferences
   - goals
   - interests
   - recurring requirements

2. Projects
   - what the user is building
   - technologies being used
   - architecture
   - important implementation decisions
   - current progress

3. Important context
   - decisions already made
   - problems already solved
   - corrections
   - constraints
   - important numbers or technical details

4. Conversation state
   - unfinished tasks
   - things the user is currently working toward
   - important questions that still need an answer

Remove:

- greetings
- small talk
- repetitive statements
- unnecessary assistant responses
- temporary conversational filler

IMPORTANT RULES:

- Keep the existing summary when it contains useful information.
- Merge new information into the existing summary.
- Do not replace useful old information with a vague shorter statement.
- Do not write "The user is building something" when the actual project
  is known.
- Preserve specific technologies and project names.
- Write concise factual notes rather than prose.
- Never invent information.

The final summary MUST be below {MAX_SUMMARY_CHARS} characters.

EXISTING SUMMARY:
--- BEGIN EXISTING SUMMARY ---
{summary}
--- END EXISTING SUMMARY ---

OLDER CONVERSATION:
--- BEGIN OLDER CONVERSATION ---
{old_conversation}
--- END OLDER CONVERSATION ---

Return ONLY the updated memory summary.
"""
    
    response = model.invoke([
        SystemMessage(content=summary_prompt)
    ])

    return response.content.strip()