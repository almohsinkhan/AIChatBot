from chat import chat_stream, update_memory
from voice import (
    listen,
    StreamingTTS,
    start_tts,
    wait_for_tts,
    stop_tts,
)


summary = ""
recent_messages = []


start_tts()

try:

    while True:

        # -----------------------------
        # Listen to user
        # -----------------------------

        user_input = listen()

        if not user_input:
            continue

        if user_input.lower().strip() == "quit":
            break

        # -----------------------------
        # Stream LLM + TTS
        # -----------------------------

        print("\nAssistant: ", end="", flush=True)

        tts = StreamingTTS()

        response = ""

        for chunk in chat_stream(
            user_input,
            summary,
            recent_messages,
        ):

            print(chunk, end="", flush=True)

            response += chunk

            # Send each LLM chunk to TTS
            tts.add_chunk(chunk)

        print()

        # Speak any remaining text
        tts.finish()

        # Wait until the assistant finishes speaking
        wait_for_tts()

        # -----------------------------
        # Update conversation memory
        # -----------------------------

        summary, recent_messages = update_memory(
            user_input,
            response,
            summary,
            recent_messages,
        )

finally:

    stop_tts()