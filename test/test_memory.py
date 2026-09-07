from chat import chat_stream, update_memory


summary = ""
recent_messages = []


def ask(question):

    global summary
    global recent_messages

    print("\nYou:", question)
    print("Assistant: ", end="", flush=True)

    response = ""

    for chunk in chat_stream(
        question,
        summary,
        recent_messages,
    ):
        print(chunk, end="", flush=True)
        response += chunk

    summary, recent_messages = update_memory(
        question,
        response,
        summary,
        recent_messages,
    )

    print("\n")
    print("-" * 60)

    print("SUMMARY:")
    print(summary)

    print("\nRECENT:")
    for message in recent_messages:
        print(
            f"{message['role']}: "
            f"{message['content']}"
        )

    print("-" * 60)


ask("My name is Mohsin.")

ask("I am building a local voice assistant.")

ask("It uses Whisper, Ollama, and Kokoro.")

ask("The assistant runs on my laptop.")

ask("What am I building?")
