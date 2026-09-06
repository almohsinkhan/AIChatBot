from chat import chat_stream


summary = ""

recent_messages = []


print("Testing Ollama streaming...")
print()
print("Assistant: ", end="", flush=True)


response = ""

for chunk in chat_stream(
    "Tell me briefly what Python is.",
    summary,
    recent_messages,
):

    print(chunk, end="", flush=True)

    response += chunk


print("\n")
print("=" * 50)
print("FULL RESPONSE:")
print(response)
print("=" * 50)