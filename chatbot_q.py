from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import time


# Initialize Ollama
model = ChatOllama(
    model="gemma3:4b",
    temperature=0.7,
    base_url="http://localhost:11434"
)


# Store ONLY previous user messages
user_memory = []


while True:

    user_input = input("You: ").strip()

    if user_input.lower() == "quit":
        break

    start_time = time.time()

   

    if user_memory:
        previous_questions = "\n".join(
            f"- {question}"
            for question in user_memory
        )
    else:
        previous_questions = "None"


    prompt = f"""
You are a helpful and witty AI assistant.

The user previously asked or told you the following:

{previous_questions}

Now answer the user's current message:

{user_input}

Use the previous user messages only when they are relevant
to the current message.

Do not invent information.
"""


    response_start = time.time()

    response = model.invoke([
        SystemMessage(content=prompt)
    ])

    response_end = time.time()


    user_memory.append(user_input)

    print(f"AI: {response.content}")

    total_time = time.time() - start_time
    response_time = response_end - response_start

    print(f"\nPrevious user messages stored: {len(user_memory)}")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Response generation time: {response_time:.4f} seconds")
    print()
