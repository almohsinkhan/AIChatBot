from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time

# Initialize Ollama
model = ChatOllama(
    model="gemma3:4b",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# Conversation history
messages = [
    SystemMessage(content="You are a helpful and witty AI assistant.")
]

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == "quit":
        break

    # Track start time for the entire loop
    start_time = time.time()
    
    # Add user message
    messages.append(HumanMessage(content=user_input))

    # Track response start time
    response_start = time.time()
    
    # Get AI response
    response = model.invoke(messages)
    
    # Track response end time
    response_end = time.time()

    # Add AI response to conversation history
    messages.append(AIMessage(content=response.content))

    # Print AI response
    print(f"AI: {response.content}")

    # Track total loop end time
    total_end = time.time()

    # Calculate times
    total_time = total_end - start_time
    response_time = response_end - response_start

    # Print results with proper formatting
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Response generation time: {response_time:.4f} seconds")

