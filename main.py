from graph.chat_graph import build_graph
from utils.input_handler import get_user_input


graph = build_graph()

summary = ""
recent_messages = []
total_time = 0.0


while True:

    user_input = get_user_input()

    if user_input.lower() == "quit":
        break

    if not user_input:
        continue

    result = graph.invoke({
        "user_input": user_input,
        "summary": summary,
        "recent_messages": recent_messages,
        "response": "",
        "response_time": 0.0,
        "total_time": total_time
    })

    summary = result["summary"]
    recent_messages = result["recent_messages"]
    total_time = result["total_time"]