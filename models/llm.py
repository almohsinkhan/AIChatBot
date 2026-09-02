from langchain_ollama import ChatOllama
from config import MODEL_NAME, BASE_URL, TEMPERATURE


def get_model():
    return ChatOllama(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        base_url=BASE_URL
    )