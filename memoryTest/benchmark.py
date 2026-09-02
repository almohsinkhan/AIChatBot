from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import time
import csv
import re
import statistics


MODEL_NAME = "gemma3:4b"
BASE_URL = "http://localhost:11434"
RUNS_PER_TEST = 3

model = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    base_url=BASE_URL
)


SYSTEM_PROMPT = """You are a helpful and witty AI assistant.
Use the conversation history and any provided memory context
to answer the user's question directly."""


def normalize(text):
    text = text.lower()
    return re.sub(r"[^\w\s]", " ", text)


def score_answer(answer, test):
    raw = answer.lower()
    normalized = normalize(answer)
    expected = test["expected"].lower()

    if test["name"] == "Project Name":
        compact = re.sub(r"[^a-z0-9]", "", raw)

        return (
            "starart" in compact
            or bool(
                re.search(
                    r"\bstar\s+art\b",
                    raw
                )
            )
        )

    if test["name"] == "Multiple Numbers":

        has_desktop_32 = bool(
            re.search(
                r"\bdesktop\b.{0,25}\b32\b",
                normalized
            )
            or
            re.search(
                r"\b32\b.{0,25}\bdesktop\b",
                normalized
            )
        )

        claims_desktop_16 = bool(
            re.search(
                r"\bdesktop\b\s+"
                r"(?:has|with|contains|is|features)\s+16\b",
                normalized
            )
            or
            re.search(
                r"\b16\s*(?:gb)?\s*"
                r"(?:in|on|for|of)\s+"
                r"(?:the\s+)?desktop\b",
                normalized
            )
        )

        return (
            has_desktop_32
            and not claims_desktop_16
        )

    if test["name"] == "Preference Update":

        has_rust = bool(
            re.search(
                r"\brust\b",
                normalized
            )
        )

        if not has_rust:
            return False

        claims_python_current = bool(
            re.search(
                r"\bpython\b\s+is\s+"
                r"(?:now\s+|currently\s+)?"
                r"(?:my|your)\s+"
                r"(?:favorite|preferred)\b",
                normalized
            )
            or
            re.search(
                r"\b(?:my|your)\s+"
                r"(?:current\s+)?"
                r"(?:favorite|preferred)"
                r"(?:\s+\w+)?\s+"
                r"is\s+python\b",
                normalized
            )
        )

        return not claims_python_current

    if test["name"] == "Entity Association":

        has_john = bool(
            re.search(
                r"\bjohn\b",
                normalized
            )
        )

        claims_sarah_migration = bool(
            re.search(
                r"\bsarah\b\s+"
                r"(?:is\s+)?"
                r"(?:working\s+on|handling|leading|doing)\s+"
                r"(?:the\s+)?"
                r"(?:database\s+)?migration\b",
                normalized
            )
        )

        return (
            has_john
            and not claims_sarah_migration
        )

    return bool(
        re.search(
            rf"\b{re.escape(expected)}\b",
            normalized
        )
    )


def get_token_counts(response):

    metadata = response.response_metadata or {}

    prompt_tokens = metadata.get(
        "prompt_eval_count",
        0
    )

    output_tokens = metadata.get(
        "eval_count",
        0
    )

    return prompt_tokens, output_tokens


def format_full_history(history):

    return "\n".join(
        f"{'User' if role == 'human' else 'Assistant'}: {content}"
        for role, content in history
    )


def format_user_history(history):

    return "\n".join(
        f"- {content}"
        for role, content in history
        if role == "human"
    )


def make_distraction_history(count):

    topics = [
        (
            "Can you explain how neural networks learn?",
            "Neural networks learn by adjusting their parameters based on training data."
        ),
        (
            "What is the difference between Python and C++?",
            "Python is generally higher level, while C++ provides lower-level control."
        ),
        (
            "How does a database index work?",
            "An index helps a database locate records more efficiently."
        ),
        (
            "What is overfitting?",
            "Overfitting occurs when a model learns training data too closely."
        ),
        (
            "Explain gradient descent.",
            "Gradient descent is an optimization method used to minimize a loss function."
        ),
        (
            "What is an API?",
            "An API defines how software components communicate with each other."
        ),
        (
            "What is Docker?",
            "Docker packages applications and their dependencies into containers."
        ),
        (
            "What is the difference between RAM and storage?",
            "RAM is temporary working memory while storage retains data."
        ),
        (
            "What is Git used for?",
            "Git is a version control system used to track changes in files."
        ),
        (
            "What is an embedding?",
            "An embedding represents information as numerical vectors."
        )
    ]

    history = []

    for i in range(count):

        question, answer = topics[
            i % len(topics)
        ]

        history.append(
            ("human", question)
        )

        history.append(
            ("ai", answer)
        )

    return history


TESTS = [

    {
        "name": "Direct Recall",
        "history": [
            (
                "human",
                "My favorite programming language is Python."
            ),
            (
                "ai",
                "Got it. Your favorite programming language is Python."
            )
        ],
        "question": "What is my favorite programming language?",
        "expected": "python"
    },

    {
        "name": "Recall After Distraction",
        "history": [
            (
                "human",
                "My favorite color is blue."
            ),
            (
                "ai",
                "Got it. Your favorite color is blue."
            ),
            (
                "human",
                "Explain what a neural network is."
            ),
            (
                "ai",
                "A neural network is a machine learning model inspired by biological neurons."
            ),
            (
                "human",
                "What is the difference between AI and machine learning?"
            ),
            (
                "ai",
                "AI is the broader field, while machine learning is a subset of AI."
            )
        ],
        "question": "What is my favorite color?",
        "expected": "blue"
    },

    {
        "name": "Multiple Facts",
        "history": [
            (
                "human",
                "My name is Alex."
            ),
            (
                "ai",
                "Nice to meet you, Alex."
            ),
            (
                "human",
                "I live in Delhi."
            ),
            (
                "ai",
                "Got it, you live in Delhi."
            ),
            (
                "human",
                "I am learning machine learning."
            ),
            (
                "ai",
                "That is great. Machine learning is a useful field to learn."
            )
        ],
        "question": "Where do I live?",
        "expected": "delhi"
    },

    {
        "name": "Preference Update",
        "history": [
            (
                "human",
                "My favorite programming language is Python."
            ),
            (
                "ai",
                "Got it. Your favorite programming language is Python."
            ),
            (
                "human",
                "Actually, Rust is now my favorite programming language."
            ),
            (
                "ai",
                "Understood. Rust is now your favorite programming language."
            )
        ],
        "question": "What is my favorite programming language now?",
        "expected": "rust"
    },

    {
        "name": "Project Name",
        "history": [
            (
                "human",
                "My project is called StarArt."
            ),
            (
                "ai",
                "StarArt sounds like an interesting project."
            )
        ],
        "question": "What is the name of my project?",
        "expected": "starart"
    },

    {
        "name": "Numerical Recall",
        "history": [
            (
                "human",
                "My project currently has 37 tests."
            ),
            (
                "ai",
                "Got it. Your project currently has 37 tests."
            )
        ],
        "question": "How many tests does my project currently have?",
        "expected": "37"
    },

    {
        "name": "Multiple Numbers",
        "history": [
            (
                "human",
                "My laptop has 16 GB RAM."
            ),
            (
                "ai",
                "Got it. Your laptop has 16 GB RAM."
            ),
            (
                "human",
                "My desktop has 32 GB RAM."
            ),
            (
                "ai",
                "Understood. Your desktop has 32 GB RAM."
            )
        ],
        "question": "How much RAM does my desktop have?",
        "expected": "32"
    },

    {
        "name": "Entity Association",
        "history": [
            (
                "human",
                "Sarah is my manager."
            ),
            (
                "ai",
                "Got it. Sarah is your manager."
            ),
            (
                "human",
                "John is my teammate."
            ),
            (
                "ai",
                "Understood. John is your teammate."
            ),
            (
                "human",
                "John is working on the database migration."
            ),
            (
                "ai",
                "Got it. John is working on the database migration."
            )
        ],
        "question": "Who is working on the database migration?",
        "expected": "john"
    },

    {
        "name": "Delayed Recall",
        "history": [
            (
                "human",
                "Remember this code: 7391."
            ),
            (
                "ai",
                "I will remember the code 7391."
            ),
            (
                "human",
                "Tell me something interesting about Linux."
            ),
            (
                "ai",
                "Linux is an open-source operating system kernel."
            ),
            (
                "human",
                "Explain recursion in simple terms."
            ),
            (
                "ai",
                "Recursion is when a function calls itself to solve a smaller version of a problem."
            )
        ],
        "question": "What code did I ask you to remember?",
        "expected": "7391"
    },

    {
        "name": "Operating System",
        "history": [
            (
                "human",
                "I use Fedora Linux on my laptop."
            ),
            (
                "ai",
                "Got it. You use Fedora Linux."
            ),
            (
                "human",
                "I use Ollama for running local models."
            ),
            (
                "ai",
                "Ollama is useful for running local language models."
            )
        ],
        "question": "What operating system do I use?",
        "expected": "fedora"
    },

    {
        "name": "Long Context Recall",
        "history": (
            [
                (
                    "human",
                    "My favorite programming language is Rust."
                ),
                (
                    "ai",
                    "Got it. Your favorite programming language is Rust."
                )
            ]
            +
            make_distraction_history(10)
        ),
        "question": "What is my favorite programming language?",
        "expected": "rust"
    },

    {
        "name": "Long Context Numerical Recall",
        "history": (
            [
                (
                    "human",
                    "My project currently has 137 tests."
                ),
                (
                    "ai",
                    "Got it. Your project currently has 137 tests."
                )
            ]
            +
            make_distraction_history(15)
        ),
        "question": "How many tests does my project currently have?",
        "expected": "137"
    },

    {
        "name": "Long Context Entity Recall",
        "history": (
            [
                (
                    "human",
                    "John is responsible for the database migration."
                ),
                (
                    "ai",
                    "Understood. John is responsible for the database migration."
                )
            ]
            +
            make_distraction_history(15)
        ),
        "question": "Who is responsible for the database migration?",
        "expected": "john"
    }
]


def run_full_history(test):

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    ]

    for role, content in test["history"]:

        if role == "human":

            messages.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "ai":

            messages.append(
                AIMessage(
                    content=content
                )
            )

    messages.append(
        HumanMessage(
            content=test["question"]
        )
    )

    start = time.perf_counter()

    response = model.invoke(
        messages
    )

    end = time.perf_counter()

    response_time = end - start

    prompt_tokens, output_tokens = (
        get_token_counts(response)
    )

    return {
        "answer": response.content,
        "response_time": response_time,
        "total_time": response_time,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            prompt_tokens +
            output_tokens
        )
    }


def run_summary(test):

    history_text = format_full_history(
        test["history"]
    )

    summary_prompt = f"""
Create a concise memory summary of the conversation below.

Preserve important facts, preferences, names, numbers,
projects, decisions, corrections, and relationships.

Do not invent information.

Conversation:
--- BEGIN CONVERSATION ---
{history_text}
--- END CONVERSATION ---

Return only the memory summary.
"""

    summary_start = time.perf_counter()

    summary_response = model.invoke([
        SystemMessage(
            content="Accurately summarize the conversation."
        ),
        HumanMessage(
            content=summary_prompt
        )
    ])

    summary_end = time.perf_counter()

    summary_time = (
        summary_end -
        summary_start
    )

    summary_prompt_tokens, summary_output_tokens = (
        get_token_counts(
            summary_response
        )
    )

    summary = summary_response.content

    answer_prompt = f"""
Memory context:

--- BEGIN MEMORY SUMMARY ---
{summary}
--- END MEMORY SUMMARY ---

Current user question:
{test["question"]}
"""

    answer_start = time.perf_counter()

    response = model.invoke([
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=answer_prompt
        )
    ])

    answer_end = time.perf_counter()

    answer_time = (
        answer_end -
        answer_start
    )

    answer_prompt_tokens, answer_output_tokens = (
        get_token_counts(response)
    )

    return {
        "answer": response.content,
        "response_time": answer_time,
        "total_time": (
            summary_time +
            answer_time
        ),
        "prompt_tokens": (
            summary_prompt_tokens +
            answer_prompt_tokens
        ),
        "output_tokens": (
            summary_output_tokens +
            answer_output_tokens
        ),
        "total_tokens": (
            summary_prompt_tokens
            +
            summary_output_tokens
            +
            answer_prompt_tokens
            +
            answer_output_tokens
        )
    }


def run_user_only(test):

    user_messages = format_user_history(
        test["history"]
    )

    memory_prompt = f"""
Memory context containing previous user statements:

--- BEGIN USER STATEMENTS ---
{user_messages}
--- END USER STATEMENTS ---

Current user question:
{test["question"]}
"""

    start = time.perf_counter()

    response = model.invoke([
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=memory_prompt
        )
    ])

    end = time.perf_counter()

    response_time = end - start

    prompt_tokens, output_tokens = (
        get_token_counts(response)
    )

    return {
        "answer": response.content,
        "response_time": response_time,
        "total_time": response_time,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": (
            prompt_tokens +
            output_tokens
        )
    }


def run_user_only_summary(test):

    user_messages = format_user_history(
        test["history"]
    )

    summary_prompt = f"""
Create a concise memory summary using ONLY the previous
user statements below.

Preserve important:
- facts
- preferences
- names
- numbers
- projects
- decisions
- corrections
- relationships

Do not use information from assistant messages.
Do not invent information.

Previous user statements:
--- BEGIN USER STATEMENTS ---
{user_messages}
--- END USER STATEMENTS ---

Return only the memory summary.
"""

    summary_start = time.perf_counter()

    summary_response = model.invoke([
        SystemMessage(
            content=(
                "Accurately summarize information "
                "from user statements."
            )
        ),
        HumanMessage(
            content=summary_prompt
        )
    ])

    summary_end = time.perf_counter()

    summary_time = (
        summary_end -
        summary_start
    )

    summary_prompt_tokens, summary_output_tokens = (
        get_token_counts(
            summary_response
        )
    )

    user_summary = summary_response.content

    answer_prompt = f"""
Memory context:

--- BEGIN USER MEMORY SUMMARY ---
{user_summary}
--- END USER MEMORY SUMMARY ---

Current user question:
{test["question"]}
"""

    answer_start = time.perf_counter()

    response = model.invoke([
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content=answer_prompt
        )
    ])

    answer_end = time.perf_counter()

    answer_time = (
        answer_end -
        answer_start
    )

    answer_prompt_tokens, answer_output_tokens = (
        get_token_counts(response)
    )

    return {
        "answer": response.content,
        "response_time": answer_time,
        "total_time": (
            summary_time +
            answer_time
        ),
        "prompt_tokens": (
            summary_prompt_tokens +
            answer_prompt_tokens
        ),
        "output_tokens": (
            summary_output_tokens +
            answer_output_tokens
        ),
        "total_tokens": (
            summary_prompt_tokens
            +
            summary_output_tokens
            +
            answer_prompt_tokens
            +
            answer_output_tokens
        )
    }


METHODS = [
    ("summary", run_summary),
    ("user_only", run_user_only),
    ("user_only_summary", run_user_only_summary),
    ("full_history", run_full_history)
]


def warm_up_model():

    print("\nWarming up model...")

    start = time.perf_counter()

    model.invoke([
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        HumanMessage(
            content="Say hello."
        )
    ])

    end = time.perf_counter()

    print(
        f"Warm-up time: {end - start:.2f}s"
    )

    print(
        "Warm-up excluded from benchmark."
    )


def main():

    warm_up_model()

    print("\n" + "=" * 110)
    print("SHORT-TERM MEMORY BENCHMARK")
    print("=" * 110)

    print(f"Model: {MODEL_NAME}")
    print("Temperature: 0.2")
    print(f"Tests: {len(TESTS)}")
    print(f"Runs per test: {RUNS_PER_TEST}")

    print("\nMethods:")
    print("1. Summary")
    print("2. User Only")
    print("3. User Only Summary")
    print("4. Full History")

    print("=" * 110)

    results = []

    for test in TESTS:

        history_text = format_full_history(
            test["history"]
        )

        history_tokens_estimate = (
            len(history_text) // 4
        )

        print("\n" + "=" * 100)

        print(
            f"TEST: {test['name']}"
        )

        print(
            f"Question: {test['question']}"
        )

        print(
            f"History size estimate: "
            f"{history_tokens_estimate} tokens"
        )

        print("=" * 100)

        for method_name, method in METHODS:

            method_results = []

            for run_number in range(
                1,
                RUNS_PER_TEST + 1
            ):

                try:

                    result = method(test)

                    correct = score_answer(
                        result["answer"],
                        test
                    )

                    row = {
                        "test": test["name"],
                        "method": method_name,
                        "run": run_number,
                        "history_tokens_estimate":
                            history_tokens_estimate,
                        "correct": correct,
                        "response_time":
                            round(
                                result["response_time"],
                                4
                            ),
                        "total_time":
                            round(
                                result["total_time"],
                                4
                            ),
                        "prompt_tokens":
                            result["prompt_tokens"],
                        "output_tokens":
                            result["output_tokens"],
                        "total_tokens":
                            result["total_tokens"],
                        "answer":
                            result["answer"]
                    }

                    results.append(row)

                    method_results.append(
                        result
                    )

                    status = (
                        "PASS"
                        if correct
                        else "FAIL"
                    )

                    print(
                        f"{method_name:<24}"
                        f"Run {run_number} | "
                        f"{status:<4} | "
                        f"Response "
                        f"{result['response_time']:.3f}s | "
                        f"Total "
                        f"{result['total_time']:.3f}s | "
                        f"Tokens "
                        f"{result['total_tokens']}"
                    )

                except Exception as e:

                    print(
                        f"{method_name:<24}"
                        f"Run {run_number} | "
                        f"ERROR: {e}"
                    )

            if not method_results:
                continue

            accuracy = (
                sum(
                    score_answer(
                        r["answer"],
                        test
                    )
                    for r in method_results
                )
                /
                len(method_results)
            ) * 100

            response_times = [
                r["response_time"]
                for r in method_results
            ]

            total_times = [
                r["total_time"]
                for r in method_results
            ]

            prompt_token_values = [
                r["prompt_tokens"]
                for r in method_results
            ]

            output_token_values = [
                r["output_tokens"]
                for r in method_results
            ]

            total_token_values = [
                r["total_tokens"]
                for r in method_results
            ]

            print(
                f"  {'MEDIAN':<18}"
                f"Accuracy "
                f"{accuracy:6.1f}% | "
                f"Response "
                f"{statistics.median(response_times):7.3f}s | "
                f"Total "
                f"{statistics.median(total_times):7.3f}s | "
                f"Prompt "
                f"{statistics.median(prompt_token_values):5.0f} | "
                f"Total Tokens "
                f"{statistics.median(total_token_values):5.0f}"
            )

    with open(
        "results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "test",
            "method",
            "run",
            "history_tokens_estimate",
            "correct",
            "response_time",
            "total_time",
            "prompt_tokens",
            "output_tokens",
            "total_tokens",
            "answer"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 110)
    print("FINAL RESULTS")
    print("=" * 110)

    for method_name, _ in METHODS:

        method_results = [
            r
            for r in results
            if r["method"] == method_name
        ]

        if not method_results:
            continue

        accuracy = (
            sum(
                r["correct"]
                for r in method_results
            )
            /
            len(method_results)
        ) * 100

        avg_response_time = statistics.mean(
            r["response_time"]
            for r in method_results
        )

        avg_total_time = statistics.mean(
            r["total_time"]
            for r in method_results
        )

        avg_prompt_tokens = statistics.mean(
            r["prompt_tokens"]
            for r in method_results
        )

        avg_output_tokens = statistics.mean(
            r["output_tokens"]
            for r in method_results
        )

        avg_total_tokens = statistics.mean(
            r["total_tokens"]
            for r in method_results
        )

        print(f"\n{method_name.upper()}")

        print(
            f"Accuracy:              "
            f"{accuracy:.2f}%"
        )

        print(
            f"Average response time: "
            f"{avg_response_time:.4f}s"
        )

        print(
            f"Average total time:    "
            f"{avg_total_time:.4f}s"
        )

        print(
            f"Average prompt tokens: "
            f"{avg_prompt_tokens:.1f}"
        )

        print(
            f"Average output tokens: "
            f"{avg_output_tokens:.1f}"
        )

        print(
            f"Average total tokens:  "
            f"{avg_total_tokens:.1f}"
        )

    print("\nDetailed results:")
    print("results.csv")

    print("=" * 110)


if __name__ == "__main__":
    main()