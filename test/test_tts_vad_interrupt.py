import threading
import time

from voice import (
    start_tts,
    interrupt_tts,
    stop_tts,
    StreamingTTS,
    monitor_for_interruption,
)


start_tts()

tts = StreamingTTS()

chunks = [
    "I am going to speak for several seconds. ",
    "While I am speaking, you can interrupt me by talking. ",
    "The moment your voice is detected, the audio should stop. ",
    "This final sentence should not finish playing. ",
]

for chunk in chunks:
    print(chunk, end="", flush=True)
    tts.add_chunk(chunk)

tts.finish()

print("\n\nTTS started.")
print("Say something while the assistant is speaking.\n")


stop_event = threading.Event()

detected = monitor_for_interruption(
    stop_event
)

if detected:

    print(">>> Interrupting assistant...")

    interrupt_tts()


time.sleep(0.5)

stop_event.set()

stop_tts()

print("Done.")
