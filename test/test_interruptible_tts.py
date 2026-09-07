import time

from voice import (
    start_tts,
    interrupt_tts,
    wait_for_tts,
    stop_tts,
    StreamingTTS,
)


start_tts()

tts = StreamingTTS()

chunks = [
    "This is a long sentence that should start playing immediately. ",
    "The assistant should continue speaking this sentence for a while. ",
    "There is also another sentence waiting in the queue. ",
    "This sentence should never finish if we interrupt the assistant. ",
]

for chunk in chunks:
    print(chunk, end="", flush=True)
    tts.add_chunk(chunk)

tts.finish()

print("\n\nTTS started.")

time.sleep(2)

print("\n>>> INTERRUPTING TTS!")

interrupt_tts()

print(">>> TTS interrupted.")

time.sleep(1)

stop_tts()

print("Done.")
