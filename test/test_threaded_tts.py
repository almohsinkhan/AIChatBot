from voice import start_tts, StreamingTTS, wait_for_tts, stop_tts


start_tts()

tts = StreamingTTS()

chunks = [
    "Hello! ",
    "This is the first sentence. ",
    "Now Kokoro should speak this sentence while the program continues. ",
    "And this is the final sentence."
]

print("Sending chunks...\n")

for chunk in chunks:

    print(chunk, end="", flush=True)

    tts.add_chunk(chunk)

tts.finish()

print("\n\nWaiting for TTS...")

wait_for_tts()

stop_tts()

print("Done.")
