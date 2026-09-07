from voice import (
    start_microphone,
    stop_microphone,
    listen,
)


start_microphone()

try:

    text = listen()

    print()
    print("Final transcription:")
    print(text)

finally:

    stop_microphone()
