import time

from voice import (
    start_microphone,
    stop_microphone,
    start_tts,
    stop_tts,
    speak_with_barge_in,
    transcribe_audio,
)


def main():

    start_microphone()
    start_tts()

    try:

        print("\nStarting barge-in capture test.")
        print("Wait for Kokoro to speak.")
        print("Then interrupt it with a complete sentence.\n")

        time.sleep(2)

        text = (
            "This is a long sentence so that you have enough "
            "time to interrupt me while I am speaking. "
            "Please interrupt me and say something completely different."
        )

        interrupted, audio = speak_with_barge_in(text)

        print()

        if interrupted:

            print("RESULT: User interrupted TTS.")

            user_text = transcribe_audio(audio)

            print(f"\nCaptured interruption:")
            print(user_text)

        else:

            print("RESULT: TTS finished normally.")

    finally:

        stop_tts()
        stop_microphone()


if __name__ == "__main__":
    main()