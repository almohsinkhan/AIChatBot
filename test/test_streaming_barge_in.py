import threading
import time
import queue

import numpy as np
import torch
from scipy.signal import resample_poly

from chat import chat_stream
from voice import (
    start_microphone,
    stop_microphone,
    StreamingTTS,
    interrupt_tts,
    wait_for_tts,
    transcribe_audio,
    get_microphone_audio,
    mic_queue,
    vad,
    MODEL_SAMPLE_RATE,
    MIC_SAMPLE_RATE,
    start_tts,
    stop_tts,
)


def monitor_for_interruption(
    stop_monitor,
    user_interrupted,
    interruption_frames,
):
    """
    Monitor the microphone while the assistant is speaking.

    Once speech is detected across several consecutive
    frames, stop TTS and preserve the audio for transcription.
    """

    START_THRESHOLD = 0.55
    START_CONFIRM_FRAMES = 3
    PRE_BUFFER_FRAMES = 10

    speech_confirmations = 0
    pre_buffer = []

    while not stop_monitor.is_set():

        audio = get_microphone_audio(timeout=0.1)

        if audio is None:
            continue

        pre_buffer.append(audio)

        if len(pre_buffer) > PRE_BUFFER_FRAMES:
            pre_buffer.pop(0)

        audio_16k = resample_poly(
            audio,
            MODEL_SAMPLE_RATE,
            MIC_SAMPLE_RATE,
        ).astype(np.float32)

        probability = vad(
            torch.from_numpy(audio_16k),
            MODEL_SAMPLE_RATE,
        ).item()

        if probability > START_THRESHOLD:
            speech_confirmations += 1
        else:
            speech_confirmations = 0

        if speech_confirmations >= START_CONFIRM_FRAMES:

            print(
                f"\n>>> User interrupted "
                f"(VAD={probability:.2f})"
            )

            interruption_frames.extend(pre_buffer)

            user_interrupted.set()

            interrupt_tts()

            return


def capture_interruption(
    interruption_frames,
):
    """
    Continue recording until the user stops speaking.
    """

    print("Capturing interruption...")

    END_THRESHOLD = 0.35
    SILENCE_LIMIT = 40

    silence_frames = 0

    while True:

        audio = get_microphone_audio(timeout=1.0)

        if audio is None:
            continue

        audio_16k = resample_poly(
            audio,
            MODEL_SAMPLE_RATE,
            MIC_SAMPLE_RATE,
        ).astype(np.float32)

        probability = vad(
            torch.from_numpy(audio_16k),
            MODEL_SAMPLE_RATE,
        ).item()

        interruption_frames.append(audio)

        if probability > END_THRESHOLD:
            silence_frames = 0
        else:
            silence_frames += 1

        if silence_frames >= SILENCE_LIMIT:

            print("Interruption ended.")

            break


def main():

    summary = ""
    recent_messages = []

    start_microphone()
    start_tts()

    print()
    print("Streaming barge-in test started.")
    print()
    print("The assistant will generate a long response.")
    print("Kokoro should start speaking before generation finishes.")
    print()
    print("While it is speaking, say:")
    print()
    print("    Hello")
    print()
    print("TTS should stop and Whisper should transcribe you.")
    print()

    # Remove stale microphone frames.
    while True:
        try:
            mic_queue.get_nowait()
        except queue.Empty:
            break
        else:
            mic_queue.task_done()

    user_interrupted = threading.Event()
    stop_monitor = threading.Event()

    interruption_frames = []

    monitor_thread = threading.Thread(
        target=monitor_for_interruption,
        args=(
            stop_monitor,
            user_interrupted,
            interruption_frames,
        ),
        daemon=True,
    )

    monitor_thread.start()

    tts = StreamingTTS()

    full_response = ""

    print("Assistant: ", end="", flush=True)

    try:

        for chunk in chat_stream(
            """
            Give me a long explanation about how a local AI voice
            assistant works. Explain the microphone, voice activity
            detection, speech recognition, language model, streaming
            response generation, text to speech, and interruption
            handling. Explain each part in several sentences.
            """,
            summary,
            recent_messages,
        ):

            # User interrupted while Ollama was generating.
            if user_interrupted.is_set():
                print("\n\nGeneration interrupted.")
                break

            print(chunk, end="", flush=True)

            full_response += chunk

            # IMPORTANT:
            # This sends chunks to TTS while Ollama
            # is still generating.
            tts.add_chunk(chunk)

        print()

        if user_interrupted.is_set():

            # Do NOT call tts.finish().
            #
            # interrupt_tts() already stopped playback
            # and cleared queued speech.

            capture_interruption(
                interruption_frames
            )

            monitor_thread.join(timeout=0.5)

            if interruption_frames:

                audio_48k = np.concatenate(
                    interruption_frames
                )

                text = transcribe_audio(audio_48k)

                print()
                print("Captured interruption:")
                print(text)

        else:

            # Normal completion.
            tts.finish()

            wait_for_tts()

            stop_monitor.set()
            monitor_thread.join(timeout=0.5)

            print()
            print("Response finished normally.")

    finally:

        stop_monitor.set()

        if monitor_thread.is_alive():
            monitor_thread.join(timeout=0.5)

        stop_tts()
        stop_microphone()


if __name__ == "__main__":
    main()
