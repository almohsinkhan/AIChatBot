import threading
import queue

import numpy as np
import torch
from scipy.signal import resample_poly

from chat import chat_stream, update_memory

from voice import (
    start_microphone,
    stop_microphone,
    start_tts,
    stop_tts,
    StreamingTTS,
    interrupt_tts,
    wait_for_tts,
    transcribe_audio,
    get_microphone_audio,
    mic_queue,
    vad,
    MODEL_SAMPLE_RATE,
    MIC_SAMPLE_RATE,
    listen,
)


def monitor_for_interruption(
    stop_monitor,
    user_interrupted,
    interruption_frames,
):
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


def capture_interruption(interruption_frames):
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


def stream_response_with_barge_in(
    user_input,
    summary,
    recent_messages,
):
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
            user_input,
            summary,
            recent_messages,
        ):
            if user_interrupted.is_set():
                break

            print(chunk, end="", flush=True)

            full_response += chunk
            tts.add_chunk(chunk)

        print()

        if user_interrupted.is_set():
            print("Response interrupted.")

            capture_interruption(
                interruption_frames
            )

            if interruption_frames:
                audio_48k = np.concatenate(
                    interruption_frames
                )

                interrupted_text = transcribe_audio(
                    audio_48k
                )
            else:
                interrupted_text = ""

            return full_response, interrupted_text

        tts.finish()
        wait_for_tts()

        return full_response, ""

    finally:
        stop_monitor.set()

        if monitor_thread.is_alive():
            monitor_thread.join(timeout=0.5)


def main():
    summary = ""
    recent_messages = []

    start_microphone()
    start_tts()

    print("\nVoice assistant started.")
    print("Speak normally. Press Ctrl+C to exit.\n")

    try:
        while True:
            user_input = listen()

            if not user_input:
                continue

            print(f"\nUser: {user_input}")

            response, interrupted_text = (
                stream_response_with_barge_in(
                    user_input,
                    summary,
                    recent_messages,
                )
            )

            if not interrupted_text:
                summary, recent_messages = update_memory(
                    user_input,
                    response,
                    summary,
                    recent_messages,
                )
                continue

            summary, recent_messages = update_memory(
                user_input,
                response,
                summary,
                recent_messages,
                interrupted=True,
            )

            user_input = interrupted_text

            print(f"\nUser: {user_input}")

            response, _ = (
                stream_response_with_barge_in(
                    user_input,
                    summary,
                    recent_messages,
                )
            )

            summary, recent_messages = update_memory(
                user_input,
                response,
                summary,
                recent_messages,
            )

    except KeyboardInterrupt:
        print("\n\nStopping assistant...")

    finally:
        stop_tts()
        stop_microphone()
        print("Assistant stopped.")


if __name__ == "__main__":
    main()