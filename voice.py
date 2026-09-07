import queue
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad
from faster_whisper import WhisperModel
from kokoro import KPipeline

MIC_DEVICE = 12

MIC_SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000

FRAME_SIZE = 512

VOICE = "af_sky"
TTS_SAMPLE_RATE = 24000

MIC_GAIN = 1.0




print("Loading Kokoro...")
kokoro_pipeline = KPipeline(lang_code="a")

print("Loading Silero VAD...")
vad = load_silero_vad()

print("Loading Whisper...")
whisper = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
)

print("Voice models loaded.\n")

mic_queue = queue.Queue()

mic_stream = None
mic_running = False


def _mic_callback(indata, frames_count, time_info, status):

    if status:
        print(f"\nAudio status: {status}")

    # Callback does almost nothing.
    # Put a copy of the microphone block into the queue.
    mic_queue.put(
        indata[:, 0].copy()
    )


def start_microphone():

    global mic_stream
    global mic_running

    if mic_running:
        return

    mic_stream = sd.InputStream(
        device=MIC_DEVICE,
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE * 3,
        callback=_mic_callback,
    )

    mic_stream.start()

    mic_running = True

    print("Microphone started.")


def stop_microphone():

    global mic_stream
    global mic_running

    if not mic_running:
        return

    mic_stream.stop()
    mic_stream.close()

    mic_stream = None
    mic_running = False

    print("Microphone stopped.")


def get_microphone_audio(timeout=1.0):

    try:

        return mic_queue.get(
            timeout=timeout
        )

    except queue.Empty:

        return None

def listen():
    """
    Listen for one complete user utterance using
    the shared microphone stream.

    Speech must be detected across multiple consecutive
    frames before recording starts, which helps reject
    background noise.
    """

    frames = []

    speaking = False
    silence_frames = 0
    speech_confirmations = 0

    pre_buffer = []

    PRE_BUFFER_FRAMES = 10

    # Threshold required to START speech detection.
    START_THRESHOLD = 0.55

    # Lower threshold used while already speaking.
    END_THRESHOLD = 0.35

    # Require several consecutive speech frames
    # before considering speech to have started.
    START_CONFIRM_FRAMES = 3

    # Number of quiet frames required to end speech.
    SILENCE_LIMIT = 40

    print("Listening...")

    while True:
        audio = get_microphone_audio(timeout=1.0)

        if audio is None:
            continue

        # Keep a small amount of audio before speech detection.
        # This prevents cutting off the first word.
        pre_buffer.append(audio)

        if len(pre_buffer) > PRE_BUFFER_FRAMES:
            pre_buffer.pop(0)

        # Resample microphone audio:
        # 48 kHz → 16 kHz for Silero VAD.
        audio_16k = resample_poly(
            audio,
            MODEL_SAMPLE_RATE,
            MIC_SAMPLE_RATE,
        ).astype(np.float32)

        # Run VAD.
        probability = vad(
            torch.from_numpy(audio_16k),
            MODEL_SAMPLE_RATE,
        ).item()

        # Waiting for speech
        if not speaking:

            if probability > START_THRESHOLD:
                speech_confirmations += 1
            else:
                speech_confirmations = 0

            # Require several consecutive frames
            # before confirming speech.
            if speech_confirmations >= START_CONFIRM_FRAMES:

                print("Speech detected...")

                speaking = True
                silence_frames = 0

                # Include audio captured immediately
                # before speech was confirmed.
                frames.extend(pre_buffer)

        # Currently recording speech
        else:

            if probability > END_THRESHOLD:
                silence_frames = 0
            else:
                silence_frames += 1

            frames.append(audio)

            # Enough silence → user finished speaking.
            if silence_frames >= SILENCE_LIMIT:

                print("Speech ended.")
                break

    # No audio safety check
    if not frames:
        return ""

    # Combine all microphone frames.
    audio_48k = np.concatenate(frames)

    # Convert to 16 kHz for Whisper.
    audio_16k = resample_poly(
        audio_48k,
        MODEL_SAMPLE_RATE,
        MIC_SAMPLE_RATE,
    ).astype(np.float32)

    print("Transcribing...")

    # Whisper
    segments, info = whisper.transcribe(
        audio_16k,
        language="en",
        beam_size=5,
        temperature=0,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    text = text.strip()

    print(f"You: {text}")

    return text
    
def transcribe_audio(audio_48k):
    """
    Transcribe 48 kHz microphone audio using Whisper.
    """

    if audio_48k is None:
        return ""

    audio_16k = resample_poly(
        audio_48k,
        MODEL_SAMPLE_RATE,
        MIC_SAMPLE_RATE,
    ).astype(np.float32)

    print("Transcribing...")

    segments, info = whisper.transcribe(
        audio_16k,
        language="en",
        beam_size=5,
        temperature=0,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    text = text.strip()

    print(f"You: {text}")

    return text

def speak(text):
    """
    Generate Kokoro audio and play it through a dedicated
    OutputStream.

    Playback can be interrupted by setting tts_stop_event.
    """

    if not text:
        return

    generator = kokoro_pipeline(
        text,
        voice=VOICE,
        speed=1.05,
    )

    with sd.OutputStream(
        samplerate=TTS_SAMPLE_RATE,
        channels=1,
        dtype="float32",
    ) as stream:

        for _, _, audio in generator:

            if tts_stop_event.is_set():
                stream.abort()
                return

            audio = np.asarray(
                audio,
                dtype=np.float32,
            )

            # Kokoro may return a 1-D array.
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)

            # Write audio in small blocks so interruption
            # can be detected quickly.
            block_size = 1024

            for start in range(
                0,
                len(audio),
                block_size,
            ):

                if tts_stop_event.is_set():

                    stream.abort()

                    return

                end = min(
                    start + block_size,
                    len(audio),
                )

                stream.write(
                    audio[start:end]
                )

tts_queue = queue.Queue()

tts_thread = None
tts_running = False

tts_stop_event = threading.Event()
tts_finished_event = threading.Event()

def _tts_worker():

    while True:

        text = tts_queue.get()

        if text is None:

            tts_queue.task_done()

            break

        try:

            tts_finished_event.clear()

            speak(text)

        except Exception as e:

            print(f"\nTTS error: {e}")

        finally:

            tts_finished_event.set()

            tts_queue.task_done()


def start_tts():

    global tts_thread
    global tts_running

    if tts_running:
        return

    tts_stop_event.clear()

    tts_running = True

    tts_thread = threading.Thread(
        target=_tts_worker,
        daemon=True,
    )

    tts_thread.start()


def stop_tts():

    global tts_running

    if not tts_running:
        return

    # Stop currently playing audio.
    tts_stop_event.set()

    # Stop sounddevice immediately.
    sd.stop()

    # Remove anything waiting in the queue.
    while True:

        try:

            item = tts_queue.get_nowait()

        except queue.Empty:

            break

        else:

            tts_queue.task_done()

    # Stop worker.
    tts_queue.put(None)

    tts_queue.join()

    tts_running = False


def interrupt_tts():

    """
    Immediately stop current TTS playback and clear
    queued speech.
    """

    tts_stop_event.set()

    sd.stop()

    # Clear queued sentences.
    while True:

        try:

            tts_queue.get_nowait()

        except queue.Empty:

            break

        else:

            tts_queue.task_done()

    # Give the current worker a moment to exit.
    time.sleep(0.02)

    tts_stop_event.clear()


def speak_with_barge_in(text):
    """
    Speak text while monitoring the shared microphone.

    If the user starts speaking:
    1. Stop TTS.
    2. Preserve the audio that triggered the interruption.
    3. Continue recording until the user stops speaking.

    Returns:
        (interrupted, audio_48k)

        interrupted:
            True if the user interrupted TTS.

        audio_48k:
            Recorded interruption audio at 48 kHz.
    """

    if not text:
        return False, None

    # Clear old microphone frames.
    while True:
        try:
            mic_queue.get_nowait()
        except queue.Empty:
            break
        else:
            mic_queue.task_done()

    stop_monitor = threading.Event()
    user_interrupted = threading.Event()

    interruption_frames = []

    def monitor():
        speech_confirmations = 0

        START_THRESHOLD = 0.55
        START_CONFIRM_FRAMES = 3

        # Keep a small buffer so the beginning of the
        # user's speech isn't lost.
        pre_buffer = []

        PRE_BUFFER_FRAMES = 10

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

                # Preserve the audio around the moment
                # speech was detected.
                interruption_frames.extend(pre_buffer)

                user_interrupted.set()

                # Stop Kokoro immediately.
                interrupt_tts()

                return

    monitor_thread = threading.Thread(
        target=monitor,
        daemon=True,
    )

    monitor_thread.start()

    # Start TTS.
    tts_queue.put(text)

    # Wait until TTS finishes or user interrupts.
    while tts_queue.unfinished_tasks > 0:

        if user_interrupted.is_set():
            break

        time.sleep(0.01)

    # Stop microphone monitor.
    stop_monitor.set()

    monitor_thread.join(timeout=0.5)

    if not user_interrupted.is_set():
        return False, None

    # ------------------------------------------------
    # Continue capturing the rest of the interruption
    # ------------------------------------------------

    speaking = True
    silence_frames = 0

    END_THRESHOLD = 0.35
    SILENCE_LIMIT = 40

    print("Capturing interruption...")

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

    audio_48k = np.concatenate(interruption_frames)

    return True, audio_48k



def monitor_for_interruption(stop_event):
    """
    Monitor the shared microphone for speech while
    the assistant is speaking.
    """

    while not stop_event.is_set():

        audio = get_microphone_audio(
            timeout=0.1
        )

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

        if probability > 0.5:

            print(
                f"\n>>> User speech detected "
                f"(VAD={probability:.2f})"
            )

            return True

    return False

def wait_for_tts():

    """
    Wait until all queued TTS has finished.
    """

    tts_queue.join()


class StreamingTTS:

    def __init__(self):

        self.sentence = ""

    def add_chunk(self, chunk):

        if not chunk:
            return

        self.sentence += chunk

        # Start speaking at natural sentence boundaries.
        if self.sentence.rstrip().endswith(
            (".", "!", "?")
        ):

            text = self.sentence.strip()

            if text:

                tts_queue.put(text)

            self.sentence = ""

    def finish(self):

        if self.sentence.strip():

            tts_queue.put(
                self.sentence.strip()
            )

        self.sentence = ""