import queue
import threading

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad
from faster_whisper import WhisperModel
from kokoro import KPipeline


# -----------------------------
# Audio configuration
# -----------------------------

MIC_DEVICE = 12

MIC_SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000

FRAME_SIZE = 512

VOICE = "af_sky"
TTS_SAMPLE_RATE = 24000


# -----------------------------
# Load models
# -----------------------------

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


# ============================================================
# SPEECH TO TEXT
# ============================================================

def listen():
    """
    Continuously listen to the microphone.

    Microphone:
        48 kHz

    Whisper/VAD:
        16 kHz
    """

    audio_queue = queue.Queue()

    speaking = False
    silence_frames = 0

    frames = []

    PRE_BUFFER_FRAMES = 10
    SILENCE_LIMIT = 25

    pre_buffer = []

    print("Listening...")

    def callback(indata, frames_count, time_info, status):

        if status:
            print(status)

        # IMPORTANT:
        # Callback only copies audio and puts it in the queue.
        # Do not run VAD/resampling here.
        audio = indata[:, 0].copy()

        audio_queue.put(audio)

    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=MIC_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE * 3,
        callback=callback,
    ):

        while True:

            # Wait for the NEXT actual microphone block.
            audio = audio_queue.get()

            # Keep a rolling pre-buffer.
            pre_buffer.append(audio)

            if len(pre_buffer) > PRE_BUFFER_FRAMES:
                pre_buffer.pop(0)

            # Resample only outside the callback.
            resampled = resample_poly(
                audio,
                MODEL_SAMPLE_RATE,
                MIC_SAMPLE_RATE,
            ).astype(np.float32)

            audio_tensor = torch.from_numpy(resampled)

            probability = vad(
                audio_tensor,
                MODEL_SAMPLE_RATE,
            ).item()

            if probability > 0.5:

                if not speaking:

                    print("Speech detected...")

                    speaking = True
                    silence_frames = 0

                    # Include audio immediately before speech.
                    frames.extend(pre_buffer)

                else:

                    silence_frames = 0

                frames.append(audio)

            elif speaking:

                # Continue recording during silence.
                frames.append(audio)

                silence_frames += 1

                if silence_frames >= SILENCE_LIMIT:

                    print("Speech ended.")

                    break

    if not frames:
        return ""

    # Combine microphone audio.
    audio_48k = np.concatenate(frames)

    # Convert 48 kHz -> 16 kHz.
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

# ============================================================
# TEXT TO SPEECH
# ============================================================

tts_queue = queue.Queue()

tts_thread = None
tts_running = False


def speak(text):
    """
    Generate and play one piece of text.
    """

    if not text:
        return

    generator = kokoro_pipeline(
        text,
        voice=VOICE,
        speed=1.05,
    )

    for _, _, audio in generator:

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        sd.play(
            audio,
            TTS_SAMPLE_RATE,
        )

        sd.wait()


def _tts_worker():

    while True:

        text = tts_queue.get()

        if text is None:
            tts_queue.task_done()
            break

        try:
            speak(text)

        except Exception as e:
            print(f"\nTTS error: {e}")

        finally:
            tts_queue.task_done()


def start_tts():

    global tts_thread
    global tts_running

    if tts_running:
        return

    tts_running = True

    tts_thread = threading.Thread(
        target=_tts_worker,
        daemon=True,
    )

    tts_thread.start()


def wait_for_tts():

    """
    Wait until everything currently in the TTS queue
    has finished speaking.
    """

    tts_queue.join()


def stop_tts():

    global tts_running

    if not tts_running:
        return

    tts_queue.put(None)

    tts_queue.join()

    tts_running = False


# ============================================================
# STREAMING TTS
# ============================================================

class StreamingTTS:

    def __init__(self):

        self.sentence = ""

    def add_chunk(self, chunk):

        if not chunk:
            return

        self.sentence += chunk

        # Wait until a natural sentence boundary.
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