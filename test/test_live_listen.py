import sounddevice as sd
import numpy as np
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad
from faster_whisper import WhisperModel


# =========================
# CONFIG
# =========================

DEVICE = 12

MIC_RATE = 48000
VAD_RATE = 16000

CHANNELS = 1

# 1536 samples at 48 kHz = 32 ms
BLOCK_SIZE = 1536

VAD_THRESHOLD = 0.5

# How much silence ends an utterance
SILENCE_DURATION = 0.8

# Keep some audio before VAD detects speech
PRE_BUFFER_DURATION = 0.3


# =========================
# MODELS
# =========================

print("Loading Silero VAD...")
vad = load_silero_vad()

print("Loading Whisper...")
whisper = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

print("Models loaded.\n")


# =========================
# AUDIO SETTINGS
# =========================

SILENCE_BLOCKS = int(
    SILENCE_DURATION * MIC_RATE / BLOCK_SIZE
)

PRE_BUFFER_BLOCKS = int(
    PRE_BUFFER_DURATION * MIC_RATE / BLOCK_SIZE
)


# =========================
# STATE
# =========================

audio_buffer = []
pre_buffer = []

speaking = False
silence_count = 0

speech_started = False


# =========================
# CALLBACK
# =========================

def callback(indata, frames, time, status):

    if status:
        print("\nAudio status:", status)

    audio_48k = indata[:, 0].copy()

    # Resample 48 kHz -> 16 kHz
    audio_16k = resample_poly(
        audio_48k,
        VAD_RATE,
        MIC_RATE
    ).astype(np.float32)

    audio_tensor = torch.from_numpy(audio_16k)

    probability = vad(
        audio_tensor,
        VAD_RATE
    ).item()

    process_audio(
        audio_48k,
        probability
    )


# =========================
# AUDIO PROCESSING
# =========================

def process_audio(audio, probability):

    global speaking
    global silence_count
    global audio_buffer
    global pre_buffer

    # Always maintain a small pre-buffer
    pre_buffer.append(audio)

    if len(pre_buffer) > PRE_BUFFER_BLOCKS:
        pre_buffer.pop(0)

    # ---------------------
    # SPEECH
    # ---------------------

    if probability > VAD_THRESHOLD:

        if not speaking:

            print("\n\n🎤 Speech detected")

            # Include audio immediately before speech
            audio_buffer = pre_buffer.copy()

            speaking = True

        audio_buffer.append(audio)

        silence_count = 0

        print(
            f"\rSpeaking... VAD={probability:.2f}",
            end="",
            flush=True
        )

    # ---------------------
    # SILENCE
    # ---------------------

    elif speaking:

        audio_buffer.append(audio)

        silence_count += 1

        print(
            f"\rSilence... {silence_count}/{SILENCE_BLOCKS}",
            end="",
            flush=True
        )

        if silence_count >= SILENCE_BLOCKS:

            print("\n\n🛑 Speech ended")

            speaking = False
            silence_count = 0

            process_utterance(
                audio_buffer
            )

            audio_buffer = []


# =========================
# WHISPER
# =========================

def process_utterance(audio_blocks):

    print("Preparing audio...")

    audio_48k = np.concatenate(audio_blocks)

    audio_16k = resample_poly(
        audio_48k,
        VAD_RATE,
        MIC_RATE
    ).astype(np.float32)

    print("Transcribing...")

    segments, info = whisper.transcribe(
        audio_16k,
        language="en",
        beam_size=5,
        temperature=0,
        vad_filter=False,
        condition_on_previous_text=False
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    text = text.strip()

    print("\n" + "=" * 50)
    print("YOU:")
    print(text)
    print("=" * 50)

    print("\n🎤 Listening...\n")


# =========================
# START MICROPHONE
# =========================

print("🎤 Listening...")
print("Speak normally.")
print("Press Ctrl+C to stop.\n")


try:

    with sd.InputStream(
        device=DEVICE,
        samplerate=MIC_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=callback
    ):

        while True:
            sd.sleep(1000)

except KeyboardInterrupt:

    print("\n\nStopped.")