import sounddevice as sd
import soundfile as sf
import numpy as np

from scipy.signal import resample_poly
from faster_whisper import WhisperModel


DEVICE = 12

MIC_RATE = 48000
WHISPER_RATE = 16000

CHANNELS = 1
DURATION = 5


print("Loading Whisper...")

whisper = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

print("Whisper loaded.")
print()
print("Speak for 5 seconds...")
print("Example:")
print("Hello, my name is Mohsin and I am testing my voice assistant.")

audio = sd.rec(
    int(DURATION * MIC_RATE),
    samplerate=MIC_RATE,
    channels=CHANNELS,
    dtype="float32",
    device=DEVICE
)

sd.wait()

audio = audio[:, 0]

print("\nRecording finished.")

# 48 kHz → 16 kHz
audio_16k = resample_poly(
    audio,
    WHISPER_RATE,
    MIC_RATE
).astype(np.float32)

# Save for debugging
sf.write(
    "whisper_test.wav",
    audio_16k,
    WHISPER_RATE
)

print("Audio saved as whisper_test.wav")
print()
print("Transcribing...")

segments, info = whisper.transcribe(
    audio_16k,
    language="en",
    beam_size=5,
    vad_filter=True
)

text = " ".join(
    segment.text.strip()
    for segment in segments
)

print()
print("================================")
print("TRANSCRIPTION:")
print(text)
print("================================")