import sounddevice as sd
import numpy as np
import torch

from scipy.signal import resample_poly
from silero_vad import load_silero_vad


DEVICE = 12

MIC_RATE = 48000
VAD_RATE = 16000

CHANNELS = 1
BLOCK_SIZE = 1536


vad = load_silero_vad()

print("Starting VAD...")
print("Speak normally.")
print("Press Ctrl+C to stop.\n")


def callback(indata, frames, time, status):

    if status:
        print("\nSTATUS:", status)

    # Microphone: 48 kHz
    audio_48k = indata[:, 0].copy()

    # 48 kHz -> 16 kHz
    audio_16k = resample_poly(
        audio_48k,
        VAD_RATE,
        MIC_RATE
    ).astype(np.float32)

    audio_tensor = torch.from_numpy(audio_16k)

    # Silero VAD
    probability = vad(
        audio_tensor,
        VAD_RATE
    ).item()

    print(
        f"\rSpeech probability: {probability:.3f}",
        end="",
        flush=True
    )


try:

    with sd.InputStream(
        device=DEVICE,
        samplerate=MIC_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=callback,
    ):

        while True:
            sd.sleep(1000)

except KeyboardInterrupt:

    print("\n\nStopped.")