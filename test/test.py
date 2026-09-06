import sounddevice as sd
import numpy as np
from scipy.signal import resample_poly

DEVICE = 12

MIC_RATE = 48000
WHISPER_RATE = 16000

CHANNELS = 1
BLOCK_SIZE = 1024


audio_buffer = []


def callback(indata, frames, time, status):
    if status:
        print("STATUS:", status)

    audio_buffer.append(indata[:, 0].copy())


print("Listening...")
print("Speak for a few seconds.")
print("Press Ctrl+C to stop.")

try:
    with sd.InputStream(
        device=DEVICE,
        samplerate=MIC_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=callback,
    ):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nStopping...")

    audio = np.concatenate(audio_buffer)

    print("Captured samples:", len(audio))
    print("Captured duration:", len(audio) / MIC_RATE, "seconds")

    # 48 kHz → 16 kHz
    audio_16k = resample_poly(
        audio,
        WHISPER_RATE,
        MIC_RATE
    ).astype(np.float32)

    print("Resampled samples:", len(audio_16k))
    print("Resampled duration:", len(audio_16k) / WHISPER_RATE, "seconds")

    import soundfile as sf

    sf.write(
        "mic_test_16k.wav",
        audio_16k,
        WHISPER_RATE
    )

    print("Saved: mic_test_16k.wav")