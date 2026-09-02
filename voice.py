import torch
import sounddevice as sd
from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
FRAME_SIZE = 512

vad = load_silero_vad()

print("Listening...")


frames = []
speaking = False
silence_count = 0

while True:

    audio = sd.rec(
        FRAME_SIZE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio_tensor = torch.from_numpy(audio[:, 0])

    probability = vad(audio_tensor, SAMPLE_RATE)

    if probability > 0.5:

        if not speaking:
            print("Speech started")

        speaking = True
        silence_count = 0
        frames.append(audio)

    elif speaking:

        frames.append(audio)
        silence_count += 1

        if silence_count > 20:
            print("Speech ended")
            print("Frames collected:", len(frames))

            frames = []
            speaking = False
            silence_count = 0