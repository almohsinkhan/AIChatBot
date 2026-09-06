from kokoro import KPipeline
import sounddevice as sd
import numpy as np


print("Loading Kokoro...")

pipeline = KPipeline(lang_code="a")

print("Kokoro loaded.\n")


chunks = [
    "Hello! ",
    "This is a test of streaming text to speech. ",
    "Kokoro should speak each sentence separately. ",
]


sentence = ""


for chunk in chunks:

    print(f"Chunk: {chunk}")

    sentence += chunk

    if sentence.rstrip().endswith((".", "!", "?")):

        text = sentence.strip()

        print(f"Speaking: {text}")

        generator = pipeline(
            text,
            voice="af_bella",
            speed=1.05,
        )

        for _, _, audio in generator:

            audio = np.asarray(
                audio,
                dtype=np.float32
            )

            print(
                f"Playing {len(audio)} samples"
            )

            sd.play(audio, 24000)
            sd.wait()

        sentence = ""


if sentence.strip():

    print(f"Speaking: {sentence}")

    generator = pipeline(
        sentence.strip(),
        voice="af_bella",
        speed=1.05,
    )

    for _, _, audio in generator:

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        sd.play(audio, 24000)
        sd.wait()


print("\nDone.")