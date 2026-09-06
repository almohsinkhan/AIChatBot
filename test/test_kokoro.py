from kokoro import KPipeline
import sounddevice as sd
import numpy as np


print("Loading Kokoro...")

pipeline = KPipeline(lang_code="a")

print("Kokoro loaded.")

text = "Hello Mohsin. This is a direct Kokoro audio test."

generator = pipeline(
    text,
    voice="af_bella",
    speed=1.05,
)

for _, _, audio in generator:

    audio = np.asarray(audio, dtype=np.float32)

    print(
        f"Playing {len(audio)} samples..."
    )

    sd.play(audio, 24000)
    sd.wait()

print("Done.")