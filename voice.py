import subprocess
import numpy as np
import torch
import sounddevice as sd
from silero_vad import load_silero_vad
from faster_whisper import WhisperModel

import threading
import queue

VOICE = "en_US-lessac-medium"

SAMPLE_RATE = 16000
FRAME_SIZE = 512

vad = load_silero_vad()

whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

tts_queue = queue.Queue()

def tts_worker():
    while True:

        sentence = tts_queue.get()

        if sentence is None:
            break

        speak(sentence)

        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def stop_tts():
    tts_queue.put(None)
    tts_queue.join()

def listen():

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

            speaking = True
            silence_count = 0
            frames.append(audio)

        elif speaking:

            frames.append(audio)
            silence_count += 1

            if silence_count > 20:

                audio_data = torch.cat([
                    torch.from_numpy(frame[:, 0])
                    for frame in frames
                ])

                segments, info = whisper.transcribe(
                    audio_data.numpy(),
                    language="en"
                )

                text = " ".join(
                    segment.text.strip()
                    for segment in segments
                )

                return text.strip()

def speak(text):

    process = subprocess.run(
        [
            "python",
            "-m",
            "piper",
            "-m",
            VOICE,
            "--output_raw"
        ],
        input=text.encode(),
        stdout=subprocess.PIPE
    )

    audio = np.frombuffer(
        process.stdout,
        dtype=np.int16
    )

    sd.play(
        audio,
        samplerate=22050
    )

    sd.wait()

MIN_SENTENCE_LENGTH = 5
def speak_stream(chunks):
    sentence =""

    for chunk in chunks:
        sentence  += chunk

        print(chunk, end="", flush=True)

        sentence += chunk

        if (sentence.rstrip().endswith((".", "!", "?"))) and (len(sentence.split()) > MIN_SENTENCE_LENGTH):
            tts_queue.put(sentence.strip())
            sentence = ""

    if sentence.strip():
        tts_queue.put(sentence.strip())

class StreamingTTS:

    def __init__(self):
        self.sentence = ""

    def add_chunk(self, chunk):
        self.sentence += chunk

        if self.sentence.rstrip().endswith((".", "!", "?")):
            tts_queue.put(self.sentence.strip())
            self.sentence = ""

    def finish(self):
        if self.sentence.strip():
            tts_queue.put(self.sentence.strip())

        self.sentence = ""