# AIChatBot

A local, real-time voice conversation AI assistant built with a modular speech pipeline.

I started this project with the goal of building a live conversation AI chatbot that feels more natural than a traditional speech-to-text chatbot. There are now models that can perform direct speech-to-speech conversation, but they generally require more computational resources than I currently have available.

Instead of using a single speech-to-speech model, I decided to build the system using separate components and connect them into a real-time pipeline.

## Why This Architecture?

The initial idea was to use a direct speech-to-speech model.

However, my available hardware is limited, particularly in terms of GPU VRAM. Running large end-to-end speech-to-speech models locally was not practical for my setup.

So I designed an alternative architecture:

```text
Microphone
    ↓
VAD
    ↓
Whisper
    ↓
Ollama
    ↓
Kokoro TTS
    ↓
Audio Output
```

The idea is simple: instead of asking one model to handle everything, each component is responsible for one part of the conversation.

This makes it possible to run the system locally while also giving me the ability to experiment with and optimize each component independently.

## Current Architecture

### Voice Activity Detection

Silero VAD is used to detect when the user starts and stops speaking.

The microphone continuously provides audio to a shared audio queue. VAD determines when speech begins and when the user has finished speaking.

This also allows the assistant to monitor the microphone while it is speaking.

### Speech Recognition

Faster-Whisper converts the user's speech into text.

I experimented with different Whisper model sizes to find a balance between transcription quality and latency on my hardware.

The current system uses a Whisper configuration that provides a good balance between speed and transcription accuracy.

### Language Model

Ollama is used to run the language model locally.

The LLM receives the user's transcription together with the relevant conversation context and generates the response as a stream.

Streaming is important because waiting for the entire response before starting TTS creates unnecessary latency.

Instead:

```text
LLM generates text
       ↓
Text chunks arrive
       ↓
Sentence buffering
       ↓
Kokoro starts speaking
```

This allows the assistant to start speaking while the LLM is still generating the rest of the response.

### Text-to-Speech

Kokoro is used for local text-to-speech.

I experimented with different Kokoro voices and configurations to find a voice and speed that work well for real-time conversation.

The TTS system runs independently from LLM generation using a queue and worker thread.

## Barge-In and Interruption

One of the important goals of this project was making the interaction feel like an actual conversation.

A normal voice chatbot usually works like this:

```text
User speaks
    ↓
Assistant processes
    ↓
Assistant speaks
    ↓
Assistant finishes
    ↓
User speaks again
```

That feels restrictive.

I wanted the user to be able to interrupt the assistant naturally:

```text
Assistant is speaking
        ↓
User starts speaking
        ↓
VAD detects speech
        ↓
TTS stops
        ↓
User's speech is captured
        ↓
Whisper transcribes it
        ↓
LLM responds to the interruption
```

This is currently working in the project.

The assistant can detect an interruption while Kokoro is speaking, stop the current speech, capture the user's interruption, transcribe it, and continue the conversation.

## Conversation Memory Experiments

Memory has been one of the areas I experimented with the most.

I tested several approaches, including:

* Using only the current user query
* Using a conversation summary
* Using a summary plus the last few messages
* Trimming the conversation history
* Sending the full conversation history

Each approach has different trade-offs.

For example, using only the current query can reduce latency and context size, but it performs poorly during longer conversations because the assistant loses important context.

Sending the entire conversation provides more context, but the amount of information sent to the LLM continually increases, which affects latency and scalability.

After experimenting with these approaches, I currently use:

```text
Long-term conversation summary
+
Recent messages
```

This gives me the best balance I have found so far between:

* Conversation accuracy
* Latency
* Context retention
* Natural conversation behavior

The memory system keeps the recent conversation directly available while summarizing older information instead of continuously sending the entire conversation to the model.

## Current Status

The core system is currently working as I initially expected it to.

It can:

* Capture microphone audio
* Detect speech using VAD
* Transcribe speech using Whisper
* Generate responses using a local LLM through Ollama
* Stream LLM responses
* Start TTS before the complete response is generated
* Generate speech using Kokoro
* Interrupt TTS when the user starts speaking
* Capture and transcribe interruptions
* Continue the conversation after an interruption
* Maintain conversation context using summary + recent messages

The project has reached an important first milestone for me: the complete live conversation pipeline is working.

## Performance and Latency

Latency is still one of the main challenges.

The system works well enough for normal conversation, but there are still delays between some stages of the pipeline.

There are several possible sources of latency:

```text
Microphone
    ↓
VAD
    ↓
Whisper
    ↓
LLM
    ↓
TTS
```

Each stage adds some processing time.

I have already experimented with different Whisper models, Kokoro voices, memory strategies, and streaming behavior to improve the overall experience.

There is still room for improvement, particularly in reducing the delay between:

```text
User stops speaking
        ↓
Whisper transcription
        ↓
LLM first token
        ↓
First spoken audio
```

Improving this latency while maintaining good conversation behavior is one of the next areas I want to explore.

## Hardware

The project is being developed on a laptop with:

* Intel Core i5-12450H
* 16 GB RAM
* NVIDIA RTX 3050 Laptop GPU
* 4 GB VRAM
* Fedora Linux

The limited GPU memory was one of the main reasons for choosing a modular architecture instead of a large end-to-end speech-to-speech model.

## Project Structure

```text
AIChatBot/
│
├── main.py
├── chat.py
├── voice.py
├── config.py
│
├── models/
│   └── llm.py
│
├── memory/
│   └── conversation_memory.py
│
├── test/
│
├── requirements-py310.txt
└── README.md
```

## Running the Project

Create the Python environment:

```bash
python3.10 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements-py310.txt
```

Make sure Ollama is installed and a local language model is available.

Then run:

```bash
python main.py
```

## What I Learned

The most interesting part of this project has not been simply making each individual component work.

It has been understanding how the components interact when combined into a real-time system.

A model can be fast by itself but still produce a slow overall experience because another component introduces latency.

Similarly, adding more conversation context can improve accuracy while making responses slower.

The memory experiments especially showed me that there is no single "best" memory strategy. The right approach depends on the balance between context quality, latency, and the type of conversation.

The project also gave me practical experience with streaming generation, audio buffering, asynchronous processing, VAD, speech recognition, TTS, interruption handling, and local LLM deployment.

## First Milestone

At this point, I consider the first major milestone complete.

The original goal was to build a local AI system capable of having a live voice conversation using hardware that was already available to me.

That is now working.

My expectations for the project have also increased as I have worked on it. What initially felt like a successful result now feels like a starting point for improving the system further.

I am still interested in reducing latency and improving the naturalness of the conversation, but I am no longer treating the project as a dedicated full-time experiment.

I will continue improving it when I have interesting ideas or when I want to experiment with something new.

Most importantly, building this project has been a lot of fun.



Thanks for checking it out.
