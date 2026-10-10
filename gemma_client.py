"""
Small helper for talking to Gemma 4 through the Gemini API.

Three functions, nothing else:
  ask(prompt)                      -> text answer
  ask_with_image(prompt, path)     -> text answer about an image file
  make_chat()                      -> a chat object that remembers earlier messages

Based on: https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api
"""

import mimetypes

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Read GEMINI_API_KEY from the .env file into the environment.
# The key is never written in code and never printed.
load_dotenv()

MODEL = "gemma-4-26b-a4b-it"

# Gemma 4 "thinks" before answering, which can take a long time. For reading a statement we
# just need the data out, so we set thinking to "minimal" (the docs say this switches it off).
FAST = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level="minimal")
)

# genai.Client() looks for GEMINI_API_KEY in the environment by itself.
# We create it once, the first time it is needed, and reuse it afterwards.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def ask(prompt):
    """Send one text prompt to Gemma and return its answer as a string."""
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=FAST,
    )
    return response.text


def ask_with_image(prompt, image_path):
    """Send an image file plus a text prompt to Gemma and return its answer."""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        image = types.Part.from_bytes(data=f.read(), mime_type=mime_type)  # picture sent inside the request

    response = _get_client().models.generate_content(
        model=MODEL,
        contents=[prompt, image],  # same order as the docs' inline-image example
        config=FAST,
    )
    return response.text


def make_chat():
    """Start a multi-turn chat. Use chat.send_message("...").text to talk to it."""
    return _get_client().chats.create(model=MODEL)