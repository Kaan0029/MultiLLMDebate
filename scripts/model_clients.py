# model_clients.py
import os
from dotenv import load_dotenv

from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai

from prompts import ROUND1_PROMPT, ROUND2_TEMPLATE, REFEREE_PROMPT
from utils import safe_parse_json
import torch

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_KEY)
claude_client = Anthropic(api_key=ANTHROPIC_KEY)
genai.configure(api_key=GEMINI_KEY)

# -----------------------------------------------------
# Helper: Build prompts for R1 and R2
# -----------------------------------------------------

from faster_whisper import WhisperModel

# load whisper once
#asr = WhisperModel("medium.en")
asr = None

def get_asr():
    global asr
    if asr is None:
        asr = WhisperModel("small.en")
    return asr


# def transcribe_audio(audio_path: str) -> str:
#     """Return text transcript from audio using Whisper."""
#     segments, _ = asr.transcribe(audio_path)
#     return " ".join(seg.text for seg in segments)

def transcribe_audio(audio_path: str) -> str:
    try:
        segments, _ = get_asr().transcribe(audio_path)
        text = " ".join(seg.text for seg in segments)
        torch.cuda.empty_cache()
        return text
    except Exception as e:
        print(f"⚠️ Whisper failed on {audio_path}: {e}")
        return ""

def upload_audio_file(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = openai_client.files.create(
            file=f,
            purpose="assistants"   # also used for multimodal
        )
    return resp.id


import base64

def call_referee_audio(prompt, audio_path):

    # ✅ Upload audio properly
    file_id = upload_audio_file(audio_path)

    resp = openai_client.responses.create(
        model="gpt-4o-audio-preview",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_file",
                        "file_id": file_id
                    }
                ]
            }
        ]
    )

    return resp.output_text





def build_referee_prompt_audio(transcript, r2_outputs):
    debate_text = ""
    for o in r2_outputs:
        debate_text += (
            f"- {o['model']} → label: {o['updated_label']}\n"
            f"  reason: {o['parsed'].get('reason', '')}\n"
        )

    return f"""
{REFEREE_PROMPT}

Transcript:
{transcript}

Examiners' updated judgments:
{debate_text}
"""



def build_round1_prompt(transcript):
    return ROUND1_PROMPT + f"\n\nTranscript:\n{transcript}\n"

def build_round2_prompt(transcript, self_label, self_reason, others):
    others_text = ""
    for o in others:
        others_text += f"- {o['model']}: label={o['label']}, reason={o['reason']}\n"

    return ROUND2_TEMPLATE.format(
        transcript=transcript,
        self_label=self_label,
        self_reason=self_reason,
        other_judgments=others_text,
    )

def build_referee_prompt(transcript, round2_outputs):
    debate_text = ""
    for o in round2_outputs:
        debate_text += (
            f"- {o['model']} → label: {o['updated_label']}\n"
            f"  reason: {o['parsed'].get('reason', '')}\n"
        )

    return f"""
{REFEREE_PROMPT}

Transcript:
{transcript}

Examiners' updated judgments:
{debate_text}
"""


# -----------------------------------------------------
# GPT-4o
# -----------------------------------------------------
def call_gpt(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content

def _extract_text_from_responses(resp) -> str:
    """
    Safely extract visible text from a Responses API result.
    Handles both the current SDK shape (c.text is a str)
    and a possible object-with-.value shape.
    """
    chunks = []

    output_items = getattr(resp, "output", None) or []
    for item in output_items:
        contents = getattr(item, "content", None) or []
        for c in contents:
            # Most common case for reasoning models
            t = getattr(c, "type", None)

            # Preferred branch: text is just a string
            if hasattr(c, "text"):
                txt = c.text
                if isinstance(txt, str):
                    chunks.append(txt)
                # If for some reason it's an object with .value
                elif hasattr(txt, "value"):
                    chunks.append(txt.value)

    return "".join(chunks) if chunks else ""


def call_o3(prompt: str) -> str:
    """
    Call the o3 reasoning model using the Responses API.
    Returns plain text that your CEFR parser can consume.
    """
    resp = openai_client.responses.create(
        model="o3",
        input=prompt,
        reasoning={"effort": "medium"},
    )
    return _extract_text_from_responses(resp)


def call_o3_mini(prompt: str) -> str:
    """
    Call the o3-mini reasoning model using the Responses API.
    """
    resp = openai_client.responses.create(
        model="o3-mini",
        input=prompt,
        reasoning={"effort": "medium"},
    )
    return _extract_text_from_responses(resp)




# -----------------------------------------------------
# Claude Sonnet
# -----------------------------------------------------
def call_claude(prompt):
    resp = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text

# -----------------------------------------------------
# Gemini 2.5 Pro
# -----------------------------------------------------
def call_gemini(prompt):
    model = genai.GenerativeModel("models/gemini-2.5-pro")

    try:
        resp = model.generate_content(prompt)
        if resp.candidates and resp.candidates[0].content.parts:
            return "".join(
                p.text for p in resp.candidates[0].content.parts
                if hasattr(p, "text")
            )
    except:
        pass

    return '{"label": "UNK", "reason": "Gemini error"}'

# -----------------------------------------------------
# Grok (simulated with GPT-4o mini)
# -----------------------------------------------------
def call_grok(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "You are GROK.\n" + prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def call_referee(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content

