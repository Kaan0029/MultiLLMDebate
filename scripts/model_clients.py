# model_clients.py
import os
import base64
from dotenv import load_dotenv

from anthropic import Anthropic
from openai import OpenAI

# New Gemini SDK — replaces the deprecated google.generativeai package.
# Install with: pip install google-genai
from google import genai as google_genai
from google.genai import types as google_genai_types

from prompts import ROUND1_PROMPT, ROUND2_TEMPLATE, REFEREE_PROMPT
from utils import safe_parse_json
import torch

load_dotenv()

OPENAI_KEY    = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY    = os.getenv("GEMINI_API_KEY")

openai_client  = OpenAI(api_key=OPENAI_KEY)
claude_client  = Anthropic(api_key=ANTHROPIC_KEY)
gemini_client  = google_genai.Client(api_key=GEMINI_KEY)   # new-SDK client

# -----------------------------------------------------
# Whisper ASR (lazy-loaded)
# -----------------------------------------------------

from faster_whisper import WhisperModel

asr = None

def get_asr():
    global asr
    if asr is None:
        asr = WhisperModel("small.en")
    return asr


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
        resp = openai_client.files.create(file=f, purpose="assistants")
    return resp.id


# -----------------------------------------------------
# Prompt builders (text pipeline)
# -----------------------------------------------------

def build_round1_prompt(transcript):
    return ROUND1_PROMPT.replace("{transcript}", transcript)


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


# -----------------------------------------------------
# GPT-4o (text)
# -----------------------------------------------------

def call_gpt(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content

def call_gpt55(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def call_gpt54(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-5.4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def call_gpt54_mini(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def call_gpt41(prompt):
    resp = openai_client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


# -----------------------------------------------------
# o3 / o3-mini  (Responses API)
# -----------------------------------------------------

def _extract_text_from_responses(resp) -> str:
    chunks = []
    for item in (getattr(resp, "output", None) or []):
        for c in (getattr(item, "content", None) or []):
            if hasattr(c, "text"):
                txt = c.text
                if isinstance(txt, str):
                    chunks.append(txt)
                elif hasattr(txt, "value"):
                    chunks.append(txt.value)
    return "".join(chunks) if chunks else ""


def call_o3(prompt: str) -> str:
    resp = openai_client.responses.create(
        model="o3",
        input=prompt,
        reasoning={"effort": "medium"},
    )
    return _extract_text_from_responses(resp)


def call_o3_mini(prompt: str) -> str:
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
# Gemini 2.5 Pro  (text, new SDK)
# -----------------------------------------------------

def call_gemini(prompt):
    try:
        resp = gemini_client.models.generate_content(
            model="models/gemini-2.5-pro",
            contents=prompt,
        )
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text.strip():
            return text
    except Exception:
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


def call_referee_audio(prompt, audio_path):
    def encode_audio(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "input_audio",
            "input_audio": {"data": encode_audio(audio_path), "format": "mp3"},
        },
    ]
    resp = openai_client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content


# -----------------------------------------------------
# Shared OpenAI audio content builder
# -----------------------------------------------------

def _build_openai_audio_content(prompt_text, audio_path, in_context_examples=None):
    """
    Build OpenAI multimodal content list for audio judging.
    in_context_examples: iterable of (label, path) pairs.
    Returns (content_list, request_preview_dict).
    """
    def encode_audio(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    content = []
    request_preview = {
        "instructions": prompt_text,
        "in_context_examples": [],
        "test_audio_path": audio_path,
    }

    if in_context_examples:
        content.append({
            "type": "text",
            "text": (
                "Below are example audio clips with their correct proficiency labels. "
                "Listen carefully to understand the difference between levels.\n"
            ),
        })
        for label, ex_path in in_context_examples:
            request_preview["in_context_examples"].append(
                {"label": label, "audio_path": ex_path}
            )
            content.append({"type": "text", "text": f"EXAMPLE — Label: {label}"})
            content.append({
                "type": "input_audio",
                "input_audio": {"data": encode_audio(ex_path), "format": "mp3"},
            })

    content.append({"type": "text", "text": prompt_text})
    content.append({"type": "text", "text": "Now evaluate the following audio:"})
    content.append({
        "type": "input_audio",
        "input_audio": {"data": encode_audio(audio_path), "format": "mp3"},
    })
    return content, request_preview


# -----------------------------------------------------
# OpenAI audio models  (gpt-4o-audio-preview, gpt-5.5)
# -----------------------------------------------------

def call_openai_audio_model(prompt_text, audio_path, model_name, in_context_examples=None):
    """
    Generic OpenAI audio call for models that accept input_audio content.
    Returns (response_text, request_preview).
    """
    content, request_preview = _build_openai_audio_content(
        prompt_text=prompt_text,
        audio_path=audio_path,
        in_context_examples=in_context_examples,
    )
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content, request_preview


def call_gpt_audio(prompt_text, audio_path, in_context_audio_paths=None):
    """Legacy single-dict interface kept for backward compatibility."""
    examples = list(in_context_audio_paths.items()) if in_context_audio_paths else None
    text, _ = call_openai_audio_model(
        prompt_text=prompt_text,
        audio_path=audio_path,
        model_name="gpt-4o-audio-preview",
        in_context_examples=examples,
    )
    return text


def call_gpt55_audio(prompt_text, audio_path, in_context_examples=None):
    return call_openai_audio_model(
        prompt_text=prompt_text,
        audio_path=audio_path,
        model_name="gpt-5.5",
        in_context_examples=in_context_examples,
    )


def call_gpt4o_audio_preview(prompt_text, audio_path, in_context_examples=None):
    return call_openai_audio_model(
        prompt_text=prompt_text,
        audio_path=audio_path,
        model_name="gpt-4o-audio-preview",
        in_context_examples=in_context_examples,
    )


# -----------------------------------------------------
# Gemini 3 Flash audio  (new SDK)
# -----------------------------------------------------

def call_gemini_3_flash_audio(prompt_text, audio_path, in_context_examples=None):
    """
    Gemini audio caller using the new google.genai SDK with inline audio parts.
    Returns (response_text, request_preview).
    """
    request_preview = {
        "instructions": prompt_text,
        "in_context_examples": [],
        "test_audio_path": audio_path,
    }

    contents = []

    if in_context_examples:
        contents.append(
            "Below are example audio clips with their correct proficiency labels. "
            "Listen carefully to understand the difference between levels."
        )
        for label, ex_path in in_context_examples:
            request_preview["in_context_examples"].append(
                {"label": label, "audio_path": ex_path}
            )
            contents.append(f"EXAMPLE — Label: {label}")
            with open(ex_path, "rb") as f:
                audio_bytes = f.read()
            contents.append(
                google_genai_types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/mpeg",
                )
            )

    contents.append(prompt_text)
    contents.append("Now evaluate the following audio:")
    with open(audio_path, "rb") as f:
        test_audio_bytes = f.read()
    contents.append(
        google_genai_types.Part.from_bytes(
            data=test_audio_bytes,
            mime_type="audio/mpeg",
        )
    )

    try:
        resp = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
        )
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text.strip():
            return text, request_preview
    except Exception as e:
        err_msg = str(e).replace('"', "'")
        return (
            f'{{"label": "UNK", "reason": "Gemini 3 Flash audio error: {err_msg}"}}',
            request_preview,
        )

    return '{"label": "UNK", "reason": "Gemini 3 Flash empty response"}', request_preview