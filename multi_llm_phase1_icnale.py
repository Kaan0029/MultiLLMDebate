import os
import csv
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import Counter

# ----------------------------
# LLM SDK IMPORTS
# ----------------------------
from openai import OpenAI                  # GPT-4o + Whisper
import anthropic                           # Claude
import google.generativeai as genai        # Gemini
from xai import Client as GrokClient       # Grok-2


# ----------------------------
# INIT CLIENTS
# ----------------------------
openai_client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
grok_client = GrokClient(api_key=os.getenv("XAI_API_KEY"))


# ----------------------------
# DATA STRUCTURES
# ----------------------------
@dataclass
class Sample:
    audio_path: str
    cefr_label: str


@dataclass
class AgentConfig:
    name: str
    provider: str
    model: str


@dataclass
class AgentResponse:
    agent_name: str
    cefr_label: str
    justification: str


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1"]


# ----------------------------
# CSV LOADING
# ----------------------------
def load_icnale_csv(path: str) -> List[Sample]:
    samples = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                Sample(
                    audio_path=row["audio_path"],
                    cefr_label=row["cefr_label"].strip().upper()
                )
            )
    return samples


# ----------------------------
# AUDIO → TRANSCRIPT
# ----------------------------
def transcribe_audio(audio_path: str) -> str:
    """Use OpenAI Whisper model."""
    with open(audio_path, "rb") as f:
        result = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
        return result.text.strip()


# ----------------------------
# BUILD PROMPT
# ----------------------------
SYSTEM_RUBRIC = """
You are an English CEFR examiner. 
Assign one of: A1, A2, B1, B2, C1.
Return ONLY JSON:
{"cefr_label": "...", "justification": "..."}
"""


def build_prompt(transcript: str, peer_responses=None):
    if peer_responses:
        peer_block = "\n".join(
            [f"- {p.agent_name}: {p.cefr_label}. Reason: {p.justification}"
             for p in peer_responses]
        )
        peer_section = f"\nOther examiners said:\n{peer_block}\n\nRe-evaluate your label if needed."
    else:
        peer_section = ""

    return f"""
Transcript:
{transcript}

{peer_section}

Use CEFR rubric. Output JSON only.
"""


# ----------------------------
# LLM CALL FUNCTIONS
# ----------------------------
def call_openai(model: str, system: str, prompt: str):
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    return resp.choices[0].message.content


def call_claude(model: str, system: str, prompt: str):
    msg = anthropic_client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_gemini(model: str, system: str, prompt: str):
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(system + "\n" + prompt)
    return response.text


def call_grok(model: str, system: str, prompt: str):
    response = grok_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    return response.choices[0].message["content"]


def llm_call(agent: AgentConfig, prompt: str):
    if agent.provider == "openai":
        return call_openai(agent.model, SYSTEM_RUBRIC, prompt)
    if agent.provider == "claude":
        return call_claude(agent.model, SYSTEM_RUBRIC, prompt)
    if agent.provider == "gemini":
        return call_gemini(agent.model, SYSTEM_RUBRIC, prompt)
    if agent.provider == "grok":
        return call_grok(agent.model, SYSTEM_RUBRIC, prompt)
    raise ValueError("Unknown provider")


# ----------------------------
# PARSE JSON OUTPUT
# ----------------------------
def parse_llm_json(txt: str):
    try:
        cleaned = txt.strip().strip("`").replace("json", "")
        data = json.loads(cleaned)
        label = data["cefr_label"].upper()
        if label not in CEFR_LEVELS:
            return "B1", data.get("justification", "")
        return label, data.get("justification", "")
    except:
        return "B1", "Could not parse. Defaulted to B1."


# ----------------------------
# MULTI-AGENT DISCUSSION
# ----------------------------
def one_round(agents: List[AgentConfig], transcript: str, peer=None):
    responses = []
    for agent in agents:
        prompt = build_prompt(transcript, peer)
        raw = llm_call(agent, prompt)
        cefr, just = parse_llm_json(raw)
        responses.append(
            AgentResponse(
                agent_name=agent.name,
                cefr_label=cefr,
                justification=just
            )
        )
    return responses


def run_discussion(transcript: str, agents: List[AgentConfig]):
    # Round 1 — independent
    round1 = one_round(agents, transcript)

    # Round 2 — see each other's votes
    round2 = one_round(agents, transcript, peer=round1)

    return round2


# ----------------------------
# MAJORITY VOTING
# ----------------------------
def majority_vote(responses: List[AgentResponse]):
    counts = Counter(r.cefr_label for r in responses)
    best = counts.most_common(1)[0][0]
    return best


# ----------------------------
# MAIN EVALUATION
# ----------------------------
def evaluate_icnale(csv_path: str):
    samples = load_icnale_csv(csv_path)

    agents = [
        AgentConfig("GPT4o", "openai", "gpt-4o"),
        AgentConfig("Claude", "claude", "claude-3-5-sonnet-20240620"),
        AgentConfig("Gemini", "gemini", "gemini-1.5-flash"),
        AgentConfig("Grok2", "grok", "grok-2-latest"),
    ]

    correct = 0

    for i, sample in enumerate(samples, start=1):
        print(f"\n=== SAMPLE {i} ===")
        print("Audio:", sample.audio_path)
        print("GT Label:", sample.cefr_label)

        transcript = transcribe_audio(sample.audio_path)
        print("Transcript:", transcript)

        responses = run_discussion(transcript, agents)
        final_label = majority_vote(responses)

        print("Agent votes:")
        for r in responses:
            print(f" - {r.agent_name}: {r.cefr_label}")

        print("Final Label:", final_label)

        if final_label == sample.cefr_label:
            correct += 1

    acc = correct / len(samples)
    print("\n==============================")
    print("FINAL ACCURACY:", acc)
    print("==============================")


# ----------------------------
# ENTRY
# ----------------------------
if __name__ == "__main__":
    evaluate_icnale("icnale_subset.csv")
