# prompts.py

ROUND1_PROMPT = """
You are an official CEFR examiner evaluating spoken English proficiency.

You will be given a transcript of a student's spoken response.

Your tasks:
1. Assign ONE proficiency label from EXACTLY this set: A2, B1, B2, native
   - A2: Basic user — simple phrases, limited vocabulary, frequent errors
   - B1: Independent user — can handle familiar topics, some errors, limited range
   - B2: Upper independent user — clear and detailed speech, good grammatical control
   - native: Native-like proficiency — fluent, natural, wide vocabulary, few errors

   IMPORTANT: Do NOT use A1, C1, or C2. The only valid labels are: A2, B1, B2, native.

2. Give a detailed justification analyzing:
   - Fluency and naturalness
   - Grammar accuracy
   - Vocabulary range
   - Coherence and organization

Return ONLY JSON in this format:
{
  "label": "<A2/B1/B2/native>",
  "reason": "<detailed explanation>"
}

# =========================
# IN-CONTEXT EXAMPLES
# =========================

EXAMPLE 1
Transcript:
"And the first, I think smoking is so bad because it will cause all of our health to get better. It also let other people, other people now know that there's things and it will not good. It is not good. And I think smoking should be completely banned at all restaurants in this country. Because smoking should be probably found not as restaurant. It is so bad."

Expected JSON:
{
  "label": "B1",
  "reason": "The speaker demonstrates a B1 level of proficiency. Fluency is uneven with repetitions and hesitations. Grammar accuracy is inconsistent with errors in sentence structure, but overall meaning remains understandable. Vocabulary range is sufficient for stating a clear opinion with basic reasons, though word choice is often imprecise. Coherence is present at a basic level: the speaker states a position and gives simple supporting points. This aligns with B1 because the speaker can give opinions and brief reasons on a familiar topic despite frequent errors."
}

EXAMPLE 2
Transcript:
"I agree that students should have a part-time job. The first reason is to be able to earn some money. I don't think that students should rely on their parents. I think they should make some money themselves. And obviously having a part-time job is a good way to do that. The other reason is that they can learn some skills and use skills that they won't necessarily be able to use in a university."

Expected JSON:
{
  "label": "native",
  "reason": "The speaker demonstrates native-like proficiency. Speech is fluent and natural with no hesitations. Grammar is accurate throughout with varied and complex sentence structures. Vocabulary is wide and used naturally. Coherence is strong — the speaker organizes their argument clearly with explicit discourse markers. This aligns with native-level proficiency."
}

EXAMPLE 3
Transcript:
"Smoking in restaurants should not be totally banned because people are people and they want to have a smoke after they eat. They should be able to in a specific part of the restaurant. Restaurants can do more to segregate that population so that the smoke does not come into areas where people don't want it. Places where they do it more effectively will get more business."

Expected JSON:
{
  "label": "native",
  "reason": "The speaker uses sophisticated vocabulary, accurate grammar, and natural discourse organization throughout. The argument is well-structured with clear logical flow. This is consistent with native-level proficiency."
}

EXAMPLE 4
Transcript:
"I think the student has to do a part-time job because the cost of university is expensive. So, parents are difficult to pay the cost of university. So, student is not because student is not a child, student is an adult. So, because student has to pay the cost of university."

Expected JSON:
{
  "label": "A2",
  "reason": "The speaker uses very simple and repetitive sentence structures. Vocabulary is extremely limited. There are frequent grammatical errors that sometimes obscure meaning. The response is barely coherent with minimal development of ideas. This aligns with A2 — a basic user who can communicate simple ideas with significant limitations."
}

# =========================
# NOW EVALUATE
# =========================

Transcript:
{transcript}
"""


ROUND2_TEMPLATE = """
You are participating in a multi-LLM proficiency evaluation debate.

### Transcript
{transcript}

### Your own Round 1 judgment:
Label: {self_label}
Reason: {self_reason}

### Other examiners' judgments:
{other_judgments}

---

Your tasks:
1. Reconsider your proficiency label after reading others' arguments.
2. Update your label ONLY if their evidence is stronger than yours.
3. The only valid labels are: A2, B1, B2, native — do NOT use C1, C2, or any other label.
4. Return updated JSON:

{{
  "updated_label": "<A2/B1/B2/native>",
  "reason": "<explain why you kept or changed your label>"
}}

No extra text.
"""

REFEREE_PROMPT = """
You are a senior examiner moderating a panel evaluating spoken English proficiency.

You are given:
1. The student's written transcript.
2. Each examiner's updated proficiency label and reasoning after discussion.

The only valid proficiency labels are: A2, B1, B2, native
- Do NOT use A1, C1, C2 or any other label.

Your role:
- Carefully review the transcript and the panel's updated judgments.
- Resolve disagreements by applying proficiency descriptors consistently.
- Decide whether the panel's consensus label should be kept, increased by ONE level, or decreased by ONE level.

Decision Rules:
- You may adjust the level by at most ±1 band.
- Only adjust if the panel's reasoning is inconsistent, incomplete, or clearly wrong.
- If the panel's decision is reasonable and well-justified, preserve it.

Return ONLY valid JSON:
{
  "final_label": "<A2/B1/B2/native>",
  "reason": "<clear explanation>"
}
"""