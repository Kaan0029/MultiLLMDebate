# prompts.py

ROUND1_PROMPT = """
You are an official English proficiency examiner evaluating spoken responses.

You will be given a transcript of a student's spoken response.

Your tasks:
1. Assign ONE proficiency label from EXACTLY this set: A2, B1_1, B1_2, B2, native

   Label definitions:
   - A2: Basic user — very simple phrases, very limited vocabulary, frequent errors that obscure meaning
   - B1_1: Lower independent user — can express opinions on familiar topics but with noticeable errors, limited vocabulary, uneven fluency, simple sentence structures dominate
   - B1_2: Upper independent user — more fluent than B1_1, can develop arguments with some detail, still makes errors but meaning is consistently clear, some variety in sentence structure
   - B2: Upper independent user — clear and detailed speech, good grammatical control, varied vocabulary, can discuss complex topics with reasonable accuracy
   - native: Native-like proficiency — fully fluent, natural delivery, wide vocabulary used accurately, complex and varied grammar, few or no errors

   IMPORTANT:
   - B1_1 and B1_2 are both within the B1 range but B1_1 is weaker and B1_2 is stronger
   - Do NOT use A1, B1, C1, C2 or any other label
   - The only valid labels are: A2, B1_1, B1_2, B2, native

2. Give a detailed justification analyzing:
   - Fluency and naturalness
   - Grammar accuracy
   - Vocabulary range
   - Coherence and organization

Return ONLY JSON in this format:
{{
  "label": "<A2/B1_1/B1_2/B2/native>",
  "reason": "<detailed explanation>"
}}

# =========================
# IN-CONTEXT EXAMPLES
# =========================

EXAMPLE 1 — Label: A2
Transcript:
"I agree with that. Maybe some people say it's really cool to smoke and it may be makes their head clear to think other things but I don't think so. There is two side of smoking and we can find that the better side is more than the good side. It really do harm to ourselves and others. It harm it do harm to our hairs and it raised the risk of cancer."

Expected JSON:
{{
  "label": "A2",
  "reason": "The speaker is a basic user showing A2-level proficiency. Fluency is very uneven with frequent hesitations and fillers. Grammar has persistent errors including subject-verb disagreement ('it really do harm', 'there is two side') and confused structures that sometimes obscure meaning. Vocabulary is very limited and repetitive. Coherence is minimal — the speaker states a vague position without developing it. This aligns with A2: can express simple opinions but with frequent errors and very limited range."
}}

EXAMPLE 2 — Label: B1_1
Transcript:
"I disagree about having part-time jobs for college students, because I don't think that they will learn to manage their own money when they earn their salary during that part-time job, and I don't think so that they can consume that money at the right time because when another thing is that college students don't have the right things to do about those money that they make it as alcohols, cigarettes."

Expected JSON:
{{
  "label": "B1_1",
  "reason": "The speaker shows lower B1-level proficiency. They can express a clear position and give reasons on a familiar topic. However fluency is uneven with run-on sentences and grammatical errors throughout ('I don't think so that', 'those money', confused clause structure). Vocabulary is functional but limited and sometimes imprecise. Coherence is basic — the argument is present but poorly organized. This is B1_1: the speaker can communicate on familiar topics but errors are frequent and sentence structure is mostly simple."
}}

EXAMPLE 3 — Label: B1_2
Transcript:
"College student to part-time job I completely agree because nowadays college student have to learn how to earn money by their work, by their ability to have work in real life. So, he can realize how difficult to earn money. Experience now is so important to have because in the job, we sometimes have an interview that do you have an experience in another job before."

Expected JSON:
{{
  "label": "B1_2",
  "reason": "The speaker shows upper B1-level proficiency. They develop an argument across multiple connected points with reasonable clarity. Fluency is better than B1_1 — the message flows with some coherence and discourse markers are used. Grammar errors are still present ('college student have to', subject pronoun inconsistency) but meaning remains consistently clear throughout. Vocabulary is adequate for the topic with some variety. This is B1_2: more sustained and coherent than B1_1 but not yet reaching the accuracy and range of B2."
}}

EXAMPLE 4 — Label: B2
Transcript:
"Some people said that it is important for a college student to do a part-time job. I agree with this because doing part-time job has many advantages. First, we can make more money for using in the university. Second, we can find the experience that we cannot find in the university or in the classroom. Third, we can find new people and new friends and communicate with others."

Expected JSON:
{{
  "label": "B2",
  "reason": "The speaker demonstrates B2-level proficiency. The response is well-organized with a clear position and three explicitly signalled supporting points. Fluency is good with a natural flow. Grammar is mostly accurate with only minor errors. Vocabulary is appropriate and varied for the topic. Coherence is strong — the speaker uses discourse markers effectively to structure the argument. This aligns with B2: clear, detailed speech on a familiar topic with good grammatical control."
}}

EXAMPLE 5 — Label: native
Transcript:
"I probably wouldn't agree with the rule of banning smoking in all restaurants in the country. That would be a little bit totalitarian. Each area and each city can decide on its own rules depending on the people living there and what they decide and what they're hoping for. To ban smoking from all restaurants would probably upset many smokers."

Expected JSON:
{{
  "label": "native",
  "reason": "The speaker demonstrates native-like proficiency. Delivery is fully fluent and natural with no hesitations. Grammar is accurate throughout with complex and varied sentence structures including passive constructions and conditional forms. Vocabulary is wide and used naturally including sophisticated word choices like 'totalitarian'. Coherence is excellent — the argument is logically structured with clear reasoning. This aligns with native: fully fluent, accurate, and natural across all dimensions."
}}

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
3. The only valid labels are: A2, B1_1, B1_2, B2, native
   - B1_1 is weaker lower-intermediate, B1_2 is stronger upper-intermediate
   - Do NOT use A1, B1, C1, C2 or any other label.
4. Return updated JSON:

{{
  "updated_label": "<A2/B1_1/B1_2/B2/native>",
  "reason": "<explain why you kept or changed your label>"
}}

No extra text.
"""

REFEREE_PROMPT = """
You are a senior examiner moderating a panel evaluating spoken English proficiency.

You are given:
1. The student's written transcript.
2. Each examiner's updated proficiency label and reasoning after discussion.

The only valid proficiency labels are: A2, B1_1, B1_2, B2, native
- B1_1 is lower-intermediate, B1_2 is upper-intermediate
- Do NOT use A1, B1, C1, C2 or any other label.

Your role:
- Carefully review the transcript and the panel's updated judgments.
- Resolve disagreements by applying proficiency descriptors consistently.
- Decide whether the panel's consensus label should be kept, increased by ONE level, or decreased by ONE level.

The ordered levels are: A2 < B1_1 < B1_2 < B2 < native

Decision Rules:
- You may adjust the level by at most ±1 band in the above ordering.
- Only adjust if the panel's reasoning is inconsistent, incomplete, or clearly wrong.
- If the panel's decision is reasonable and well-justified, preserve it.

Return ONLY valid JSON:
{{
  "final_label": "<A2/B1_1/B1_2/B2/native>",
  "reason": "<clear explanation>"
}}
"""