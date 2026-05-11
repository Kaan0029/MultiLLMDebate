# prompts.py
"""
Transcript-only prompts for ICNALE spoken-response proficiency classification.

Revised to use ICNALE-style label names:
A2_0, B1_1, B1_2, B2_0, XX_0

Note on XX_0:
ICNALE native-speaker reference files may use XX_1, XX_2, or XX_3
for different ENS participant groups. This prompt collapses native-speaker
reference performance into one output label, XX_0, to preserve a 5-way task.
"""

VALID_LABELS = "A2_0, B1_1, B1_2, B2_0, XX_0"

LABEL_DEFINITIONS = """Label definitions:

These labels follow the ICNALE proficiency-label convention and are grounded in the CEFR Global Scale.

ICNALE-to-CEFR mapping:
- A2_0: ICNALE learner label corresponding to CEFR A2.
- B1_1: ICNALE learner label corresponding to the lower subdivision of CEFR B1.
- B1_2: ICNALE learner label corresponding to the upper subdivision of CEFR B1.
- B2_0: ICNALE learner label described by ICNALE as B2+ and grounded in the CEFR B2 band.
- XX_0: collapsed ICNALE native-speaker reference label for this task. ICNALE native-speaker files may use XX_1, XX_2, or XX_3 depending on native-speaker subgroup, but this task uses one single native-speaker reference output label, XX_0. This is not a CEFR learner level.

Official CEFR Global Scale descriptors:

A2 — Basic User:
Can understand sentences and frequently used expressions related to areas of most immediate relevance, such as very basic personal and family information, shopping, local geography, and employment. Can communicate in simple and routine tasks requiring a simple and direct exchange of information on familiar and routine matters. Can describe in simple terms aspects of his/her background, immediate environment, and matters in areas of immediate need.

B1 — Independent User:
Can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc. Can deal with most situations likely to arise whilst travelling in an area where the language is spoken. Can produce simple connected text on topics which are familiar or of personal interest. Can describe experiences and events, dreams, hopes, and ambitions and briefly give reasons and explanations for opinions and plans.

B2 — Independent User:
Can understand the main ideas of complex text on both concrete and abstract topics, including technical discussions in his/her field of specialisation. Can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party. Can produce clear, detailed text on a wide range of subjects and explain a viewpoint on a topical issue, giving the advantages and disadvantages of various options.

Operational distinction for this task:
- A2_0: Use this label when the response fits CEFR A2 more than B1: simple/routine communication, limited range, simple descriptions, and difficulty sustaining connected explanation.
- B1_1: Use this label for lower CEFR B1 performance: the speaker can communicate the main idea and give some reasons on familiar topics, but control, fluency, vocabulary range, and organization are clearly limited.
- B1_2: Use this label for upper CEFR B1 performance: the speaker gives a more sustained, connected, and understandable response than B1_1, but does not yet meet B2-level fluency, range, detail, or control.
- B2_0: Use this label when the response fits the ICNALE B2_0 / CEFR B2+ learner category: clear, detailed, well-organized communication with sufficient fluency, vocabulary range, and grammatical control to discuss the topic effectively.
- XX_0: Use this label only for ICNALE ENS/native-speaker reference-level performance: natural, fully fluent, idiomatic, and native-like speech. Do not assign XX_0 merely because the response is strong B2_0.

IMPORTANT:
- The only valid output labels are: A2_0, B1_1, B1_2, B2_0, XX_0.
- Do NOT output A1, A2, B1, B2, B2+, C1, C2, ENS, native, XX_1, XX_2, XX_3, or any other label.
- Because CEFR has one B1 band but ICNALE splits it, decide between B1_1 and B1_2 by relative strength within B1.
- B1_1 and B1_2 are both B1-level labels; B1_1 is lower B1 and B1_2 is upper B1.
- Treat XX_0 as a separate ICNALE native-speaker reference category, not as CEFR C1 or C2.
- Base the label on the actual evidence in the response, not on topic knowledge or agreement with the speaker's opinion."""

ROUND1_PROMPT = f"""
You are an English proficiency examiner evaluating spoken responses using ICNALE proficiency labels grounded in the CEFR Global Scale.

You will be given a transcript of a student's spoken response.

Your tasks:
1. Assign ONE proficiency label from EXACTLY this set: {VALID_LABELS}

{LABEL_DEFINITIONS}

2. Give a detailed justification analyzing:
   - CEFR/ICNALE label fit
   - Grammar accuracy
   - Vocabulary range
   - Coherence and organization
   - Complexity and connectedness of expression
   - Any transcript-visible fluency markers, such as repetitions, false starts, fillers, or fragmented syntax

Return ONLY JSON in this format:
{{
  "label": "<A2_0/B1_1/B1_2/B2_0/XX_0>",
  "reason": "<detailed explanation>"
}}

# =========================
# IN-CONTEXT EXAMPLES
# =========================

EXAMPLE 1 — Label: A2_0
Transcript:
"I agree with that. Maybe some people say it's really cool to smoke and it may be makes their head clear to think other things but I don't think so. There is two side of smoking and we can find that the better side is more than the good side. It really do harm to ourselves and others. It harm it do harm to our hairs and it raised the risk of cancer."

Expected JSON:
{{
  "label": "A2_0",
  "reason": "The speaker fits ICNALE A2_0 / CEFR A2 more than B1. The response communicates a simple opinion on a familiar topic, but expression is limited, repetitive, and often grammatically unstable. Errors such as subject-verb disagreement and confused phrasing sometimes interfere with clarity. The speaker does not sustain a clearly connected explanation at B1 level."
}}

EXAMPLE 2 — Label: B1_1
Transcript:
"I disagree about having part-time jobs for college students, because I don't think that they will learn to manage their own money when they earn their salary during that part-time job, and I don't think so that they can consume that money at the right time because when another thing is that college students don't have the right things to do about those money that they make it as alcohols, cigarettes."

Expected JSON:
{{
  "label": "B1_1",
  "reason": "The speaker fits lower CEFR B1, represented by ICNALE B1_1. They communicate a clear position and give reasons on a familiar topic, but control is limited. The response contains run-on structure, frequent grammatical errors, and imprecise vocabulary. The main idea is understandable, but fluency, organization, and accuracy are clearly weaker than upper B1."
}}

EXAMPLE 3 — Label: B1_2
Transcript:
"College student to part-time job I completely agree because nowadays college student have to learn how to earn money by their work, by their ability to have work in real life. So, he can realize how difficult to earn money. Experience now is so important to have because in the job, we sometimes have an interview that do you have an experience in another job before."

Expected JSON:
{{
  "label": "B1_2",
  "reason": "The speaker fits upper CEFR B1, represented by ICNALE B1_2. The response gives a sustained explanation with several connected points and the meaning remains consistently understandable. Grammar errors are still frequent and the response does not reach B2-level control, range, or detail, but it is more developed and connected than lower B1."
}}

EXAMPLE 4 — Label: B2_0
Transcript:
"Some people said that it is important for a college student to do a part-time job. I agree with this because doing part-time job has many advantages. First, we can make more money for using in the university. Second, we can find the experience that we cannot find in the university or in the classroom. Third, we can find new people and new friends and communicate with others."

Expected JSON:
{{
  "label": "B2_0",
  "reason": "The speaker fits ICNALE B2_0 / CEFR B2+ better than B1. The response is clear, organized, and detailed enough for the task, with explicit discourse markers and several supporting points. Grammar is not perfect, but control is sufficient for effective communication, and the response is more structured and fluent than B1-level performance."
}}

EXAMPLE 5 — Label: XX_0
Transcript:
"I probably wouldn't agree with the rule of banning smoking in all restaurants in the country. That would be a little bit totalitarian. Each area and each city can decide on its own rules depending on the people living there and what they decide and what they're hoping for. To ban smoking from all restaurants would probably upset many smokers."

Expected JSON:
{{
  "label": "XX_0",
  "reason": "The speaker fits the collapsed ICNALE ENS/native-speaker reference category, represented here as XX_0. The response is natural, fluent, idiomatic, and grammatically controlled. Vocabulary and phrasing are native-like, and the argument is expressed smoothly without learner-like limitations. This is not being treated as CEFR C1 or C2, but as the native-speaker reference category for this task."
}}

# =========================
# NOW EVALUATE
# =========================

Transcript:
{{transcript}}
"""


ROUND2_TEMPLATE = f"""
You are participating in a multi-LLM proficiency evaluation debate.

### Transcript
{{transcript}}

### Your own Round 1 judgment:
Label: {{self_label}}
Reason: {{self_reason}}

### Other examiners' judgments:
{{other_judgments}}

---

Your tasks:
1. Reconsider your proficiency label after reading the other examiners' arguments.
2. Apply the ICNALE-to-CEFR mapping consistently:
   - A2_0 = ICNALE A2_0 / CEFR A2
   - B1_1 = lower CEFR B1
   - B1_2 = upper CEFR B1
   - B2_0 = ICNALE B2_0 / CEFR B2+
   - XX_0 = collapsed ICNALE ENS/native-speaker reference category, not CEFR C1/C2
3. Update your label ONLY if the other examiners provide stronger evidence from the response.
4. The only valid labels are: {VALID_LABELS}.
5. Do NOT output A1, A2, B1, B2, B2+, C1, C2, ENS, native, XX_1, XX_2, XX_3, or any other label.

Return ONLY JSON:
{{
  "updated_label": "<A2_0/B1_1/B1_2/B2_0/XX_0>",
  "reason": "<explain why you kept or changed your label>"
}}

No extra text.
"""

REFEREE_PROMPT = f"""
You are a senior examiner moderating a panel evaluating spoken English proficiency.

You are given:
1. The student's written transcript.
2. Each examiner's updated proficiency label and reasoning after discussion.

The labels follow the ICNALE proficiency-label convention grounded in CEFR:
- A2_0 = ICNALE A2_0 / CEFR A2
- B1_1 = lower CEFR B1
- B1_2 = upper CEFR B1
- B2_0 = ICNALE B2_0 / CEFR B2+
- XX_0 = collapsed ICNALE ENS/native-speaker reference group, not CEFR C1/C2

The only valid proficiency labels are: {VALID_LABELS}.
Do NOT output A1, A2, B1, B2, B2+, C1, C2, ENS, native, XX_1, XX_2, XX_3, or any other label.

Your role:
- Carefully review the transcript and the panel's updated judgments.
- Resolve disagreements by applying the ICNALE-to-CEFR mapping and CEFR descriptors consistently.
- Decide whether the panel's consensus label should be kept, increased by ONE level, or decreased by ONE level.

The ordered levels are: A2_0 < B1_1 < B1_2 < B2_0 < XX_0

Decision Rules:
- You may adjust the level by at most ±1 band in the above ordering.
- Only adjust if the panel's reasoning is inconsistent, incomplete, or clearly wrong.
- If the panel's decision is reasonable and well-justified, preserve it.
- Treat XX_0 as a native-speaker reference category, not as CEFR C1 or C2.

Return ONLY valid JSON:
{{
  "final_label": "<A2_0/B1_1/B1_2/B2_0/XX_0>",
  "reason": "<clear explanation>"
}}
"""
