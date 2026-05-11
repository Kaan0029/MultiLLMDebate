"""
Prompts for the audio-only pipeline.
No transcript text is passed — the model hears the audio directly.
In-context examples are also audio clips.

Revised to use ICNALE-style label names:
A2_0, B1_1, B1_2, B2_0, XX_0

Note on XX_0:
ICNALE native-speaker reference files may use XX_1, XX_2, or XX_3
for different ENS participant groups. This prompt collapses native-speaker
reference performance into one output label, XX_0, to preserve a 5-way task.
"""

# Paths to training set audio files used as in-context examples
IN_CONTEXT_AUDIO_PATHS = {
    "A2_0": "/home/ke2461/MultiLLMDebate/ICNALE/ICNALE_SM_Audio/ICNALE_SM_CHN_N600/SM_CHN_SMK2_113_A2_0.mp3",
    "B1_1": "/home/ke2461/MultiLLMDebate/ICNALE/ICNALE_SM_Audio/ICNALE_SM_PHL_N400/SM_PHL_PTJ2_004_B1_1.mp3",
    "B1_2": "/home/ke2461/MultiLLMDebate/ICNALE/ICNALE_SM_Audio/ICNALE_SM_IDN_N400/SM_IDN_PTJ1_005_B1_2.mp3",
    "B2_0": "/home/ke2461/MultiLLMDebate/ICNALE/ICNALE_SM_Audio/ICNALE_SM_THA_N200/SM_THA_PTJ1_007_B2_0.mp3",
    "XX_0": "/home/ke2461/MultiLLMDebate/ICNALE/ICNALE_SM_Audio/ICNALE_SM_ENS_N600/SM_ENS_SMK2_033_XX_2.mp3",
}

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

AUDIO_ROUND1_PROMPT = f"""You are an English proficiency examiner evaluating spoken responses using ICNALE proficiency labels grounded in the CEFR Global Scale.

You have just listened to example audio clips showing each proficiency level.
Now evaluate the new audio clip you are about to hear.

Assign ONE proficiency label from EXACTLY this set: {VALID_LABELS}

{LABEL_DEFINITIONS}

Base your judgment on what you HEAR:
- Pronunciation and accent
- Speech rate and fluency: hesitations, fillers, pauses, smoothness
- Grammatical accuracy as heard
- Vocabulary range as heard
- Coherence and discourse organization
- Whether the response is simple/routine, connected but limited, clearly detailed, or native-like

Return ONLY JSON in this format:
{{
  "label": "<A2_0/B1_1/B1_2/B2_0/XX_0>",
  "reason": "<detailed explanation referencing what you heard and explicitly connecting it to the ICNALE/CEFR label definitions>"
}}"""

AUDIO_ROUND2_TEMPLATE = f"""You are participating in a multi-LLM proficiency evaluation debate.

You previously listened to the audio and gave your judgment.

### Your own Round 1 judgment:
Label: {{self_label}}
Reason: {{self_reason}}

### Other examiners' judgments (they also heard the same audio):
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
