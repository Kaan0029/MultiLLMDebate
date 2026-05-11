# scripts/prompts_few_shot.py
"""
Few-shot prompt variants for the transcript-only pipeline.
Provides ROUND1_PROMPT with 1, 2, 4, or 8 in-context examples per class.
All examples sourced from the ICNALE training set.

Revised to use ICNALE-style label names: A2_0, B1_1, B1_2, B2_0, XX_0.
"""

# ─────────────────────────────────────────────────────────
# Example pools — 8 examples per class from training set
# ─────────────────────────────────────────────────────────

EXAMPLES = {
    "A2_0": [
        "I agree with that. Maybe some people say it's really cool to smoke and it may be makes their head clear to think other things but I don't think so. There is two side of smoking and we can find that the better side is more than the good side. It really do harm to ourselves and others. It harm it do harm to our hairs and it raised the risk of cancer.",
        "Yes, I agree. I think we can't smoking in restaurants because it is very stinky, and it will make the children a bad effect for children and it is best for everyone to so I agree smoking is bad.",
        "Yeah, as I told before, I disagree with that because the smoke make the people cough and bring the disease. Smoke from that I think not good for our health also and disturb the people who eat in the restaurant. And then I hope the government can do something to decrease the smoker.",
        "I'm disagree with this point that college students should do part time job because spending time on working may affect our studies may affect their studies badly, may affect their grade, it may also affect their health. It may affect their relationship between the teacher and the student.",
        "I think university should choose one part-time job, like a tutor. But at first, you can earn some money around the part-time job because you are only a student. You have a chance to earn a lot of money but a part-time job can give you some.",
        "I agree with this statement because they will have income to their self and they and it will be helpful for their parents and their family because the college students usually have to be their self, and they have to get income by their self after their graduating.",
        "I agree with statement that smoking should be banned in restaurant and any place and smoking is smoking has the bad effects on around the people not only smoking people, children and women have bad influence by smoking people on your health.",
        "I think smoke is so bad so I believe smoking we can smoke in all restaurants and rest in restaurants when parents sometimes take their baby so smoking is so bad but influence for their baby in smoking is.",
    ],
    "B1_1": [
        "I disagree about having part-time jobs for college students, because I don't think that they will learn to manage their own money when they earn their salary during that part-time job, and I don't think so that they can consume that money at the right time because when another thing is that college students don't have the right things to do about those money that they make it as alcohols, cigarettes.",
        "Um. I mean many students, maybe, do so many part-time jobs that they couldn't focus on their study. Um. I think that's not reasonable and is not feasible, since we are still students, um, our first task is to, um, do well in our study. So if we can do well in our study and we still have part-time I think that's fine.",
        "I think I agree with the statement because in restaurant, it is a public place, so there is a lot of people standing and sitting there. The most important thing that if you smoke in the public places, you can risk another people in there, for example you can make other people suck your smoke.",
        "I believe that college students nowadays can have and needs to have part-time jobs. First of all, there are people that need to have part-time jobs and there are some, who chose not to. It's actually optional. For the people who need to have part-time jobs, it is beneficial for them.",
        "Yes, I definitely think that it is important thing to ban the smoking in the restaurant. It is really necessary. I can affect anyone smoking in the restaurant. I think that is really dirty thing, and I don't want to smell any smoke in the restaurant when I am eating my dinner or lunch.",
        "I think college students work part-time job. We are not children and teenager anymore, so we earn money by ourselves. We don't depend on our parents, so I think we we work part-time jobs. I don't like we don't from parents. I don't want it so we.",
        "I don't like people smoking, so I agree. Smoking is harmful to people and it also affect people who don't smoke. I think smoking should be not permitted anywhere inside the buildings and the government should choose some places for the smoker and set on restriction.",
        "I believe that college students really do have to have job before they start work, because first of all, it's their actual first time out of a place where they aren't being told what to do or like given template or what to do.",
    ],
    "B1_2": [
        "College student to part-time job I completely agree because nowadays college student have to learn how to earn money by their work, by their ability to have work in real life. So, he can realize how difficult to earn money. Experience now is so important to have because in the job, we sometimes have an interview that do you have an experience in another job before.",
        "Nowadays, banning smoking in a restaurant is a hot topic between people. Some people think that smoking should be completely banned at restaurant in country but others think smoker should not be completely banned at those places, but most of people who supported the who support the second opinion.",
        "Banning smoking in all the restaurants would be something very difficult, because we have the freedom of what we want to do, and there are people who really can't live their life without smoking, which means that they are already addicted to it, but smoking is actually bad for the health.",
        "The students who are doing part-time job, they have to make some shortcomings. There are some experience by doing part-time job. Students have to work hard more. Some students can't manage their studies with their job and their minds are not free and they have to work hard.",
        "Yes, I do agree with it is important for college students to have a part-time job because some college students leave their home live in the new city so they don't have much money to live, so I think college students can have the opportunity to make money and they can live in the new city.",
        "I think that part-time jobs can be very important to a person, because it makes them more independent in the future. It allows them to experience having a job in an early age, so that after college life they know how to pay taxes or how to handle budgets and to limit their expenditure.",
        "I agree about the argument. Already I have said I really hate smoking smell. Maybe some people think my answer is selfish, but I hate smoking smell. Second, as you know smoking is bad for health and indirect smoke is worse so in these two I mean we should protect nonsmoking people.",
        "Okay, smoking is not good for us, but why? Many people smoke and loved it. I don't know. I think in my country, Indonesia, smoke is like habitual and must. Every people, women or men like smoke, but yeah, I don't know. Now there is smoke, but I think it's bad.",
    ],
    "B2_0": [
        "Some people said that it is important for a college student to do a part-time job. I agree with this because doing part-time job has many advantages just as first, we can make more money for using in the university. And second, we can find the experience that we cannot find in the university or in the classroom. And the third, we can find the new people and new friends and communicate with others.",
        "Yes I do agree that part-time jobs are important, however it is not a necessity because not everyone is able to juggle everything or put down the little time that they have during the holidays to rest. However, it is important because it's just a way to juggle everything and the experience that it provides.",
        "I absolutely agree that smoking must be banned in restaurants, not only in restaurants but also in every place in this world because not only I am an asthmatic person, I really don't like smoking because it really affects my health and smoking is very dangerous for your health.",
        "I think it depends on the person. If your parents is very rich, you don't have to work. You have to study. So, you don't have to part-time job, but your parents is not so rich, so you have to work. It's very important to work and help your parents.",
        "Um. Basically, I agree with the statement that smoking should be banned in all restaurants and in the country. First, let's talk about the restaurants, because when you are smoking you affect the people around you because the smell of smoke is really bad and it really affects the health of the others.",
        "I agree the statement. Nowadays, many restaurant separate smokers and non-smokers, but this measure is not perfect because smoke can easily spread. The smoker can smoke outside the restaurant, but smoking inside the restaurant is very bad thing for non-smokers because while eating the bad smell.",
        "I disagree what they are saying. I think it's okay to I think there is no need to ban smoking completely at restaurants in companies because there are so many people who smokes for if they're stressed. Certainly, smoking is not good for health.",
        "Smoking does harm to those who smokes but as well as those who surrounds those smokers. Those smoker, they also observe this smoke that's from the smokers. They also get infected, they also get harmed from those smokes and they are likely to get cancer.",
    ],
    "XX_0": [
        "Uh, no, I probably, wouldn't agree with the rule of banning smoking in all restaurants in the country. Uh, that would be a little bit totalitarian. Um, each area and each city can decide on its own rules depending on the people living there and and what they decide, um, and what they're hoping for. Uh, to ban smoking from all restaurants would probably upset many smokers.",
        "Smoking should not be unilaterally banned in restaurants. I think smoking should be banned unilaterally everywhere. Smoking is bad. Smoking is a bad habit. I think everybody should still smoke not smoking any tobacco. Tobacco has a lot of nicotine and tar and other carcinogenic substances.",
        "Smoking is a is a health determent, is carcinogenic; carcinogenic means it is cancer causing or is associated with cancer. Smoking in a restaurant is uncomfortable for the people around them as it's not just smokers who are affected by the carcinogens in this tobacco smoke but those who are around them.",
        "Well, basically, yes, I do agree that students should have a part-time job because of the fact that university is not cheap and a lot of students have to, like, ask their parents for money to pay it and sometimes the parents may not have the money to supply them.",
        "I am all in favor of smoking, smoking ban in restaurants all across the country. I think, uh, second-hand smoke is quite a terrible thing, especially for kids, and to be able to go to a restaurant and have a family go there without being air smokers or seeing that or smelling that those bad smells.",
        "So, it's good for students who have a part time job during their college because if you have an extra income, you do not depend so much on your parents, should have received by future employees if student had some experience. And it doesn't have to be matching to your course.",
        "I did not think anyone should force someone to have a part time job, but if you want one, you can have one, if you don't want one, you don't get it. It really depends on a person and you might not have a part time job that they have a hard nature and then you have to study a lot.",
        "Although, there are some benefits to holding a part time job while in college, I believe that the risks outweigh the benefit and here are my reasons. When I first started college, my father told me to be careful. Even I would need to work and help pay for my tuition.",
    ],
}

LABEL_DEFINITIONS = """   Label definitions:

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

LABEL_REASONS = {
    "A2_0": "The speaker fits ICNALE A2_0 / CEFR A2 more than B1. The response communicates a simple opinion on a familiar topic, but expression is limited, repetitive, and often grammatically unstable. The speaker does not sustain a clearly connected explanation at B1 level.",
    "B1_1": "The speaker fits lower CEFR B1, represented by ICNALE B1_1. They communicate a clear position and give reasons on a familiar topic, but control is limited. The main idea is understandable, but fluency, organization, vocabulary range, and accuracy are clearly weaker than upper B1.",
    "B1_2": "The speaker fits upper CEFR B1, represented by ICNALE B1_2. The response gives a more sustained explanation with several connected points and the meaning remains consistently understandable. Grammar errors are still present and the response does not reach B2-level control, range, or detail, but it is more developed and connected than lower B1.",
    "B2_0": "The speaker fits ICNALE B2_0 / CEFR B2+ better than B1. The response is clear, organized, and detailed enough for the task. Grammar is not perfect, but control is sufficient for effective communication, and the response is more structured and fluent than B1-level performance.",
    "XX_0": "The speaker fits the collapsed ICNALE ENS/native-speaker reference category, represented here as XX_0. The response is natural, fluent, idiomatic, and grammatically controlled. This is not treated as CEFR C1 or C2, but as the native-speaker reference category for this task.",
}


def build_examples_block(n_per_class):
    """
    Returns a string containing n_per_class examples per label.
    n_per_class must be 1, 2, 4, or 8.
    """
    assert n_per_class in (1, 2, 4, 8), "n_per_class must be 1, 2, 4, or 8"
    lines = []
    example_num = 1
    for label in ["A2_0", "B1_1", "B1_2", "B2_0", "XX_0"]:
        for i in range(n_per_class):
            transcript = EXAMPLES[label][i]
            reason     = LABEL_REASONS[label]
            lines.append(f'EXAMPLE {example_num} — Label: {label}')
            lines.append(f'Transcript:')
            lines.append(f'"{transcript}"')
            lines.append(f'')
            lines.append(f'Expected JSON:')
            lines.append('{')
            lines.append(f'  "label": "{label}",')
            lines.append(f'  "reason": "{reason}"')
            lines.append('}')
            lines.append('')
            example_num += 1
    return "\n".join(lines)


def build_round1_prompt_few_shot(transcript, n_per_class):
    """
    Builds the Round 1 prompt with n_per_class examples per label.
    """
    examples_block = build_examples_block(n_per_class)
    return f"""You are an official English proficiency examiner evaluating spoken responses.

You will be given a transcript of a student's spoken response.

Your tasks:
1. Assign ONE proficiency label from EXACTLY this set: A2_0, B1_1, B1_2, B2_0, XX_0

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
# IN-CONTEXT EXAMPLES ({n_per_class} per class = {n_per_class * 5} total)
# =========================

{examples_block}
# =========================
# NOW EVALUATE
# =========================

Transcript:
{transcript}
"""