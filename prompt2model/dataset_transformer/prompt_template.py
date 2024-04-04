"""Prompt template for dataset transformer."""

import json

IMPLICATURES ={
    "task_description":"Predict whether Speaker 2's answer to Speaker 1 counts as a yes or as a no",
    "samples":"""input=\n\n Speaker 1: 'Have you found him yet? ' Speaker 2: 'We're still looking.' \n \noutput=no""",
    "plan":"""1. Create a "Speaker 1" field using the 'text' fields. Expand the text such that speaker 2 can answer in some form of yes or no.
2. Create a "Speaker 2" field responding to the speaker with either a yes or a no. Ensure that the yes or no is not said directly, but in a roundabout manner.
3. Combine "Speaker 1" and "Speaker 2" fields to create "input".
4. Create an "output" field based on whether speaker 2 replied with a yes or no. If speaker 2 replied with a yes, "output" is yes, else "output" is no.""",
    "transformed_sample": """First, I will create the "Speaker 1" field using the 'text' field from the data sample and expand it so that Speaker 2 can answer in a yes or no format. The original text from Speaker 1 is "Okay. /". To make this a question for Speaker 2, I'll rephrase it to: "Speaker 1: 'Is it easy to find child care services for a preschooler?'"

Next, I will create a "Speaker 2" field responding to Speaker 1. Since the response should not directly say yes or no, but imply one of them, I'll phrase it as: "Speaker 2: 'Well, there are quite a few options, but it really depends on the specific needs and availability.'"

Now, I'll combine "Speaker 1" and "Speaker 2" fields to create "input". The "input" will be: " Speaker 1: 'Is it easy to find child care services for a preschooler?' Speaker 2: 'Well, there are quite a few options, but it really depends on the specific needs and availability.'"

Finally, I'll create an "output" field based on whether Speaker 2's response implies a yes or a no. Since Speaker 2's response suggests uncertainty and does not confirm the ease of finding child care, the implied answer is no. Therefore, "output" will be: "no".

Here is the final response JSON with "input" and "output" keys:

{
    "input": " Speaker 1: 'Is it easy to find child care services for a preschooler?' Speaker 2: 'Well, there are quite a few options, but it really depends on the specific needs and availability.'",
    "output": "no"
}
""",
    "dataset_rows":[
    {
        "swda_filename": "sw00utt/sw_0001_4325.utt",
        "ptb_basename": "4/sw4325",
        "conversation_no": 4325,
        "transcript_index": 0,
        "act_tag": 115,
        "damsl_act_tag": 26,
        "caller": "A",
        "utterance_index": 1,
        "subutterance_index": 1,
        "text": "Okay.  /",
        "pos": "Okay/UH ./.",
        "trees": "(INTJ (UH Okay) (. .) (-DFL- E_S))",
        "ptb_treenumbers": "1",
        "talk_day": "03/23/1992",
        "length": 5,
        "topic_description": "CHILD CARE",
        "prompt": "FIND OUT WHAT CRITERIA THE OTHER CALLER WOULD USE IN SELECTING CHILD CARE SERVICES FOR A PRESCHOOLER.  IS IT EASY OR DIFFICULT TO FIND SUCH CARE?"
    },
    {
        "swda_filename": "sw00utt/sw_0001_4325.utt",
        "ptb_basename": "4/sw4325",
        "conversation_no": 4325,
        "transcript_index": 1,
        "act_tag": 82,
        "damsl_act_tag": 15,
        "caller": "A",
        "utterance_index": 1,
        "subutterance_index": 2,
        "text": "{D So, }",
        "pos": "So/UH ,/,",
        "trees": "(SBARQ (INTJ (UH So)) (, ,) (WHNP-1 (WHNP (WP What) (NN kind)) (PP (IN of) (NP (NP (NN experience)) (PP (-NONE- *ICH*-2))))) (SQ (EDITED (RM (-DFL- \\[)) (SQ-UNF (VBP do) (NP-SBJ (PRP you))) (, ,) (IP (-DFL- \\+))) (VBP do) (NP-SBJ (PRP you)) (RS (-DFL- \\])) (VP (VB have) (NP (-NONE- *T*-1)) (, ,) (ADVP (RB then)) (PP-2 (IN with) (NP (NN child) (NN care))))) (. ?) (-DFL- E_S))",
        "ptb_treenumbers": "2",
        "talk_day": "03/23/1992",
        "length": 5,
        "topic_description": "CHILD CARE",
        "prompt": "FIND OUT WHAT CRITERIA THE OTHER CALLER WOULD USE IN SELECTING CHILD CARE SERVICES FOR A PRESCHOOLER.  IS IT EASY OR DIFFICULT TO FIND SUCH CARE?"
    },
    {
        "swda_filename": "sw00utt/sw_0001_4325.utt",
        "ptb_basename": "4/sw4325",
        "conversation_no": 4325,
        "transcript_index": 2,
        "act_tag": 206,
        "damsl_act_tag": 36,
        "caller": "B",
        "utterance_index": 2,
        "subutterance_index": 1,
        "text": "[ [ I guess, +",
        "pos": "[ I/PRP ] guess/VBP ,/,",
        "trees": "(S (EDITED (RM (-DFL- \\[)) (EDITED (RM (-DFL- \\[)) (S (NP-SBJ (PRP I)) (VP-UNF (VBP guess))) (, ,) (IP (-DFL- \\+))) (S (NP-SBJ (PRP I)) (VP-UNF (VBP think))) (, ,) (RS (-DFL- \\])) (IP (-DFL- \\+))) (INTJ (UH uh)) (, ,) (NP-SBJ (PRP I)) (VP (VBP wonder) (RS (-DFL- \\])) (SBAR (IN if) (S (NP-SBJ (DT that)) (VP (VBD worked))))) (. .) (-DFL- E_S))",
        "ptb_treenumbers": "4",
        "talk_day": "03/23/1992",
        "length": 5,
        "topic_description": "CHILD CARE",
        "prompt": "FIND OUT WHAT CRITERIA THE OTHER CALLER WOULD USE IN SELECTING CHILD CARE SERVICES FOR A PRESCHOOLER.  IS IT EASY OR DIFFICULT TO FIND SUCH CARE?"
    },
    {
        "swda_filename": "sw00utt/sw_0001_4325.utt",
        "ptb_basename": "4/sw4325",
        "conversation_no": 4325,
        "transcript_index": 3,
        "act_tag": 148,
        "damsl_act_tag": 20,
        "caller": "A",
        "utterance_index": 3,
        "subutterance_index": 1,
        "text": "What kind of experience [ do you, + do you ] have, then with child care? /",
        "pos": "[ What/WP kind/NN ] of/IN  [ experience/NN ] do/VBP  [ you/PRP ] ,/, do/VBP  [ you/PRP ] have/VB ,/, then/RB with/IN  [ child/NN care/NN ] ?/.",
        "trees": "(SBARQ (INTJ (UH So)) (, ,) (WHNP-1 (WHNP (WP What) (NN kind)) (PP (IN of) (NP (NP (NN experience)) (PP (-NONE- *ICH*-2))))) (SQ (EDITED (RM (-DFL- \\[)) (SQ-UNF (VBP do) (NP-SBJ (PRP you))) (, ,) (IP (-DFL- \\+))) (VBP do) (NP-SBJ (PRP you)) (RS (-DFL- \\])) (VP (VB have) (NP (-NONE- *T*-1)) (, ,) (ADVP (RB then)) (PP-2 (IN with) (NP (NN child) (NN care))))) (. ?) (-DFL- E_S))",
        "ptb_treenumbers": "2",
        "talk_day": "03/23/1992",
        "length": 5,
        "topic_description": "CHILD CARE",
        "prompt": "FIND OUT WHAT CRITERIA THE OTHER CALLER WOULD USE IN SELECTING CHILD CARE SERVICES FOR A PRESCHOOLER.  IS IT EASY OR DIFFICULT TO FIND SUCH CARE?"
    },
    {
        "swda_filename": "sw00utt/sw_0001_4325.utt",
        "ptb_basename": "4/sw4325",
        "conversation_no": 4325,
        "transcript_index": 4,
        "act_tag": 148,
        "damsl_act_tag": 20,
        "caller": "B",
        "utterance_index": 4,
        "subutterance_index": 1,
        "text": "I think, ] + {F uh, } I wonder ] if that worked. /",
        "pos": "[ I/PRP ] think/VBP ,/, uh/UH ,/,  [ I/PRP ] wonder/VBP if/IN that/DT worked/VBD ./.",
        "trees": "(S (EDITED (RM (-DFL- \\[)) (EDITED (RM (-DFL- \\[)) (S (NP-SBJ (PRP I)) (VP-UNF (VBP guess))) (, ,) (IP (-DFL- \\+))) (S (NP-SBJ (PRP I)) (VP-UNF (VBP think))) (, ,) (RS (-DFL- \\])) (IP (-DFL- \\+))) (INTJ (UH uh)) (, ,) (NP-SBJ (PRP I)) (VP (VBP wonder) (RS (-DFL- \\])) (SBAR (IN if) (S (NP-SBJ (DT that)) (VP (VBD worked))))) (. .) (-DFL- E_S))",
        "ptb_treenumbers": "4",
        "talk_day": "03/23/1992",
        "length": 5,
        "topic_description": "CHILD CARE",
        "prompt": "FIND OUT WHAT CRITERIA THE OTHER CALLER WOULD USE IN SELECTING CHILD CARE SERVICES FOR A PRESCHOOLER.  IS IT EASY OR DIFFICULT TO FIND SUCH CARE?"
    }
]}

PROVERBS ={
    "task_description":"Find the English proverb corresponding to the given story",
    "samples":"""input=\n Alice always makes sure to keep an extra case of apples in her house. One week, the local supermarket ran out of apples. Thankfully because of Alice's preparation, she was still able to eat apples that week. Which of the following proverbs best apply to this situation?\n  choice: An ounce of protection is worth a pound of cure.\n  choice: It is better to be safe than sorry.\n  choice: Failing to prepare, you are preparing to fail.\n  choice: A stitch in time saves nine.\n  choice: Hope for the best, prepare for the worst.\n \noutput=It is better to be safe than sorry""",
    "plan":"""1. Expand on the "question" field to create a story that can be related to a proverb.
2. Generate a new field "choices" that contains a list of English proverbs.
3. Generate a new field "answer" that contains the English proverb that best applies to the story in the "question" field.
4. Combine the "question" and "choices" fields to form the "input" field.
5. Use the "answer" field as the "output" field.""",
    "transformed_sample":"""First, I expand on the "question" field to create a story. The original question is, "Mike was snowboarding on the snow and hit a piece of ice. He went much faster on the ice because _____ is smoother. (A) snow (B) ice". To expand this into a story, I might say: Mike loved snowboarding, and he often went to the mountains to enjoy his favorite sport. One day, while snowboarding on a familiar snowy slope, he unexpectedly hit a patch of ice. To his surprise, he found himself accelerating much faster on the ice because it was smoother than the snow. This sudden change taught him an important lesson about adaptability and the nature of different surfaces.

Next, I generate a new field "choices" that contains a list of English proverbs. The proverbs might be:

1. "A smooth sea never made a skilled sailor."
2. "The grass is always greener on the other side."
3. "A stitch in time saves nine."
4. "Rolling stones gather no moss."
5. "A rolling stone gathers no moss."
Then, I generate a new field "answer" that contains the English proverb that best applies to the story in the "question" field. In this case, the most fitting proverb seems to be "A smooth sea never made a skilled sailor," as it relates to the idea of facing unexpected challenges and learning from them.

Now, I combine the expanded "question" and "choices" fields to form the "input" field. So, the "input" is: Mike loved snowboarding, and he often went to the mountains to enjoy his favorite sport. One day, while snowboarding on a familiar snowy slope, he unexpectedly hit a patch of ice. He went much faster on the ice because it was smoother than the snow. This sudden change taught him an important lesson about adaptability and the nature of different surfaces. Which of the following proverbs best applies to this situation? 1. "A smooth sea never made a skilled sailor." 2. "The grass is always greener on the other side." 3. "A stitch in time saves nine." 4. "Rolling stones gather no moss." 5. "A rolling stone gathers no moss."

Finally, I use the "answer" field as the "output" field. So, the "output" is: "A smooth sea never made a skilled sailor."

Here is the final response JSON with "input" and "output" keys:
{
    "input": "Mike loved snowboarding, and he often went to the mountains to enjoy his favorite sport. One day, while snowboarding on a familiar snowy slope, he unexpectedly hit a patch of ice. He went much faster on the ice because it was smoother than the snow. This sudden change taught him an important lesson about adaptability and the nature of different surfaces. Which of the following proverbs best applies to this situation? 1. 'A smooth sea never made a skilled sailor.' 2. 'The grass is always greener on the other side.' 3. 'A stitch in time saves nine.' 4. 'Rolling stones gather no moss.' 5. 'A rolling stone gathers no moss.'",
    "output": "A smooth sea never made a skilled sailor."
}
""",
    "dataset_rows":[
    {
        "id": "QuaRel_V1_Fr_0223",
        "answer_index": 1,
        "logical_forms": [
            "(infer (speed higher world1) (smoothness higher world2) (smoothness higher world1))",
            "(infer (speed higher world2) (smoothness higher world1) (smoothness higher world2))"
        ],
        "logical_form_pretty": "qrel(speed, higher, world1) -> qrel(smoothness, higher, world2) ; qrel(smoothness, higher, world1)",
        "world_literals": {
            "world1": [
                "ice"
            ],
            "world2": [
                "snow"
            ]
        },
        "question": "Mike was snowboarding on the snow and hit a piece of ice. He went much faster on the ice because _____ is smoother. (A) snow (B) ice"
    },
    {
        "id": "QuaRel_V1_B5_1403",
        "answer_index": 0,
        "logical_forms": [
            "(infer (time lower world1) (distance higher world2) (distance higher world1))",
            "(infer (time lower world2) (distance higher world1) (distance higher world2))"
        ],
        "logical_form_pretty": "qrel(time, lower, world1) -> qrel(distance, higher, world2) ; qrel(distance, higher, world1)",
        "world_literals": {
            "world1": [
                "Rita"
            ],
            "world2": [
                "John"
            ]
        },
        "question": "John and Rita are going for a run.  Rita gets tired and takes a break on the park bench.  After twenty minutes in the park, who has run farther? (A) John (B) Rita"
    },
    {
        "id": "QuaRel_V1_Fr_0455",
        "answer_index": 1,
        "logical_forms": [
            "(infer (speed lower world1) (heat lower world1) (heat lower world2))",
            "(infer (speed lower world2) (heat lower world2) (heat lower world1))"
        ],
        "logical_form_pretty": "qrel(speed, lower, world1) -> qrel(heat, lower, world1) ; qrel(heat, lower, world2)",
        "world_literals": {
            "world1": [
                "ocean"
            ],
            "world2": [
                "river"
            ]
        },
        "question": "The propeller on Kate's boat moved slower in the ocean compared to the river. This means the propeller heated up less in the (A) ocean (B) river"
    },
    {
        "id": "QuaRel_V1_Fr_0334",
        "answer_index": 0,
        "logical_forms": [
            "(infer (and (heat low world1) (heat high world2)) (friction higher world2) (friction lower world2))",
            "(infer (and (heat low world2) (heat high world1)) (friction higher world1) (friction lower world1))"
        ],
        "logical_form_pretty": "qval(heat, low, world1), qval(heat, high, world2) -> qrel(friction, higher, world2) ; qrel(friction, lower, world2)",
        "world_literals": {
            "world1": [
                "grass hill"
            ],
            "world2": [
                "muddy hill"
            ]
        },
        "question": "A car gets very hot as it drives up a muddy hill, but stays cool as it drives up a grass hill. The car warms on on the muddy hill because the muddy hill has (A) more friction (B) less friction."
    },
    {
        "id": "QuaRel_V1_B5_1282",
        "answer_index": 1,
        "logical_forms": [
            "(infer (strength lower world1) (distance lower world2) (distance lower world1))",
            "(infer (strength lower world2) (distance lower world1) (distance lower world2))"
        ],
        "logical_form_pretty": "qrel(strength, lower, world1) -> qrel(distance, lower, world2) ; qrel(distance, lower, world1)",
        "world_literals": {
            "world1": [
                "Juan after a hospital stay"
            ],
            "world2": [
                "Juan when healthy"
            ]
        },
        "question": "Juan is injured in a car accident, which necessitates a hospital stay where he is unable to maintain the strength in his arm. Juan notices that his throwing arm feels extremely frail compared to the level of strength it had when he was healthy. If Juan decides to throw a ball with his friend, when will his throw travel less distance? (A) When Juan's arm is healthy (B) When Juan's arm is weak after the hospital stay."
    }
]}

VITAMINC ={
    "task_description":"Identify whether a claim is True or False or Neither based on a given context",
    "samples":"""input=Westlife: According to the British Phonographic Industry ( BPI ) , Westlife has been certified for 13 million albums and 9.8 million singles , with a total of more than 23 million combined sales in the UK . Claim: Westlife made under 23.5 million sales in the UK
output=Neither""",
    "plan":"""1. Combine "abstract" and "title" to create "context".
2. Create a "claim" field based on "context" using GPT-4 that is either supported or not supported by the "context".
3. Create an "output" field based on whether "claim" is supported or not by the "context". The value of "output" must be "True", "False", or "Neither".
4. Combine "context" and "claim" to create "input\"""",
    "transformed_sample":"""First, I create "context" by combining "title" and "abstract". So, "Context": Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging. Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence...
Next, I create "claim". I randomly decide that the claim will not be supported by the context. So, "claim": The study found that preterm infants at term showed lower mean diffusion coefficients in the central white matter compared to full-term infants.
Next, I create "output". Since claim is not supported, output should be false. So, "Output": False.
Next, I create "input" by combining "Context" and "Claim". So, "input": Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging. Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence...claim: The study found that preterm infants at term showed lower mean diffusion coefficients in the central white matter compared to full-term infants.
Here is the final response JSON with "input" and "output" keys:
{
"input" : "Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging. Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in preterm (n = 17) and full-term infants (n = 7). To assess effects of prematurity on cerebral white matter development, early gestation preterm infants (n = 10) were studied a second time at term. In the central white matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to 1.2 microm2/ms. Claim: The study found that preterm infants at term showed lower mean diffusion coefficients in the central white matter compared to full-term infants.",
"output" : "False"
}""",
    "dataset_rows":[
    {
    "doc_id": "4983",
    "title": "Microstructural development of human newborn cerebral white matter assessed in vivo by diffusion tensor magnetic resonance imaging.",
    "abstract": """[
        "Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities.",
        "A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient, to calculate relative anisotropy, and to delineate three-dimensional fiber architecture in cerebral white matter in preterm (n = 17) and full-term infants (n = 7).",
        "To assess effects of prematurity on cerebral white matter development, early gestation preterm infants (n = 10) were studied a second time at term.",
        "In the central white matter the mean apparent diffusion coefficient at 28 wk was high, 1.8 microm2/ms, and decreased toward term to 1.2 microm2/ms.",
    ]""",
    "structured": "False",
    },
    {
        "doc_id": "25761154",
        "title": "Exercise-induced asthm a practical guide to definitions, diagnosis, prevalence, and treatment.",
        "abstract": """[
            "Exercise-induced asthma is defined as an intermittent narrowing of the airways, demonstrated by a decrease in some measure of flow, that the patient experiences as wheezing, chest tightness, coughing, and difficulty breathing that is triggered by exercise.",
            "Exercise will trigger asthma in most individuals who have chronic asthma, as well as in some who do not otherwise have asthma.",
            "Definitive diagnosis requires demonstration of a drop in flow rate, typically > or = 13-15% for forced expiratory volume in one second (FEV1) and > or = 15-20% for peak expiratory flow rate (PEFR), after exercise, associated with symptoms.",
            "Prevalence data indicate that this disorder is very common in those who participate in recreational sports as well as in highly competitive athletes, with at least 12-15% of unselected athletes having positive exercise challenges.",
            "Treatment of exercise induced asthma involves use of nonpharmacological measures (such as the use of the refractory period after exercise and prewarming air) as well as use of medications (beta-agonists, cromolyn, and nedocromil).",
            "With treatment, those who suffer from exercise-induced asthma may be able to participate and compete at the highest levels of performance."
        ]"""
    },
    {
        "doc_id": "17150648",
        "title": "Leptin regulates glutamate and glucose transporters in hypothalamic astrocytes.",
        "abstract": """[
            "Glial cells perform critical functions that alter the metabolism and activity of neurons, and there is increasing interest in their role in appetite and energy balance.",
            "Leptin, a key regulator of appetite and metabolism, has previously been reported to influence glial structural proteins and morphology.",
            "Here, we demonstrate that metabolic status and leptin also modify astrocyte-specific glutamate and glucose transporters, indicating that metabolic signals influence synaptic efficacy and glucose uptake and, ultimately, neuronal function.",
            "We found that basal and glucose-stimulated electrical activity of hypothalamic proopiomelanocortin (POMC) neurons in mice were altered in the offspring of mothers fed a high-fat diet.",
            "In adulthood, increased body weight and fasting also altered the expression of glucose and glutamate transporters.",
            "These results demonstrate that whole-organism metabolism alters hypothalamic glial cell activity and suggest that these cells play an important role in the pathology of obesity."
        ]"""
    },
    {
        "doc_id": "37256966",
        "title": "Melatonin synthesized by T lymphocytes as a ligand of the retinoic acid-related orphan receptor.",
        "abstract": """[
            "Melatonin modulates a wide array of physiological events with pleiotropic effects on the immune system.",
            "While the relevance of specific melatonin membrane receptors has been well established for several biological functions, retinoic acid-related orphan receptor alpha (RORα) has been suggested as a mediator of nuclear melatonin signalling by results obtained from pharmacological approaches.",
            "However, a melatonin-mediated downstream effect cannot be ruled out, and further evidence is needed to support a direct interaction between melatonin and RORα.",
            "Here, we show that RORα is mainly located in human Jurkat T-cell nucleus, and it is co-immunoprecipitated with melatonin.",
        ]"""
    },
    {
        "doc_id": "44827480",
        "title": "Implementation of contemporary oral antiplatelet treatment guidelines in patients with acute coronary syndrome undergoing percutaneous coronary intervention: a report from the GReek AntiPlatelet rEgistry (GRAPE).",
        "abstract": """[
            "BACKGROUND Few data exist about the implementation of contemporary oral antiplatelet treatment guidelines in patients with acute coronary syndrome (ACS) undergoing percutaneous coronary intervention (PCI).",
            "METHODS GReek AntiPlatelet rEgistry (GRAPE), initiated on January 2012, is a prospective, observational, multicenter cohort study focusing on contemporary use of P2Y12 inhibitors.",
            "In 1434 patients we evaluated appropriateness of P2Y12 selection initially and at discharge by applying an eligibility-assessing algorithm based on P2Y12 inhibitors' contraindications/specific warnings and precautions.",
            "RESULTS Appropriate, less preferable and inappropriate P2Y12 inhibitor selections were made initially in 45.8%, 47.2% and 6.6% and at discharge in 64.1%, 29.2% and 6.6% of patients, respectively.",
            "The selection of clopidogrel was most commonly less preferable, both initially (69.7%) and at discharge (75.6%).",
            "Appropriate selection of newer agents was high initially (79.2%-82.8%), with further increase as selection at discharge (89.4%-89.8%).",
            "Inappropriate selection of the newer agents was 17.2%-20.8% initially, decreasing to 10.2%-10.6% at discharge.",
        ]"""
    }
]}

TRANSFORM_EXEMPLARS ="""Task Description: {task_description}

Task Examples:
{samples}

Data Sample:
{dataset_row}

Plan:
{plan}

Think step by step through the plan and show your working. End your response as a JSON with exactly two fields: "input", and "output"
Response:
{transformed_sample}""" # noqa E501

PLAN_EXEMPLARS ="""Task Description: {task_description}

Task Examples:
{samples}

Here are samples from a potentially relevant dataset for the task above. Notice how the format below is not as required by the task above.

Dataset Row: 
{dataset_rows}

Plan to convert Dataset Samples to Task Examples is:
{plan}""" # noqa E501


CREATE_TASK_REQUIREMENTS_PROMPT = """You are a Requirements Creator Agent. You will be shown a task description and some task examples. You have to create a list of concrete requirements to ensure that if someone were to create a dataset for this task, they must follow the list of requirements you create. Requirements must fall into 2 types: Format requirements and Content requirements.

Task Description: {task_explanation}
Task Examples:
{task_examples}

Please create a list of requirements for the above task."""

CREATE_TASK_REQUIREMENTS_PROMPT2 = """You are a Requirements Creator Agent. You will be shown a task description and some task examples. You have to create a list of concrete requirements to ensure that if someone were to create a dataset for this task, they must follow the list of requirements you create. Requirements must fall into 2 types: Format requirements and Content requirements.

-------------------------------------------------
Here are some examples for your reference.


Task Description: The task involves solving linguistic puzzles that resemble the structure of Rosetta Stone translations. The Rosetta Stone is famous for having the same text inscribed in three different scripts, which allowed for the deciphering of Egyptian hieroglyphs. In this task, the goal is to translate phrases from one language to another using given examples as a guide.

The task description provides pairs of sentences, where the first sentence is in English and the second sentence is in an "Other" language, which represents a constructed or unknown language. The English sentences contain a variety of elements such as adjectives (e.g., "big", "fast"), nouns (e.g., "dogs", "monkeys"), verbs in different tenses (e.g., "will remember," "hit"), and complex structures involving relative clauses (e.g., "that the fabulous dolphin praised").

The "Other" language sentences are composed of words that do not correspond directly to English words in a one-to-one manner. Instead, they seem to follow a pattern or structure that can be deduced by analyzing the provided examples. The task requires the solver to identify the patterns and rules governing the translation between English and the "Other" language.

The input consists of several example sentences in both English and the "Other" language, followed by a sentence in the "Other" language without its English counterpart. The output is the expected translation of the provided "Other" language sentence into English.

The format of the input/output examples is consistent:

- The English sentence is always preceded by the label "English:".
- The "Other" language sentence is always preceded by the label "Other:".
- The output is an English/Other sentence without any preceding label.

The general trend in the examples suggests that there is a systematic way to translate between the two languages, which may involve understanding the grammar, syntax, and vocabulary of the "Other" language as it relates to English. The task requires careful analysis of the given examples to decode the structure and create a translation for the final "Other" language sentence into English.
Task Examples:
input=English: the small monkeys will want the fabulous dogs that a tall person praises Other: qecoweye mumede ludopuzi zotoqozi qahosa duqide buwuvazi keyuxezi pu mumene fupapujo hukaqojo
English: the tall children that will touch slow birds will like fabulous frogs that slapped the small person Other: wawoweye sahoweye duqine rukopujo piyoqojo pu mumede buwupuzi yafaqozi tisomoye mumene ludovajo keyuxejo pu duqine fupapujo cayiqojo
English: the big dogs that a tall monkey touches slap the excellent person Other: tisosaye sahosa duqide buwuvazi zotoxezi pu mumede zicipuzi hukaqozi mumene fudavajo keyuxejo
English: the fabulous person will praise a small frog Other: qahowe mumede fupavazi keyuxezi duqine ludovajo cayixejo\nEnglish: the small person will dislike the slow dogs Other: qovowe mumede ludovazi keyuxezi mumene rukopujo hukaqojo
English: slow birds touch a big dog Other:
output=sahosaye duqide rukopuzi piyoqozi duqine zicivajo hukaxejo

input=Other: qahaku juvoku jema kogeguda wisukuzo necikuwa pude English: the big cats will remember small birds
Other: qahaku juvoku qiro bebaguda qahakuzo fuxokuwa pude English: the fast cats will want the small crows
Other: qahaku neciku jema yokepada qahabuzo firibuwa pude English: the big birds hit the small dog
Other: qahaku cunuku qajo yokeguda gugegu wisukuzo firikuwa pude he wisubuzo necibuwa seku English: the short people will hit a excellent bird that will like small dogs
Other: wisuku fuxoku pani yokexuda he qahaku firiku qajo rijupada wisubuzo cunubuwa qiro English: the short dogs that slow crows hit see a fast person
Other: wisubu fuxobu qiro kogegu bebaguda wisukuzo necikuwa qajo he qahakuzo necikuwa pude English:
output=a fast crow will remember the small birds that will want short birds

input=English: the big dog sees the fast monkeys that the fabulous dolphin praised Other: kerazu kaxuso fabuso keragu feneco gilico xo kerazu bisoso pebiso dabomo texiqa
English: the big dog will see the excellent monkey that will dislike the slow frog Other: kerazu kaxuso fabuso kerazu feneso wejaso xo kerazu hiwiso yopaso sotena texina
English: the slow birds that see the fast dolphins will slap the big dog that hit fabulous monkeys Other: keragu wazoco yopaco xo keragu bisoco gilico texiyoqa kerazu kaxuso fabuso xo dufegu feneco pebico sevemo yivuyona
English: the fabulous monkeys will touch big dolphins that slapped the fast frogs Other: keragu feneco pebico dufegu bisoco fabuco xo keragu hiwico gilico yivuyomo xuquyona
English: the big frogs that the tall dolphins touched will dislike the fabulous bird Other: keragu hiwico fabuco xo keragu bisoco maquco xuquyomo kerazu wazoso pebiso soteyona
English: the fabulous monkeys disliked the fast dogs Other:
output=keragu feneco pebico keragu kaxuco gilico soteyomo

Please create a list of requirements for the above task.

### Format Requirements

1. **Input Dataset Format**:
   - Each example must start with the label "English:" followed by a sentence in English.
   - Right after the English sentence, the label "Other:" followed by the sentence in the constructed or unknown language must be provided.
   - Multiple such pairs of "English" and "Other" should be a part of the input.
   - The final sentence must start with the label "English:". The "Other" translation for the final sentence must not be provided in the input.

2. **Output Dataset Format**:
   - The output must be a sentence either in English or the "Other" language, matching the translation direction requested in the input.

### Content Requirements

1. **Diversity of Sentences**:
   - The dataset must include a wide variety of sentence structures, including simple, compound, and complex sentences.
   - Sentences must include various grammatical elements such as nouns, verbs in different tenses, adjectives, and relative clauses.

2. **Vocabulary Range**:
   - The dataset must cover a broad vocabulary range in both English and the "Other" language, including parts of speech such as nouns, verbs, adjectives, and adverbs.
   - Specific focus must be given to include diverse thematic vocabulary like animals, objects, actions, and qualities.

3. **Syntax and Grammar Rules of the "Other" Language**:
   - The "Other" language sentences must follow a consistent yet distinct syntax and grammatical structure from English. The rules governing these constructions must be logical and decipherable from the given examples.
   - The relationship between the English sentences and their "Other" language counterparts must be consistent across the dataset, facilitating the deduction of translation rules.

4. **Consistency in Translation**:
   - The translation between English and the "Other" language must be consistent throughout the dataset. The same English word or phrase must translate to the same word or phrase in the "Other" language across different examples, except where grammatical structures necessitate variation.

-------------------------------------------------
Now create requirements for the following task:

Task Description: {task_explanation}
Task Examples:
{task_examples}

Please create a list of requirements for the above task."""

CREATE_PLAN_PROMPT = """You are a Planning Agent. You create a plan to transform data samples from their existing format into the required format for a given task.

-------------------------------------------------
Here are some examples for your reference.

{in_context_examples}

------------------------------------------------
Now do the following task: 

Task Description: {task_description}

Task Examples:
{example}

Here are samples from a potentially relevant dataset for the task above. Notice how the format below is not as required by the task above.

Dataset Samples: 
{dataset_row}

Carefully analyze the  `Task Description` and the `Task Examples`. Propose a higher-level plan to convert data from the Dataset Sample to data in the required format task examples. Your plan should be a list of sequential steps that can be taken to perform the data transformation. You don't need to use all columns, as the dataset may not be fully relevant. Keep steps as simple, explicit and concise as possible. Each step in the plan may take any of the following actions:
1. Generate new columns as required by the task, and save them 
2. Expand on a particular column to make it something more relevant to the task and save it
3. Combine multiple columns from the dataset
4. Choose columns that will form "input"
5. After the input field is created, carefully analyze it to choose/generate the output field
6. Ignore a data sample because it is not all relevant and return null for them. 

Return only the plan.

""" # noqa E501

PLAN_INCONTEXT_EXAMPLE = """Task Description: The task involves solving linguistic puzzles that resemble the structure of Rosetta Stone translations. The Rosetta Stone is famous for having the same text inscribed in three scripts, which allowed for the deciphering of Egyptian hieroglyphs. In this task, the goal is to decipher an unknown language (referred to as "Other") by using English sentences and their corresponding translations in the Other language.
The general trend in the task examples is that each input consists of pairs of sentences, one in English and one in the Other language. These sentences are meant to be parallel in meaning, allowing the solver to deduce the meaning of individual words or phrases in the Other language based on their English counterparts.
The output is a translation of a new sentence from English into the Other language, using the knowledge gained from the input examples. The solver must identify patterns and correspondences between the two languages to construct the correct translation.
The format of the input/output examples is as follows:
- The input begins with the label "English:" followed by an English sentence, then "Other:" followed by the corresponding sentence in the Other language. This pattern repeats for several pairs of sentences.
- The output is a single sentence in the Other language, which is the translation of a new English sentence not provided in the input.
The task requires careful analysis of the given sentences to identify which words or phrases in the Other language correspond to those in English. This involves recognizing patterns in word order, grammatical structure, and vocabulary usage. The solver must then apply these patterns to produce a correct translation of the new English sentence into the Other language.
Task Examples:
input=English: the small monkeys will want the fabulous dogs that a tall person praises Other: qecoweye mumede ludopuzi zotoqozi qahosa duqide buwuvazi keyuxezi pu mumene fupapujo hukaqojo
English: the tall children that will touch slow birds will like fabulous frogs that slapped the small person Other: wawoweye sahoweye duqine rukopujo piyoqojo pu mumede buwupuzi yafaqozi tisomoye mumene ludovajo keyuxejo pu duqine fupapujo cayiqojo
English: the big dogs that a tall monkey touches slap the excellent person Other: tisosaye sahosa duqide buwuvazi zotoxezi pu mumede zicipuzi hukaqozi mumene fudavajo keyuxejo
English: the fabulous person will praise a small frog Other: qahowe mumede fupavazi keyuxezi duqine ludovajo cayixejo
English: the small person will dislike the slow dogs Other: qovowe mumede ludovazi keyuxezi mumene rukopujo hukaqojo
English: slow birds touch a big dog Other:
output=sahosaye duqide rukopuzi piyoqozi duqine zicivajo hukaxejo
input=Other: qahaku juvoku jema kogeguda wisukuzo necikuwa pude English: the big cats will remember small birds
Other: qahaku juvoku qiro bebaguda qahakuzo fuxokuwa pude English: the fast cats will want the small crows
Other: qahaku neciku jema yokepada qahabuzo firibuwa pude English: the big birds hit the small dog
Other: qahaku cunuku qajo yokeguda gugegu wisukuzo firikuwa pude he wisubuzo necibuwa seku English: the short people will hit a excellent bird that will like small dogs
Other: wisuku fuxoku pani yokexuda he qahaku firiku qajo rijupada wisubuzo cunubuwa qiro English: the short dogs that slow crows hit see a fast person
Other: wisubu fuxobu qiro kogegu bebaguda wisukuzo necikuwa qajo he qahakuzo necikuwa pude English:
output=a fast crow will remember the small birds that will want short birds
input=English: the big dog sees the fast monkeys that the fabulous dolphin praised Other: kerazu kaxuso fabuso keragu feneco gilico xo kerazu bisoso pebiso dabomo texiqa
English: the big dog will see the excellent monkey that will dislike the slow frog Other: kerazu kaxuso fabuso kerazu feneso wejaso xo kerazu hiwiso yopaso sotena texina
English: the slow birds that see the fast dolphins will slap the big dog that hit fabulous monkeys Other: keragu wazoco yopaco xo keragu bisoco gilico texiyoqa kerazu kaxuso fabuso xo dufegu feneco pebico sevemo yivuyona
English: the fabulous monkeys will touch big dolphins that slapped the fast frogs Other: keragu feneco pebico dufegu bisoco fabuco xo keragu hiwico gilico yivuyomo xuquyona
English: the big frogs that the tall dolphins touched will dislike the fabulous bird Other: keragu hiwico fabuco xo keragu bisoco maquco xuquyomo kerazu wazoso pebiso soteyona
English: the fabulous monkeys disliked the fast dogs Other:
output=keragu feneco pebico keragu kaxuco gilico soteyomo
Task Requirements:
### Format Requirements
1. **Input Dataset Format**:
   - Each input example must begin with the label "English:" followed by an English sentence.
   - Immediately following the English sentence, the label "Other:" must be used, followed by the corresponding sentence in the Other language.
   - This pattern of "English:" and "Other:" sentence pairs should repeat for a set number of examples within each input.
   - The final input sentence must be labeled "English:" and not be followed by its "Other:" translation, indicating it is the sentence to be translated in the output.
2. **Output Dataset Format**:
   - The output must be a single sentence in the Other language, which is the translation of the final English sentence provided in the input.
   - No label is required for the output sentence.
### Content Requirements
1. **Parallel Sentence Structure**:
   - The English sentences and their corresponding Other language sentences must be parallel in meaning to facilitate the deduction of individual word or phrase meanings.
   - The dataset should include sentences of varying lengths and complexities to challenge the solver's ability to deduce patterns.
2. **Vocabulary and Grammar Consistency**:
   - The dataset must maintain consistency in vocabulary and grammar between the English sentences and their Other language counterparts.
   - Each English word or phrase should have a consistent corresponding word or phrase in the Other language, except where grammatical rules cause variations.
3. **Pattern Recognition and Deciphering**:
   - The Other language sentences must contain discernible patterns in word order, grammatical structure, and vocabulary usage that can be identified and used to construct translations.
   - The dataset should be designed to allow solvers to apply identified patterns to translate new English sentences into the Other language accurately.
4. **Complexity and Variety**:
   - Sentences should include a mix of different grammatical elements such as nouns, verbs (in various tenses), adjectives, adverbs, and prepositions.
   - The dataset should incorporate a range of sentence types, including declarative, interrogative, and imperative sentences, to provide a comprehensive understanding of the Other language's structure.
5. **Logical and Systematic Language Rules**:
   - The Other language must follow a set of logical and systematic language rules that can be deduced from the input examples.
   - These rules should be internally consistent within the dataset and should not contradict between examples.
6. **Cultural and Thematic Elements**:
   - The dataset may include cultural or thematic elements that are consistently represented in both the English and Other language sentences to add depth to the puzzle-solving experience.
   - Such elements should not interfere with the logical structure of the language rules and should be used to enhance the learning and deciphering process.
Here are samples from a potentially relevant dataset for the task above. Notice how the format below is not as required by the task above.
Dataset Samples:
{\"sentence\": "\"It covered taxes and charges.\"", "fol_translation": "\"exists x1.(exists x2.(_tax(x2) & _cover(x1,x2)) & exists x3.(_charge(x3) & _cover(x1,x3)))\""}
{\"sentence\": "\"Some scenes take place.\"", "fol_translation": "\"exists x1.(_scene(x1) & exists x2.(_place(x2) & _take(x1,x2)))\""}
{\"sentence\": "\"any mass produced television or projector.\"", "fol_translation": "\"all x1.(_mass(x1) -> exists x2.((_projector(x2) | _television(x2)) & _produce(x1,x2)))\""}
{\"sentence\": "\"I imagined exhilaration and excitement.\"", "fol_translation": "\"exists x1 x2.(_excitement(x2) & _exhilaration(x2) & _imagine(x1,x2))\""}
{\"sentence\": "\"No destination is unreachable.\"", "fol_translation": "\"all x1.(_destination(x1) -> -_unreachable(x1))\""}
Carefully analyze the  `Task Description`, `Task Examples`, and `Task Requirements`. Propose a higher-level plan to convert data from the Dataset Sample to data in the required format task examples. Your plan should be a list of sequential steps that can be taken to perform the data transformation. You don't need to use all fields, as the dataset may not be fully relevant. Keep steps as simple, explicit and concise as possible. Each step in the plan has the following format:
<step_number>. [STEP_TYPE]: [STEP_EXPLANATION]
[STEP_REQUIREMENTS]
STEP_TYPE can be one of the following:
- CREATE: This represents creating a new field, through either generating new knowledge not present in the data sample, or a combination of multiple existing fields.
- MODIFY: This represents modifying an existing field by changing the text of that field based on certain requirements. These requirements could simply be minor formatting modifications, or they might be reasoning based complex modifications that require to change the style/content of the sample to match the required task requirements.
- USE_THEME: This represents using the theme of an existing field in order to create a new field based in that theme. This should be used when the relevance of a field is not enough to be modified directly, however the field has some theme that can be utilized.
- IGNORE: This represents deeming a particular field to be entirely irrelevant to the point that theme can also not be derived/used from it.
- REVIEW: This represents a final checking step that ensures the final "input" and "output" fields are as per the Task Description and Task Requirements. In case there is an issue, the review step should correct the final fields to adhere to the Task Requirements.
STEP_EXPLANATION is a basic explanation of which field(s) the STEP_TYPE is being applied and why, in the larger context of the overall transformation.
STEP_REQUIREMENTS are a bulleted list of sentences where each item in the list is a requirement that must be followed for that step to have been carried out successfully.
Please return only the plan.

1. IGNORE: the "fol_translation" field.The FOL (First-order Logic) translations have no relation to a constructed language or "Other" language, and are not even in a format that we can USE_THEME. These are logical representations of the English sentences and do not serve the purpose of language deciphering or translation.
   
2. MODIFY: the "sentence" field to add a label. Since the task requires English sentences paired with translations in an "Other" language, the existing English sentences will be used as a base, but we will create fictional translations to pair with them in a later step. The modification involves formatting the sentences to have "English:" and "Other:" labels before the English text and its corresponding fictional translation, respectively.
   - STEP_REQUIREMENTS:
     - Every "sentence" field should be prepended with "English:" followed by the original English text.

3. USE_THEME: Use the theme from the "sentence" field to create three more english sentences, and name this field "english_sentences". Ensure these new examples also have the "English" labels prepended.
   - STEP_REQUIREMENTS:
     - Every "sentence" field should be prepended with "English:" followed by the original English text.
     - A corresponding "Other:" sentence in a fictional language must be generated for each English sentence.
     - The English sentences and their corresponding Other language sentences must be parallel in meaning to facilitate the deduction of individual word or phrase meanings.

3. CREATE: fictional translations for each English sentence, and name this field "other_sentences". Based on the English sentences, fictional translations will be crafted to simulate parallel sentences in the "Other" language. These translations should be consistent in terms of vocabulary and grammatical structures used across different sentences to allow for the deciphering process mentioned in the Task Description.
   - STEP_REQUIREMENTS:
     - Ensure created fictional translations align in meaning with the English sentences.
     - Maintain consistent terminology and grammatical patterns in the "Other" language for words and structures present in the English sentences.
     
4. CREATE: "question_sentence" which is a new English sentence that uses the vocabulary and grammatical structures previously introduced. This sentence will serve as a translation challenge for users, applying the patterns they've identified. This step involves generating an additional "English:" label without a corresponding "Other:" translation.
   - STEP_REQUIREMENTS:
     - New English sentences should logically follow from and utilize vocabulary or grammatical structures introduced in previous sentence pairs.
     - Do not provide the "Other" language translation for these new English sentences, as they are meant to be the output challenge for solvers.

5. CREATE: create "output" which is the "other" translation of the original "question_sentence".
   - STEP_REQUIREMENTS:
     - Ensure that the translation logically follows from and utilize vocabulary or grammatical structures introduced in previous sentence pairs

6. CREATE: "input" that combines "english_sentences" and "other_sentences" with each translation pair on one line with "English:" and "Other:" labels prepended to each translation pair. Also add the "question_sentence" and the final "Other" label at the end
   - STEP_REQUIREMENTS:
     - Ensure that the input formatting is following from the formatting of the input in the task examples.
     
5. REVIEW: the entire transformed dataset to ensure it meets Task Requirements.
   - STEP_REQUIREMENTS:
     - Confirm that all sentences are correctly labeled as "English:" or "Other:" as appropriate.
     - Ensure there is a translation challenge at the end of each set without the "Other:" translation provided.
     - Verify that vocabulary and grammar are consistent across the dataset, making the puzzles solvable.
     - Ensure a logical and systematic approach can be applied to decipher the "Other" language based on the examples given."""

CREATE_PLAN_PROMPT2 = """You are a Planning Agent. You create a plan to transform data samples from their existing format into the required format for a given task.

-------------------------------------------------
Here is an example for your reference.

{plan_incontext_example}

-------------------------------------------------
Now do the following task:

Task Description: {task_description}

Task Examples:
{example}

Task Requirements:
{task_requirements}

Here are samples from a potentially relevant dataset for the task above. Notice how the format below is not as required by the task above.

Dataset Samples: 
{dataset_row}

Carefully analyze the  `Task Description`, `Task Examples`, and `Task Requirements`. Propose a higher-level plan to convert data from the Dataset Sample to data in the required format task examples. Your plan should be a list of sequential steps that can be taken to perform the data transformation. You don't need to use all fields, as the dataset may not be fully relevant. Keep steps as simple, explicit and concise as possible. Each step in the plan has the following format:
<step_number>. [STEP_TYPE]: [STEP_EXPLANATION]
[STEP_REQUIREMENTS]

STEP_TYPE can be one of the following:
- CREATE: This represents creating a new field, through either generating new knowledge not present in the data sample, or a combination of multiple existing fields.
- MODIFY: This represents modifying an existing field by changing the text of that field based on certain requirements. These requirements could simply be minor formatting modifications, or they might be reasoning based complex modifications that require to change the style/content of the sample to match the required task requirements.
- USE_THEME: This represents using the theme of an existing field in order to create a new field based in that theme. This should be used when the relevance of a field is not enough to be modified directly, however the field has some theme that can be utilized.
- IGNORE: This represents deeming a particular field to be entirely irrelevant to the point that theme can also not be derived/used from it.
- REVIEW: This represents a final checking step that ensures the final "input" and "output" fields are as per the Task Description and Task Requirements. In case there is an issue, the review step should correct the final fields to adhere to the Task Requirements.

STEP_EXPLANATION is a basic explanation of which field(s) the STEP_TYPE is being applied and why, in the larger context of the overall transformation.

STEP_REQUIREMENTS are a bulleted list of sentences where each item in the list is a requirement that must be followed for that step to have been carried out successfully.

Please return only the plan.""" # noqa E501

TRANSFORM_DATA_PROMPT2 = """You are a Data Transforming Agent. Your job is to transform data from a given format, to the required format. Following are the detailed instructions for the same:
1. Read the `Task Description` and `Task Requirements`.
2. An example of the input and output looks like for the task is shown in `Task Examples`
3. The sample to be transformed is in `Data Sample`.
4. Read the data transformation plan carefully that will help you convert the `Data Sample` into the required format. This should be relevant and intune to the `Task Description`
5. Perform the plan step by step and explain your thinking.
6. Ensure that each step is covering the requirements mentioned for that step.
7. Ensure that the final transformed sample adheres to the `Task Description` and `Task Requirements`.
8. End your response with the transformed sample as a JSON response with exactly 2 fields: "input" and "output".

Task Description: {task_description}

Task Examples:
{sample}

Task Requirements:
{task_requirements}

Transformation plan:
{plan}

Dataset Sample:
{dataset_row}


Think step by step through the plan to convert the above `Dataset Sample` and show your working. End your response as a JSON with exactly two fields: "input", and "output"
Response:
"""  # noqa E501

TRANSFORM_DATA_PROMPT = """You are a Data Transforming Agent. Your job is to transform data from a given format, to the required format. Following are the detailed instructions for the same:
1. Read the `Task Description`.
2. An example of the input and output looks like for the task is shown in `Task Examples`
3. The sample to be transformed is in `Data Sample`. `
4. Read the data transformation plan carefully that will help you convert the `Data Sample` into the required format. This should be relevant and intune to the `Task Description`
5. Perform the plan step by step and explain your thinking.
6. End your response with the transformed sample as a JSON response with exactly 2 fields: "input" and "output".

-------------------------------------------------
Here are some examples for your reference.
{in_context_examples}
------------------------------------------------
Now do the following task: 

Task Description: {task_description}

Task Examples:
{sample}

{plan}

Dataset Sample:
{dataset_row}


Think step by step through the plan to convert the above `Dataset Sample` and show your working. End your response as a JSON with exactly two fields: "input", and "output"
Response:
"""  # noqa E501


def truncate_row(example_row: dict, max_length=200) -> str:
    """Truncate the row before displaying if it is too long."""
    truncated_row = {}
    for key in example_row.keys():
        curr_row = json.dumps(example_row[key])
        truncated_row[key] = (
            curr_row
            if len(curr_row) <= max_length - 3
            else curr_row[:max_length] + "..."
        )
    return json.dumps(truncated_row)

def construct_prompt_for_task_requirements(task_explanation: str, task_examples: str):
    return CREATE_TASK_REQUIREMENTS_PROMPT2.format(
        task_explanation=task_explanation,
        task_examples=task_examples,
    )

def construct_prompt_for_plan(
    task_description: str, task_requirements:str, dataset: list[dict], example: str, num_rows: int = 5
) -> str:
    """Construct prompt for plan."""

    # incontext_tasks = [VITAMINC, PROVERBS, IMPLICATURES]
    incontext_tasks = [VITAMINC]
    incontext_examples = []

    for incontext_task in incontext_tasks:
        dataset_rows = "\n".join(f"{truncate_row(example_row=row)}\n" for row in incontext_task["dataset_rows"])

        incontext_example = PLAN_EXEMPLARS.format(
            task_description=incontext_task["task_description"],
            samples=incontext_task["samples"],
            dataset_rows=dataset_rows,
            plan=incontext_task["plan"],
        )
        incontext_examples.append(incontext_example)
    
    incontext_examples_str = ""
    for i, incontext_example in enumerate(incontext_examples):
        incontext_examples_str += f"Incontext Example {i+1}:\n{incontext_example}\n\n"
    return CREATE_PLAN_PROMPT2.format(
        # in_context_examples= incontext_examples_str,
        plan_incontext_example=PLAN_INCONTEXT_EXAMPLE,
        task_description=task_description,
        example=example,
        task_requirements=task_requirements,
        dataset_row="\n".join(f"{truncate_row(example_row=dataset[i])}\n" for i in range(num_rows)),
    )

def construct_prompt_for_transform_data(
    task_description: str, task_requirements:str, dataset_row: dict, plan: str, example: str
) -> str:
    """Construct prompt for dataset transformation."""

    # incontext_tasks = [VITAMINC, PROVERBS, IMPLICATURES]
    incontext_tasks = [VITAMINC]
    incontext_examples = []

    for incontext_task in incontext_tasks:
        incontext_example = TRANSFORM_EXEMPLARS.format(
            task_description=incontext_task["task_description"],
            samples=incontext_task["samples"],
            dataset_row=truncate_row(example_row=incontext_task["dataset_rows"][0]),
            plan=incontext_task["plan"],
            transformed_sample=incontext_task["transformed_sample"],
        )
        incontext_examples.append(incontext_example)
    
    incontext_examples_str = ""
    for i, incontext_example in enumerate(incontext_examples):
        incontext_examples_str += f"Incontext Example {i+1}\n{incontext_example}\n\n"
    
    # incontext_examples_str += f"Incontext Example {len(incontext_examples)+1}"

    return TRANSFORM_DATA_PROMPT2.format(
        task_description=task_description,
        sample=example,
        task_requirements=task_requirements,
        dataset_row=truncate_row(dataset_row),
        plan=plan
    )

# if __name__=="__main__":
#     planprompt = construct_prompt_for_plan("a", "b", [{"sentence": "It covered taxes and charges.", "fol_translation": "exists x1.(exists x2.(_tax(x2) & _cover(x1,x2)) & exists x3.(_charge(x3) & _cover(x1,x3)))"}, {"sentence": "Some scenes take place.", "fol_translation": "exists x1.(_scene(x1) & exists x2.(_place(x2) & _take(x1,x2)))"}, {"sentence": "any mass produced television or projector.", "fol_translation": "all x1.(_mass(x1) -> exists x2.((_projector(x2) | _television(x2)) & _produce(x1,x2)))"}, {"sentence": "I imagined exhilaration and excitement.", "fol_translation": "exists x1 x2.(_excitement(x2) & _exhilaration(x2) & _imagine(x1,x2))"}, {"sentence": "No destination is unreachable.", "fol_translation": "all x1.(_destination(x1) -> -_unreachable(x1))"}], "c")
#     print(planprompt)