"""Tools for generating hypothetical documents from prompts."""

from __future__ import annotations  # noqa FI58

import logging

from prompt2model.prompt_parser import PromptSpec
from prompt2model.utils import API_ERRORS, api_tools, handle_api_error

PROMPT_PREFIX = """HuggingFace contains models, which are each given a user-generated description. The first section of the description, delimited with two "---" lines, consists of a YAML description of the model. This may contain fields like "language" (supported by model), "datasets" (used to train the model), "tags" (e.g. tasks relevant to the model), and "metrics" (used to evaluate the model). Create a hypothetical HuggingFace model description that would satisfy a given user instruction. Here are some examples:

Instruction: "Give me some translation from English to Vietnamese. Input English and output Vietnamese."
Hypothetical model description:
---
language:
- en
- vi

tags:
- translation

license: apache-2.0
---

### eng-vie

* source group: English
* target group: Vietnamese
*  OPUS readme: [eng-vie](https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/eng-vie/README.md)

*  model: transformer-align
* source language(s): eng
* target language(s): vie vie_Hani
* model: transformer-align
* pre-processing: normalization + SentencePiece (spm32k,spm32k)
* a sentence initial language token is required in the form of `>>id<<` (id = valid target language ID)
* download original weights: [opus-2020-06-17.zip](https://object.pouta.csc.fi/Tatoeba-MT-models/eng-vie/opus-2020-06-17.zip)
* test set translations: [opus-2020-06-17.test.txt](https://object.pouta.csc.fi/Tatoeba-MT-models/eng-vie/opus-2020-06-17.test.txt)
* test set scores: [opus-2020-06-17.eval.txt](https://object.pouta.csc.fi/Tatoeba-MT-models/eng-vie/opus-2020-06-17.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| Tatoeba-test.eng.vie 	| 37.2 	| 0.542 |


### System Info:
- hf_name: eng-vie

- source_languages: eng

- target_languages: vie

- opus_readme_url: https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/eng-vie/README.md

- original_repo: Tatoeba-Challenge

- tags: ['translation']

- languages: ['en', 'vi']

- src_constituents: {'eng'}

- tgt_constituents: {'vie', 'vie_Hani'}

- src_multilingual: False

- tgt_multilingual: False

- prepro:  normalization + SentencePiece (spm32k,spm32k)

- src_alpha3: eng

- tgt_alpha3: vie

- short_pair: en-vi

- chrF2_score: 0.542

- bleu: 37.2

- brevity_penalty: 0.973

- ref_len: 24427.0

- src_name: English

- tgt_name: Vietnamese

- train_date: 2020-06-17

- src_alpha2: en

- tgt_alpha2: vi

- prefer_old: False

- long_pair: eng-vie


Instruction: "I want to summarize things like news articles."
Hypothetical model description:
---
language: en
license: apache-2.0
tags:
- pegasus
- seq2seq
- summarization
model-index:
- name: tuner007/pegasus_summarizer
  results:
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: cnn_dailymail
      type: cnn_dailymail
      config: 3.0.0
      split: train
    metrics:
    - name: ROUGE-1
      type: rouge
      value: 36.604
      verified: true
    - name: ROUGE-2
      type: rouge
      value: 14.6398
      verified: true
    - name: ROUGE-L
      type: rouge
      value: 23.8845
      verified: true
---

## Model description
[PEGASUS](https://github.com/google-research/pegasus) fine-tuned for summarization

> Created by [Arpit Rajauria](https://twitter.com/arpit_rajauria)
[![Twitter icon](https://cdn0.iconfinder.com/data/icons/shift-logotypes/32/Twitter-32.png)](https://twitter.com/arpit_rajauria)


### Framework versions

- Transformers 4.31.0
- Pytorch 2.0.1+cu118
- Datasets 2.13.1
- Tokenizers 0.13.3


Instruction: "I want to classify sentences by their sentiment (positive/negative/neutral)."
Hypothetical model description:
---
language: en
license: apache-2.0
datasets:
- sst2
- glue
model-index:
- name: distilbert-base-uncased-finetuned-sst-2-english
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: glue
      type: glue
      config: sst2
      split: validation
    metrics:
    - type: accuracy
      value: 0.9105504587155964
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiN2YyOGMxYjY2Y2JhMjkxNjIzN2FmMjNiNmM2ZWViNGY3MTNmNWI2YzhiYjYxZTY0ZGUyN2M1NGIxZjRiMjQwZiIsInZlcnNpb24iOjF9.uui0srxV5ZHRhxbYN6082EZdwpnBgubPJ5R2-Wk8HTWqmxYE3QHidevR9LLAhidqGw6Ih93fK0goAXncld_gBg
---

# DistilBERT base uncased finetuned SST-2

## Table of Contents
- [Model Details](#model-details)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)

## Model Details
**Model Description:** This model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned on SST-2.
This model reaches an accuracy of 91.3 on the dev set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).
- **Developed by:** Hugging Face
- **Model Type:** Text Classification
- **Language(s):** English
- **License:** Apache-2.0
- **Parent Model:** For more details about DistilBERT, we encourage users to check out [this model card](https://huggingface.co/distilbert-base-uncased).
- **Resources for more information:**
    - [Model Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification)
    - [DistilBERT paper](https://arxiv.org/abs/1910.01108)

## Uses

#### Direct Use

This model can be used for  topic classification. You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task. See the model hub to look for fine-tuned versions on a task that interests you.

#### Misuse and Out-of-scope Use
The model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.


## Risks, Limitations and Biases

Based on a few experimentations, we observed that this model could produce biased predictions that target underrepresented populations.

For instance, for sentences like `This film was filmed in COUNTRY`, this binary classification model will give radically different probabilities for the positive label depending on the country (0.89 if the country is France, but 0.08 if the country is Afghanistan) when nothing in the input indicates such a strong semantic shift.

# Training


#### Training Data


The authors use the following Stanford Sentiment Treebank([sst2](https://huggingface.co/datasets/sst2)) corpora for the model.
```
:"""  # noqa: E501


def generate_hypothetical_model_description(
    prompt: PromptSpec, max_api_calls: int = None
) -> str:
    """Generate a hypothetical model description for the user's instruction.

    This method is based on HyDE by Gao et al 2022 (https://arxiv.org/abs/2212.10496).

    Args:
        prompt: PromptSpec object containing the user's instruction.

    Returns:
        a hypothetical model description for the user's instruction.
    """
    if max_api_calls and max_api_calls <= 0:
        raise ValueError("max_api_calls must be > 0.")
    api_call_counter = 0

    instruction = prompt.instruction
    api_agent = api_tools.default_api_agent
    chatgpt_prompt = (
        PROMPT_PREFIX
        + "\n"
        + f'Instruction: "{instruction}"\nHypothetical model description:\n'
    )
    while True:
        try:
            chatgpt_completion = api_agent.generate_one_completion(
                chatgpt_prompt,
                temperature=0.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
            return chatgpt_completion.choices[0]["message"]["content"]
        except API_ERRORS as e:
            handle_api_error(e)
            api_call_counter += 1
            if max_api_calls and api_call_counter >= max_api_calls:
                logging.error("Maximum number of API calls reached.")
                raise ValueError("Maximum number of API calls reached.") from e
