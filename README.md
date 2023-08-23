# prompt2model

## What is this?

`Prompt2model` is a powerful package designed
to streamline the process of creating models
based on prompts indicating a task for OpenAI
models such as ChatGPT
or GPT-4. This package takes a prompt
as input and leverages it to generate a model
that can effectively solve the task described
in the prompt. With `prompt2model`, you can
easily convert your prompts into functional
models, making it easier to utilize the
power of general OpenAI LLMs in your projects.

## Installation

To install the necessary dependencies,
run the following command in your terminal
to install the package using `pip`:

```bash
pip install .
```

## Configuration

Before using `prompt2model`, there is a
few configuration steps you need to complete:

- Sign up on the OpenAI website and obtain an
OpenAI API key.

- Provide OpenAI API key in the
initialization function of the classes which
requires calling OpenAI Models.

- Alternatively, you can set
the environment variable
`OPENAI_API_KEY` to your API key by running
the following command in your terminal:

```bash
export OPENAI_API_KEY=<your key>
```

- After setting the environment
Variable `OPENAI_API_KEY`, just
reference  load it in your Python:

```python
import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
```

To enable the model retriever, we need to untar the model_info.tgz file:

```bash
cd huggingface_models
tar -xvf model_info.tgz
```

## Components

The `prompt2model` package is composed
of several components, each designed
to fulfill a specific purpose. To gain
a comprehensive understanding of how to
utilize each component effectively,
please consult the `readme.md` file
situated in the directory of the respective
component. These files can be found at
`./prompt2model/<component>/readme.md`.
They provide detailed information and
instructions on maximizing the
functionality and benefits of each
component within the package.

## Usage

The `prompt2model` pipeline is a versatile
pipeline for task solving using a language
model. It covers stages including dataset retrieval,
generation, processing, model retrieval,
training, execution, evaluation, and
interface creation. The
`.cli_demo.py`
By directly run `python cli_demo.py`,
users can efficiently
leverage language models for various tasks
by customizing the components according to
their specific requirements.

## How to Write a Good Prompt

A good prompt can make the generated dataset
follow exactly the format of demonstrations.
It contains the instruction and few-shot examples.

The instruction should contain the following:

1. The exact format description for the input
and output, i.e., a string, a dictionary, or whatever.
2. The exact contents of each part of the
input and their relationship as possible as you can.
3. The range of possible input. For example,
"And the question can range from Math, Cultural,
Social, Geometry, Biology, History, Sports, Technology,
Science, and so on."

The few-shot examples should contain the following:

1. Use `=` rather than other ambiguous symbols like `:`.
2. Avoid unnecessary line breaks at the beginning.
For example, `input=""` is better than breaking
the line after `=`.
3. Use `input` rather than `Input`, `ouput` is
preferable likewise.
4. Warp the `input` and `output` into a string with `“”`.

Though the examples are optional, we strongly
suggest including them to guide the format and
content for the generator.

Also, we recommend providing several precise examples
in the specified format and inquiring with ChatGPT
about the format and scope of your examples.

## Customization

If you want to customize a specific component,
see the relevant doc page and class document string.

## Contribution

If you're interested in contributing
to the `prompt2model` project, please
refer to the [CONTRIBUTING.md](CONTRIBUTING.md)
file for detailed guidelines and
information tailored specifically
