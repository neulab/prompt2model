# prompt2model

## What is this?

`Prompt2model` is a powerful package designed
to streamline the process of creating models
based on prompts for OpenAI models such as
GPT-3 or GPT-4. This package takes a prompt
as input and leverages it to generate a model
that can effectively solve the task described
in the prompt. With `prompt2model`, you can
easily convert your prompts into functional
models, making it easier to utilize the
power of OpenAI models in your projects.

## Installation

To install the necessary dependencies,
run the following command in your terminal
to install the package using `pip`:

```bash
pip install .
```

## Configuration

Before using `prompt2model`, there are a
few configuration steps you need to complete:

1. Obtain an OpenAI API key for the
`gpt-3.5-turbo` model. You can sign
up on the OpenAI website to get your key.

2. Once you have obtained your API
key, you need to provide it in the
 initialization function of the
`DatasetGenerator` and `OpenAIInstructionParser`
 classes. Alternatively, you can set
 the environment variable
`OPENAI_API_KEY` to your API key by running
 the following command in your terminal:

```bash
export OPENAI_API_KEY=<your key>
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

## Usage[TODO]

The `prompt2model` pipeline is a versatile
pipeline for task solving using a language
model. It covers stages like dataset retrieval,
generation, processing, model retrieval,
training, execution, evaluation, and
interface creation. The
`./prompt2model/run_skeleton.py`
script executes the pipeline by
preprocessing the input prompt,
retrieving and processing datasets,
retrieving and training models,
selecting the best model, executing it,
evaluating the results, and generating
a Gradio interface. The pipeline utilizes
mock components for demonstration, but
in real-world scenarios, actual implementations
should replace them. By following
this pipeline, users can efficiently
leverage language models for various tasks
by customizing the components according to
their specific requirements.

## Contribution

If you're interested in contributing
to the `prompt2model` project, please
refer to the [CONTRIBUTING.md](CONTRIBUTING.md)
file for detailed guidelines and
information tailored specifically
for developers.
