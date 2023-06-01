# prompt2model

## What is this?

This package takes a prompt, like one that you might send to an OpenAI model such as GPT-3 or GPT-4, and uses it to create a model that you can use to solve the task in the prompt.

## Installation

It’s very easy to install the dependence. You can install it by running pip locally.

```bash
pip install .
```

## Configuration

Here are something related that needs to be configured.

- OPENAI_KEY: We are using the `gpt-3.5-turbo` model of OpenAI. So you need to provide your OpenAI key in the init function of `DatasetGenerator`, `OpenAIInstructionParser`. Alternatively, you can set the environment variable with `export OPENAI_API_KEY=<your key>`.


## Contribution

There is more information for developers in the [CONTRIBUTING.md](CONTRIBUTING.md) file.
