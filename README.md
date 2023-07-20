# prompt2model

## What is this?

This is a package that takes a prompt, like one that you might send to an
OpenAI model such as GPT-3 or GPT-4, and uses it to create a model that you
can use to solve the task in the prompt.

## Installation

You can install by running pip locally.

```bash
pip install .
```

There is more information for developers in the [CONTRIBUTING.md](CONTRIBUTING.md)
file.

To enable the model retriever, we need to untar the model_info.tgz file:
```
cd huggingface_models
tar -xvf model_info.tgz
```