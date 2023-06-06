# Gradio Interface Creation

This module provides a function to automatically create a Gradio interface for
interacting with a model.

## Usage

The `create_gradio` function takes a `GenerationModelExecutor` and an
`OpenAIInstructionParser` as an argument and returns a Gradio interface.

```python
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import OpenAIInstructionParser
from prompt2model.gradio_interface import create_gradio
model_executor = GenerationModelExecutor(...)
prompt_parser = OpenAIInstructionParser(...)
interface = create_gradio(model_executor, prompt_parser)
```

### Parameters

- `model_executor`: An instance of `GenerationModelExecutor` to expose via a
Gradio interface. - `prompt_parser`: An instance of `OpenAIInstructionParser` to
parse the prompt.

### Return

- A Gradio interface for interacting with the model.

## Interface Components

The Gradio interface consists of the following components:

- A header displaying the title "Prompt2Model". - Task description and few-shot
examples parsed from the prompt. - A chatbot interface for interacting with the
model. - A textbox for user input. - Two buttons: "Submit" to submit the
user input to the model and "Clear History" to reset the chat history.

## Functionality

- The "Submit" button triggers the model prediction on the current user input
and updates the chatbot interface and chat history. - The "Clear History" button
resets the chatbot interface and chat history. - The chatbot interface displays
the users and model conversation history. - The model's responses
are post-processed to convert Markdown to HTML for better readability.

Please ensure you have installed the necessary dependencies (`gradio` and
`mdtex2html`) and have a properly configured `GenerationModelExecutor` and
`OpenAIInstructionParser`.
