# Demo Creator

## Overview

- **create_gradio**: A function to set up and return a Gradio
interface for model interactions.

## Getting Started

- **Import the Necessary Modules**:

```python
from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import OpenAIInstructionParser
from prompt2model.gradio_interface import create_gradio
```

- **Initialize Components**:

```python
model_executor = GenerationModelExecutor(...)
prompt_parser = OpenAIInstructionParser(...)
# Refer to the documentation of ModelExecutor and PromptParser for details.
```

- **Create and Run the Gradio Interface**:

```python
interface = create_gradio(model_executor, prompt_parser)
interface.launch(shared=True)
```
