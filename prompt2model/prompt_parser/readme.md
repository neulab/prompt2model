# Prompt Parser

## Overview

- `PromptSpec`: Interface for parsing prompts into instructions and
examples.
- `TaskType`: Enum for classifying NLP tasks like text generation,
classification, etc.
- `OpenAIInstructionParser`: A `PromptSpec` subclass that uses GPT-3.5
API for parsing.

## Getting Started

- Import Modules:

```python
from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
```

- Setup OpenAI API Key and Initialize Parser:

```python
api_key = "<your-api-key>"
task_type = TaskType.<task_type>  # Choose the type of NLP task.
prompt_spec = OpenAIInstructionParser(task_type, api_key)
```

Or, set the API Key as an environment variable:

```bash
export OPENAI_API_KEY="<your-api-key>"
```

And then initialize the Parser:

```python
task_type = TaskType.<task_type>
prompt_spec = OpenAIInstructionParser(task_type)
```

### Parse the Prompt

```python
prompt = "<your-prompt>"
prompt_spec.parse_from_prompt(prompt)
```

### Access Parsed Fields

```python
instruction = prompt_spec.get_instruction  # Retrieves parsed instruction.
demonstrations = prompt_spec.get_examples  # Retrieves parsed examples.
```

### Mock

If you want to mock a `PromptSpec` object without parsing from a prompt,
you can use the `MockPromptSpec` class.

```python
prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
instruction = """...""" # A string indicating the task description.
examples = """...""" # A string indicating the examples.
prompt_spec._instruction = prompt
prompt_spec._examples = examples
```
