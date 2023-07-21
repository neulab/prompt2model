# PromptParser Usage

## PromptSpec

In our pipeline, a prompt is defined as an instruction (task description) for
the user's requirement and optional few-shot examples following
this requirement.`PromptSpec` is an interface for parsing the prompt
to get the instruction and few-shot examples. It provides
a structured way to parse and store information about a prompt input
by any user. The `PromptSpec` class defines three abstract
 methods `parse_from_prompt`, `get_instruction`, and
`get_examples`, which its subclasses must implement. Note that
the `get_instruction` and `get_examples` are warped as `property`
and should be used like `prompt_spec.get_examples`.

## TaskType

`TaskType` is an enumeration that defines a high-level taxonomy of possible NLP
model outputs. It provides constants for different types of tasks, such as text
generation, classification, sequence tagging, and span extraction.

## OpenAIInstructionParser

`OpenAIInstructionParser` is a subclass of `PromptSpec` that parses a prompt to
separate instructions from task demonstrations. It utilizes OpenAI's GPT-3.5 API
to perform the parsing. The class provides methods to extract the instruction
and demonstrations from the API response.

## Usage

- Import the necessary modules:

```python
from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
```

- Initialize an instance of the `OpenAIInstructionParser`:

```python
prompt_spec = OpenAIInstructionParser(task_type=TaskType.<task_type>)
```

You can optionally pass the OpenAI API key and the maximum number of API calls
as parameters. If you don't pass in the OpenAI API key, please set the
environment variable `OPENAI_API_KEY`:

```bash
export OPENAI_API_KEY=<your key>
```

- Parse the prompt using the `parse_from_prompt()` method:

```python
prompt = "<your-prompt>"
prompt_spec.parse_from_prompt(prompt)
```

The `parse_from_prompt()` method sends the prompt to the GPT-3.5 API and
extracts the instruction and demonstrations from the API response.

- Access the parsed fields:

```python
instruction = prompt_spec.get_instruction
demonstrations = prompt_spec.get_examples
```

The `get_instruction` property returns the parsed instruction and the
`get_examples` property returns the parsed demonstrations.

Please ensure you have valid OpenAI API credentials and feel free to
adjust and create subclasses to match your setup.
