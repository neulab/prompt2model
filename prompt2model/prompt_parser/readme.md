# PromptParser Usage

## PromptSpec

`PromptSpec` is an interface for prompt parsing. It provides a structured way to
parse and store information about a prompt. The `PromptSpec` class defines two
abstract methods, `parse_from_prompt()` and `get_instruction()`, which its
subclasses must implement.

To see an example of how to use `PromptSpec` and its subclasses, you can refer
to the unit tests in the [prompt_parser_test.py](../../tests/prompt_parser_test.py)
file.

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

"`python
from prompt2model.prompt_parser import OpenAIInstructionParser, TaskType
```

- Initialize an instance of the `OpenAIInstructionParser`:

"`python
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

Please ensure you have valid OpenAI API credentials and adjust the unit tests
accordingly to match your setup.
