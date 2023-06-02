# Create Gradio Interface

The `create_gradio` function is a utility function that creates a Gradio interface for interacting with a generation model. The interface allows users to input a message or chat history and receive model-generated responses.

To see an example of how to use `create_gradio`, you can refer to the unit tests in the [demo_creator_test.py](../../tests/demo_creator_test.py) file.

## Usage

```python
import gradio as gr

from prompt2model.model_executor import GenerationModelExecutor
from prompt2model.prompt_parser import OpenAIInstructionParser
from prompt2model.demo_creator import create_gradio

# Create an instance of GenerationModelExecutor and OpenAIInstructionParser
model_executor = GenerationModelExecutor(...)
prompt_parser = OpenAIInstructionParser(...)

# Create the Gradio interface
interface = create_gradio(model_executor, prompt_parser)

# Run the interface
interface.launch()
```

The `create_gradio` function takes a `GenerationModelExecutor` instance and an `OpenAIInstructionParser` instance as inputs and returns a Gradio interface. The `GenerationModelExecutor` encapsulates the generation model and its execution logic, while the `OpenAIInstructionParser` handles parsing prompts and extracting instructions and examples.

To use the `create_gradio` function, you need to provide the appropriate `GenerationModelExecutor` and `OpenAIInstructionParser` instances based on your specific model and prompt parsing requirements.

After creating the Gradio interface, you can run it using the `launch` method, which opens the interface in a web browser. Users can input a message or chat history, and the model will generate responses based on the provided inputs.

Make sure to customize the `GenerationModelExecutor` and `OpenAIInstructionParser` instances to match your specific model and prompt parsing logic.

Please note that you need to have Gradio and its dependencies installed in your environment to use the `create_gradio` function and run the generated interface.

Feel free to modify the code as needed to fit your project's requirements and extend the functionality of the Gradio interface.
