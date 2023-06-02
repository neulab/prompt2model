# prompt2model Pipeline

The `run_skeleton.py` script demonstrates the execution of the `prompt2model` pipeline locally using mock/stub components. This pipeline covers various stages, including dataset retrieval, dataset generation, dataset processing, model retrieval, model training, model execution, evaluation, and interface creation.

## Usage

The script can be executed with the following command:

```bash
python run_skeleton.py --prompt <prompt> [--metrics-output-path <metrics_output_path>]
```

The script accepts the following arguments:

- `--prompt`: A prompt indicate a task to solve, including optional few-shot examples. This is the main input for the language model.
- `--metrics-output-path` (optional): The path to a JSON file where the model metrics will be stored. By default, the metrics are saved to "/tmp/metrics.json".

## Pipeline Overview

The `run_skeleton` function serves as the entry point for the pipeline. It takes the input prompt str and the metrics output path as arguments.

### Preprocessing

The `process_input_prompt` function preprocesses the input prompt given by the user. It removes quotation marks at the start and end of the prompt if they are present. The function returns a `PromptSpec` object, which is used to parse the prompt and specify the task type.

### Dataset Retrieval

The pipeline begins with dataset retrieval. The `MockRetriever` class is used to retrieve dataset dictionaries from HuggingFace based on the given `PromptSpec`. The retrieved dataset dictionaries represent the training, validation, and testing datasets.

### Dataset Generation

The `MockDatasetGenerator` class is used to generate additional dataset dictionaries from OpenAI LLMs based on the given `PromptSpec` and the desired number of examples for each dataset split (train, validation, and test).

### Dataset Processing

The pipeline then proceeds with dataset processing. The `MockProcessor` class is used to process the retrieved and generated dataset dictionaries. The processor has the option to include an encoder in the processing step. The processed dataset dictionaries are obtained by calling the `process_dataset_dict` method of the processor.

### Model Retrieval

The next step is model retrieval. The `MockModelRetriever` class is used to retrieve a model name from HuggingFace based on the given `PromptSpec`. The model name represents a pre-trained model suitable for the task specified in the prompt.

### Model Training

The `MockTrainer` class is used for model training. It takes the retrieved model name and performs training on the processed training datasets. The trainer prepares the model for subsequent steps.

### Model Selection

The `MockParamSelector` class is responsible for model selection. It takes the trained model and tokenizer, along with the validation dataset, and selects the best model based on hyperparameters or other selection criteria.

### Model Execution

The selected model and tokenizer are used to create a `MockModelExecutor` instance. The model executor is responsible for executing the model on the testing dataset, generating model outputs (predictions), and providing confidence scores.

### Evaluation

The pipeline continues with model evaluation. The `MockEvaluator` class is used to evaluate the model outputs. It computes metrics, such as accuracy, precision, recall, or any other relevant metrics based on the task specified in the prompt.

### Metrics Output

The evaluated metrics are written to a JSON file specified by the `metrics_output_path`. The `MockEvaluator` class handles writing the metrics dictionary to the file.

### Interface Creation

Finally, the `mock_gradio_create` function is called to create a Gradio interface for interacting with the model. This interface allows users to input text and receive model responses in real time.

## Mock Components

Note that the pipeline in `run_skeleton.py` utilizes mock versions.

 of each component for demonstration purposes. In a real-world scenario, you would replace these mock components with the actual implementations specific to your use case. The mock components serve as stubs that mimic the behavior of the real components but do not perform actual data retrieval, processing, training, or evaluation.

It is recommended to replace the mock components with the appropriate implementations based on your specific requirements to utilize the full potential of the `prompt2model` pipeline.
