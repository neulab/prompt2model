# prompt2model Pipeline

The `run_skeleton.py` script demonstrates the execution of the `prompt2model`
pipeline locally using mock components. This pipeline covers various
stages, including dataset retrieval, dataset generation, dataset processing,
model retrieval, model training, model execution, evaluation, and interface
creation.

## Usage

The script can be executed with the following command:

```bash
python run_skeleton.py --prompt <prompt> [--metrics-output-path <metrics_output_path>]
```

The script accepts the following arguments:

- `--prompt`: A prompt indicates a task to solve, including optional few-shot
examples. This is the main input for LLMs.
`--metrics-output-path` (optional): The path to a JSON file storing the model metrics.
By default, the metrics are saved to "/tmp/metrics.json".

## Pipeline Overview

The `run_skeleton` function serves as the entry point for the pipeline. It takes
the input prompt str and the metrics output path as arguments.

### Preprocessing

The `process_input_prompt` function preprocesses the input prompt given by the
user. It removes quotation marks at the start and end of the prompt if they are
present. The function returns a `PromptSpec` object, which is used to parse the
prompt and specify the task type.

### Dataset Retrieval

The pipeline begins with dataset retrieval. The `MockRetriever` class is used to
retrieve dataset from HuggingFace based on the given `PromptSpec`.
The retrieved dataset contains the training, validation, and
testing dataset splits.

### Dataset Generation

The `MockDatasetGenerator` class is used to generate the additional datasets
from OpenAI LLMs based on the given prompt and the desired
number of examples for each dataset split (train, validation, and test).

### Dataset Processing

The pipeline then proceeds with dataset processing. The `MockProcessor` class is
used to process the retrieved and generated dataset, transferring the data
into seq2seq fashion. The processed
datasets are obtained by calling the `process_dataset_dict` method
of the processor.

### Model Retrieval

The next step is model retrieval. The `MockModelRetriever` class is used to
retrieve a model name of a pre-trained model on HuggingFace,
suitable for the task specified in the prompt.

### Model Training

The `MockTrainer` class is used for model training. It takes the retrieved model
name and performs training on the processed training datasets. The trainer
prepares the model for subsequent steps.

### Model Selection

The `MockParamSelector` class is responsible for model selection. It takes the
trained model and tokenizer, along with the validation dataset, and selects the
best model among several hyperparameters or other selection criteria.

### Model Execution

The selected model and tokenizer are used to create a `MockModelExecutor`
instance. The model executor executes the model on the
testing dataset, generates model outputs (predictions), and provides
confidence scores.

### Evaluation

The pipeline continues with model evaluation. The `MockEvaluator` class is used
to evaluate the model outputs. It computes metrics, including ChrF++, Exact Match,
and BERTScore.

### Metrics Output

The `MockEvaluator` class evaluates and writes the metrics dictionary to a JSON
file specified by the `metrics_output_path`. 

### Interface Creation

Finally, the `mock_gradio_create` function creates a Gradio
interface for interacting with the model. This interface allows users to input
text and receive model responses in real-time.

## Mock Components

Note that the pipeline in `run_skeleton.py` utilizes mock versions of each component
for demonstration purposes. In a real-world scenario, you
would replace these mock components with the actual implementations specific to
your use case. The mock components serve as stubs that mimic the behavior of the
real components but do not perform actual data retrieval, processing, training,
or evaluation.

It is recommended to replace the mock components with the appropriate
implementations based on your specific requirements to utilize the full
potential of the `prompt2model` pipeline.
