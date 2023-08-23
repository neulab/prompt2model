# Model Evaluator

## Overview

- **ModelEvaluator**: Interface for evaluating models‘ outputs.
- **Seq2SeqEvaluator**: Offers metrics (`ChrF++`, `Exact Match`,
`BERTScore`) for evaluating conditional generation models on specific
datasets.

## Getting Started

- **Import Required Modules**:

```python
from prompt2model.evaluator import Seq2SeqEvaluator
from prompt2model.model_executor import ModelOutput
```

- **Instantiate Seq2SeqEvaluator**:

```python
evaluator = Seq2SeqEvaluator()
```

- **Prepare Dataset & Predictions**:

1. Autoregressive models might include the input in their outputs. For
such evaluations, refer to the document string of
 `model_input_column` in `Seq2SeqEvaluator.evaluate_model`.
2. Default metrics: `ChrF++`, `Exact Match`, `BERTScore`.

```python
PREDICTIONS = [...]
VALIDATION_DATASET = ...
# Dataset with ground truth column `gt_column` and optionally input column `model_input_column`.
```

- **Evaluate**:

```python
metric_values = evaluator.evaluate_model(
  dataset=VALIDATION_DATASET,
  gt_column="model_ouput",
  predictions=PREDICTIONS,
)
```
