# This is something we need to decide, what should be the max value
# because the more the max value, the more it will take time to get the
# max batch size

MAX_SUPPORTED_BATCH_SIZE = 128

DEFAULT_HYPERPARAMETERS = {
    "output_dir": "./result",
    "logging_steps": 1,
    "save_strategy": "no",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 2,
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "logging_dir": "./result",
    "learning_rate": 1e-4,
    "evaluation_strategy": "epoch",
    "test_size": 0.15,
}

DEFAULT_HYPERPARAMETERS_SPACE = {
    "min_num_train_epochs": 5,
    "max_num_train_epochs": 10,
    "save_strategy": ["epoch", "steps", "no"],
    "evaluation_strategy": ["epoch", "no"],
    "per_device_train_batch_size": [4, 8, 16, 32],
    "min_weight_decay": 1e-5,
    "max_weight_decay": 1e-1,
    "min_learning_rate": 1e-5,
    "max_learning_rate": 1e-1,
}