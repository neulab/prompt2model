"""Place to store all the default configurations."""
MAX_SUPPORTED_BATCH_SIZE = 128

# All the default hyperparameters has to be set to some standardised settings

DEFAULT_HYPERPARAMETERS_SPACE = {
    "min_num_train_epochs": 10,
    "max_num_train_epochs": 20,
    "save_strategy": ["epoch", "steps", "no"],
    "evaluation_strategy": ["epoch", "no"],
    "per_device_train_batch_size": [2, 4],
    "min_weight_decay": 4e-5,
    "max_weight_decay": 1e-1,
    "min_learning_rate": 4e-5,
    "max_learning_rate": 1e-1,
}
