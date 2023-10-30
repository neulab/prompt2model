"""Place to store all the default configurations."""
MAX_SUPPORTED_BATCH_SIZE = 4

DEFAULT_HYPERPARAMETERS_SPACE = {
    "min_num_train_epochs": 5,
    "max_num_train_epochs": 15,
    "save_strategy": ["no"],
    "evaluation_strategy": ["no"],
    "per_device_train_batch_size": MAX_SUPPORTED_BATCH_SIZE,
    "min_weight_decay": 4e-5,
    "max_weight_decay": 1e-1,
    "min_learning_rate": 4e-5,
    "max_learning_rate": 1e-1,
}
