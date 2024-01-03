"""Testing dataset utility functions."""
from unittest.mock import patch

from prompt2model.utils import dataset_utils


@patch("prompt2model.utils.dataset_utils.query")
def test_get_dataset_size(mock_request):
    """Test function for get_dataset_size."""
    mock_request.return_value = {
        "size": {
            "dataset": {
                "dataset": "rotten_tomatoes",
                "num_bytes_original_files": 487770,
                "num_bytes_parquet_files": 881052,
                "num_bytes_memory": 1345449,
                "num_rows": 10662,
            },
            "configs": [
                {
                    "dataset": "rotten_tomatoes",
                    "config": "default",
                    "num_bytes_original_files": 487770,
                    "num_bytes_parquet_files": 881052,
                    "num_bytes_memory": 1345449,
                    "num_rows": 10662,
                    "num_columns": 2,
                }
            ],
            "splits": [
                {
                    "dataset": "rotten_tomatoes",
                    "config": "default",
                    "split": "train",
                    "num_bytes_parquet_files": 698845,
                    "num_bytes_memory": 1074806,
                    "num_rows": 8530,
                    "num_columns": 2,
                },
                {
                    "dataset": "rotten_tomatoes",
                    "config": "default",
                    "split": "validation",
                    "num_bytes_parquet_files": 90001,
                    "num_bytes_memory": 134675,
                    "num_rows": 1066,
                    "num_columns": 2,
                },
                {
                    "dataset": "rotten_tomatoes",
                    "config": "default",
                    "split": "test",
                    "num_bytes_parquet_files": 92206,
                    "num_bytes_memory": 135968,
                    "num_rows": 1066,
                    "num_columns": 2,
                },
            ],
        },
        "pending": [],
        "failed": [],
        "partial": False,
    }
    assert dataset_utils.get_dataset_size("rotten_tomatoes") == "1.28"
