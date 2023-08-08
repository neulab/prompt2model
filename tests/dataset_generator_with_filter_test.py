"""Testing DatasetGenerator through OpenAIDatasetGenerator."""

import gc
import os
import tempfile
from collections import Counter, namedtuple

from datasets import Dataset

from prompt2model.dataset_generator.openai_gpt import OpenAIDatasetGenerator
from test_helpers import are_datasets_identical

# Define a namedtuple to represent an example with 'input_col' and 'output_col' fields.
Example = namedtuple("Example", ["input_col", "output_col"])


def test_construct_map_with_duplicate_inputs_unique_outputs():
    """Test constructing a map with duplicate inputs but unique outputs.

    This test case verifies the behavior of the construct_input_output_map()
    method in OpenAIDatasetGenerator when there are duplicate inputs but
    unique outputs in the generated examples.

    Attributes:
        api_key (str): The fake API key used for testing.
        expected_output (dict): The expected input-output map to be constructed.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a list of generated examples with duplicate inputs and unique outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="E"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
        ]

        # Call the construct_input_output_map()
        # method to create the input-output map.
        input_output_map = data_generator.construct_input_output_map(generated_examples)

        # The expected input-output map afte
        # r constructing it from the generated examples.
        expected_output = {
            "apple": Counter({"A": 1, "E": 1, "D": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }

        # Assertions to verify that the input-output
        # map matches the expected output.
        assert input_output_map == expected_output

    # Collect garbage to release memory
    # resources after the test.
    gc.collect()


def test_construct_map_with_duplicate_inputs_duplicate_outputs():
    """Test constructing a map with duplicate inputs and duplicate outputs.

    This test case verifies the behavior of the construct_input_output_map()
    method in OpenAIDatasetGenerator when there are duplicate inputs and
    duplicate outputs in the generated examples.

    Attributes:
        api_key (str): The fake API key used for testing.
        expected_output (dict): The expected input-output map to be constructed.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a list of generated examples with
        # duplicate inputs and duplicate outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="C"),
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="apple", output_col="G"),
            Example(input_col="apple", output_col="A"),
            Example(input_col="orange", output_col="O"),
            Example(input_col="apple", output_col="D"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="F"),
        ]

        # Call the construct_input_output_map()
        # method to create the input-output map.
        input_output_map = data_generator.construct_input_output_map(generated_examples)

        # The expected input-output map after
        # constructing it from the generated examples.
        expected_output = {
            "apple": Counter({"A": 3, "D": 1, "G": 1}),
            "banana": Counter({"B": 2, "C": 1}),
            "orange": Counter({"O": 1, "F": 1}),
        }

        # Assertions to verify that the input-output
        # map matches the expected output.
        assert input_output_map == expected_output

    # Collect garbage to release memory
    # resources after the test.
    gc.collect()


def test_construct_map_with_unique_inputs_outputs():
    """Test constructing a map with unique inputs and outputs.

    This test case verifies the behavior of the construct_input_output_map()
    method in OpenAIDatasetGenerator when all generated examples have unique
    inputs and outputs.

    Attributes:
        api_key (str): The fake API key used for testing.
        expected_output (dict): The expected input-output map to be constructed.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create a list of generated examples with unique inputs and outputs.
        generated_examples = [
            Example(input_col="apple", output_col="A"),
            Example(input_col="banana", output_col="B"),
            Example(input_col="orange", output_col="O"),
        ]

        # Call the construct_input_output_map()
        # method to create the input-output map.
        input_output_map = data_generator.construct_input_output_map(generated_examples)

        # The expected input-output map after
        # constructing it from the generated examples.
        expected_output = {
            "apple": Counter({"A": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }

        # Assertions to verify that the input-output
        # map matches the expected output.
        assert input_output_map == expected_output

    # Collect garbage to release memory
    # resources after the test.
    gc.collect()


def test_construct_map_with_empty_examples_list():
    """Test constructing a map with an empty list of inputs and outputs.

    This test case verifies the behavior of the construct_input_output_map()
    method in OpenAIDatasetGenerator when no generated examples are available.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Create an empty list of generated examples.
        generated_examples = []

        # Call the construct_input_output_map()
        # method to create the input-output map.
        input_output_map = data_generator.construct_input_output_map(generated_examples)

        # The input-output map should be empty
        # when there are no generated examples.
        assert input_output_map == {}

    # Collect garbage to release memory
    # resources after the test.
    gc.collect()


def test_multi_vote_with_duplicate_inputs_unique_outputs():
    """Test multi-voting with duplicate inputs but unique outputs.

    This test case verifies the application of multi-voting mechanism in the
    apply_multi_vote_to_construct_generated_dataset() method of
    OpenAIDatasetGenerator. It specifically tests the scenario when
    the input-output map contains duplicate inputs but unique outputs.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Provide an input-output map with duplicate inputs but unique outputs.
        input_output_map = {
            "apple": Counter({"A": 1, "E": 1, "D": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }

        # Apply multi-voting mechanism to construct the generated dataset.
        generated_dataset = (
            data_generator.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )
        )

        # Define the expected dataset after multi-voting.
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_multi_vote_with_duplicate_inputs_duplicate_outputs():
    """Test multi-voting with duplicate inputs and duplicate outputs.

    This test case verifies the application of multi-voting mechanism in the
    apply_multi_vote_to_construct_generated_dataset() method of
    OpenAIDatasetGenerator. It specifically tests the scenario when
    the input-output map contains duplicate inputs and duplicate outputs.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with filter_duplicated_examples=True.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(
            filter_duplicated_examples=True, cache_root=cache_dir
        )

        # Provide an input-output map with duplicate inputs and duplicate outputs.
        input_output_map = {
            "apple": Counter({"A": 3, "D": 1, "G": 1}),
            "banana": Counter({"B": 2, "C": 1}),
            "orange": Counter({"O": 1, "F": 1}),
        }

        # Apply multi-voting mechanism to construct the generated dataset.
        generated_dataset = (
            data_generator.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )
        )

        # Define the expected dataset after multi-voting.
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_multi_vote_with_unique_inputs_outputs():
    """Test multi-voting with unique inputs and outputs.

    This test case verifies the application of the multi-voting mechanism in the
    apply_multi_vote_to_construct_generated_dataset() method of OpenAIDatasetGenerator.
    It specifically tests the scenario when the input-output map contains unique
    inputs and outputs.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Set a fake API key in the environment variable for testing purposes.
    os.environ["OPENAI_API_KEY"] = "fake_api_key"

    # Initialize the OpenAIDatasetGenerator with an empty input-output map.
    with tempfile.TemporaryDirectory() as cache_dir:
        data_generator = OpenAIDatasetGenerator(cache_root=cache_dir)

        # Provide an input-output map with unique inputs and outputs.
        input_output_map = {
            "apple": Counter({"A": 1}),
            "banana": Counter({"B": 1}),
            "orange": Counter({"O": 1}),
        }

        # Apply multi-voting mechanism to construct the generated dataset.
        generated_dataset = (
            data_generator.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )
        )

        # Define the expected dataset after multi-voting.
        expected_dataset = Dataset.from_dict(
            {"input_col": ["apple", "banana", "orange"], "output_col": ["A", "B", "O"]}
        )

        # Verify that the generated dataset matches the expected dataset.
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()


def test_multi_vote_with_empty_examples_list():
    """Test multi-voting with empty inputs and outputs.

    This test case verifies the application of the multi-voting mechanism in the
    apply_multi_vote_to_construct_generated_dataset() method of OpenAIDatasetGenerator.
    It specifically tests the scenario when the input-output map is empty.

    Attributes:
        api_key (str): The fake API key used for testing.
    """
    # Initialize the OpenAIDatasetGenerator with an empty input-output map.
    with tempfile.TemporaryDirectory() as cache_dir:
        os.environ["OPENAI_API_KEY"] = "fake_api_key"
        data_generator = OpenAIDatasetGenerator(
            cache_root=cache_dir, filter_duplicated_examples=True
        )

        # Set the input-output map to be empty.
        input_output_map = {}

        # Apply multi-voting mechanism to construct the generated dataset.
        generated_dataset = (
            data_generator.apply_multi_vote_to_construct_generated_dataset(
                input_output_map
            )
        )

        # Define the expected dataset after multi-voting (empty dataset).
        expected_dataset = Dataset.from_dict({})

        # Verify that the generated dataset matches
        # the expected dataset (empty dataset).
        assert are_datasets_identical(generated_dataset, expected_dataset)

    # Collect garbage to release memory resources after the test.
    gc.collect()
