import json

# Replace this with the path to your JSON file
# json_file_path = '../../huggingface_data/huggingface_datasets/dataset_index.json'
json_file_path = "mp10.json"


with open(json_file_path, "r") as file:
    data_dict = json.load(file)
    breakpoint()

    print("Data loaded successfully.")
    # Use data_dict as a regular Python dictionary
    # For example, you ca
