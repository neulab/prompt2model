import json
from huggingface_hub import list_datasets
from scripts.dataset_index.retrieve_dataset_info import process_datasets

og_datasets_file = "huggingface_data/huggingface_datasets/dataset_index.json"
curr_datasets_file = "huggingface_data/huggingface_datasets/reranking_dataset_index.json"
with open(curr_datasets_file, "r") as f:
            curr_datasets = json.load(f)
with open(og_datasets_file, "r") as f:
            og_datasets = json.load(f)

unique_descriptions = set([curr_datasets[x]['description'] for x in curr_datasets.keys()])
min_downloads=10

all_datasets = list(list_datasets())
all_datasets_dict = {x.id: x.__dict__ for x in all_datasets}
left_over_dataset = {key: og_datasets[key] for key in og_datasets.keys() - curr_datasets.keys()}
counter = 1
curr_counter = 0
possibly_useful_datasets = []
for dataset_name in left_over_dataset:
    print(f"on curr counter: {curr_counter} / {len(left_over_dataset)}")
    curr_counter+=1
    if dataset_name not in all_datasets_dict: continue
    dataset_info = all_datasets_dict[dataset_name]
    if "downloads" in dataset_info and dataset_info["downloads"] < min_downloads:
        continue
    if "description" not in left_over_dataset[dataset_name]: continue 
    if left_over_dataset[dataset_name]["description"] in unique_descriptions:
        continue
    if "description" not in dataset_info or dataset_info["description"] is None: 
        dataset_info["description"] = left_over_dataset[dataset_name]["description"]

    # possible_info = process_datasets([dataset_info], True) 
    possibly_useful_datasets.append(dataset_info)
    # if possible_info!= {}:
    #     print("got dataset!!", str(counter))
    #     counter+=1
    #     curr_datasets.update(possible_info)
new_useful_datasets = process_datasets(possibly_useful_datasets, True)

with open("new_curr_datastes.json", "w") as f:
    json.dump(new_useful_datasets, f)
