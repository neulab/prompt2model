# load reranking_dataset_index.json
import json
with open('reranking_dataset_index.json') as f:
    reranking_dataset_index = json.load(f)

counter = 0

for dataset_name, top_dataset_info in reranking_dataset_index.items():
    for config_name, config_info in top_dataset_info["configs"].items():
        for key in config_info["columns_mapping"]:
            print(f"{counter=}")
            counter += 1
            
            new_key = config_info["columns_mapping"][key]
            if new_key!=key:
                config_info["sample_row"] = json.loads(config_info["sample_row"])
                config_info["sample_row"][new_key] = config_info["sample_row"][key]
                del config_info["sample_row"][key]
                config_info["sample_row"] = json.dumps(config_info["sample_row"])
                
        reranking_dataset_index[dataset_name]["configs"][config_name] = config_info


#write to reranking_dataset_index.json
with open('reranking_dataset_index_v2.json', 'w') as f:
    json.dump(reranking_dataset_index, f, indent=4) 