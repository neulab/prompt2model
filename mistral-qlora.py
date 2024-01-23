from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import load_dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

train_dataset = load_dataset("gem/viggo", split="train")
print(train_dataset)
