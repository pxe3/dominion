import os
import torch
from torch.cuda import is_available
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType



def init_distributed(rank, world_size, backend: str = "nccl"):
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(rank)
    
    print(f"[Rank {rank}] Initialized (world_size={world_size})")


def wrap_model_fsdp(model):
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    )

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_full_state_dict(fsdp_model):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
        return fsdp_model.state_dict()