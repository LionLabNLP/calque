from __future__ import annotations

import os

import torch

_owns_process_group = False


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def _process_group_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def init_distributed_for_tp() -> None:
    global _owns_process_group

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())

    if _process_group_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    _owns_process_group = True


def distributed_barrier() -> None:
    if _process_group_initialized():
        torch.distributed.barrier()


def cleanup_distributed() -> None:
    global _owns_process_group

    if _owns_process_group and _process_group_initialized():
        torch.distributed.destroy_process_group()
    _owns_process_group = False


def model_input_device(model):
    if is_distributed() and _process_group_initialized() and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return model.device
