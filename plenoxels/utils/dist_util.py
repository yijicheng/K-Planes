"""
Helpers for distributed training.
"""

import io
import os
import socket
from typing import Callable, List
 
from mpi4py import MPI
import torch as th
import torch.distributed as dist
# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = th.cuda.device_count()

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    if not os.environ.get("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)
    port = comm.bcast(_find_free_port(), root=0)
    if not os.environ.get("MASTER_PORT"):
        os.environ["MASTER_PORT"] = str(port)
    th.cuda.set_device(dev())
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")

 


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def is_main_process():
    if th.cuda.is_available():
        return dev() == 0
    else:
        return True

def decorate_all(decorators: List[Callable]) -> Callable:
    """A decorator to decorate all member functions of a class

    Args:
        decorators: list of decorators to add to all functions in the class
    """

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr != "__init__":
                for decorator in decorators:
                    setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate

def check_main_thread(func: Callable) -> Callable:
    """Decorator: check if you are on main thread"""

    def wrapper(*args, **kwargs):
        ret = None
        if is_main_process():
            ret = func(*args, **kwargs)
        return ret

    return wrapper