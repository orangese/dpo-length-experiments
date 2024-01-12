import os
import getpass
import subprocess
from datetime import datetime
import torch
import random
import numpy as np
import torch.distributed as dist
import inspect
import importlib.util
import socket
import gcsfs
from typing import Dict, Union, Type, List

GCP_BUCKET = None
USING_XLA = False
try:
    import torch_xla.core.xla_model as xm
except (ModuleNotFoundError, ImportError):
    print("WARNING: torch_xla not found")


def fmt_msg(msg, master=False):
    """Format a message with timestamp and ordinal."""
    ordinal = xm.get_ordinal() if not master else "MASTER"
    now = datetime.now()
    time_fmt = now.strftime("%m-%d %H:%M:%S") + "." + str(now.microsecond)[:3]
    return f"[{time_fmt}] [XM:{ordinal}] {msg}"


def mprint(msg):
    """Master print on master ordinal"""
    xm.master_print(fmt_msg(msg, master=True))


def lprint(msg):
    """Local print on any ordinal"""
    print(fmt_msg(msg))


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0)) # bind to all interfaces and use an OS provided port
        return s.getsockname()[1] # return only the port number


def get_gcp_project():
    """Gets current gcp project."""
    command = "gcloud config get-value project".split()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise ValueError(f"gcloud failed: {result.stderr}")


def upload_to_gcp(local_path: str, gcp_path: str):
    """Uploads a file to gcp."""
    command = f"gsutil cp {local_path} {gcp_path}".split()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise ValueError(f"gsutil upload failed: {result.stderr}")


def download_from_gcp(gcp_path: str, local_path: str):
    """Downloads a file from gcp."""
    command = f"gsutil cp {gcp_path} {local_path}".split()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise ValueError(f"gsutil download failed: {result.stderr}")


def load_from_gcp(gcp_path: str, **kwargs):
    """Load torch model directly from GCS filesystem"""
    fs = gcsfs.GCSFileSystem(project=get_gcp_project())
    with fs.open(gcp_path, "rb") as f:
        return torch.load(f, **kwargs)


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if USING_XLA:
        xm.master_print(*args, **kwargs)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"
    

def get_local_run_dir(exp_name: str, local_dirs: List[str]) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = f"{get_local_dir(local_dirs)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    if USING_XLA:
        # this should never be called with xla since preference_datasets has a native distrbuted dataloader
        raise NotImplementedError("slice_and_move_batch_for_device not implemented for XLA")
    
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if USING_XLA:
        # this should never be called with xla to minimize communication between xmp processes
        raise NotImplementedError("all_gather_if_needed not implemented for XLA")

    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(model_class: Type, block_class_name: str) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith('.py'), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find('transformers'):].replace('/', '.')[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(rank: int, world_size: int, master_addr: str = 'localhost', port: int = 12355, backend: str = 'nccl'):
    print(rank, 'initializing distributed')
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)
