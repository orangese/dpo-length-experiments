import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
import utils
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource

try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.core.xla_model as xm
except (ModuleNotFoundError, ImportError):
    print("WARNING: torch_xla not found")


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_sample(rank: int, world_size: int, config: DictConfig, policy: nn.Module):
    """Samples from model (only BasicTrainer supported)."""
    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=None, rank=rank, world_size=world_size)

    to_save = trainer.sample(n_per=config.samples_per_prompt)
    with open(config.sample_path, "w+") as d:
        json.dump(to_save, d, indent=4)
    print(f'Saved samples on {len(to_save)} eval prompts to {config.sample_path}')


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    utils.USING_XLA = config.use_tpu
    utils.GCP_BUCKET = config.gcp_bucket

    if 'FSDP' in config.trainer and not config.use_tpu:
        init_distributed(rank, world_size, port=config.fsdp_port)
 
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if config.wandb.enabled and ((not config.use_tpu and rank == 0) or (config.use_tpu and xm.is_master_ordinal(local=False))):
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    if config.use_tpu:
        assert "XLA" in config.trainer, "only XLA trainers supported on TPU"
        xm.master_print(f"XLA: created {xm.xrt_world_size()} xrt processes")
    else:
        print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    #trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
   
    utils.USING_XLA = config.use_tpu
    print("NOTE: config.use_tpu:", config.use_tpu)
    if config.use_tpu:
        print("NOTE: FSDP implementation differs on TPU vs GPU; using XLA implementation")

    try:
        print(f"NOTE: using GCP bucket {config.gcp_bucket}")
    except:
        print("NOTE: no GCP bucket provided")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if not config.use_tpu and 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
    
    if config.use_tpu:
        config.model.policy_dtype = "float32"
        config.model.reference_dtype = "float32"
        print("NOTE: XLA FDSP requires float32 parameters")
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print(f'building policy ({config.model.policy_dtype=})')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)

    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    if config.loss.name == 'dpo' and not config.sample_only:
        print(f'building reference model ({config.model.reference_dtype=})')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.archive is not None:
        if config.gcp_bucket is not None and config.model.archive.startswith("gs://"):
            print(f"downloading archive from gcp path", config.model.archive)
            archive_path = config.model.archive.replace(config.gcp_bucket, os.path.join(get_local_dir(config.local_dirs)))
            if not os.path.exists(archive_path):
                utils.download_from_gcp(config.model.archive, archive_path)
                print(f"downloaded gcp archive to", archive_path)
            print("loaded gcp archive from", archive_path)

        else:
            archive_path = config.model.archive

        state_dict = torch.load(archive_path, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {archive_path} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name == 'dpo' and not config.sample_only:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    if config.sample_only:
        assert not config.use_tpu, 'sampling not supported on TPU'
        print(f'not training, just sampling (saving to {config.sample_path})')
        worker_sample(0, 1, config, policy)
        return
    
    if 'FSDP' in config.trainer:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
    
        if config.use_tpu:
            xmp.spawn(worker_main, args=(1, config, policy, reference_model))

        else:
            world_size = torch.cuda.device_count()
            print('GPU: starting', world_size, 'processes for FSDP training')
            mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)

    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
