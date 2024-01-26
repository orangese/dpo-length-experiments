import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
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


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_sample(rank: int, world_size: int, config: DictConfig, policy: nn.Module):
    """Samples from model (only BasicTrainer supported)."""
    config.n_eval_examples = None
    print('warning: setting config.n_eval_examples to none, use n_eval_model_samples to control how many samples')

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=None, rank=rank, world_size=world_size)

    to_save = trainer.sample(n_per=config.samples_per_prompt)
    with open(config.sample_path, "w+") as d:
        json.dump(to_save, d, indent=4)
    print(f'Saved samples on {len(to_save)} eval prompts to {config.sample_path}')


def worker_rewards(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: nn.Module):
    """Gets rewards from model (only BasicTrainer supported)."""
    config.n_eval_examples = None
    print('warning: setting config.n_eval_examples to none, use n_eval_model_samples to control how many rewards are sampled')

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    to_save = trainer.get_rewards()
    to_save.to_csv(config.rewards_save_path, index=False)
    print(f'Saved rewards on {len(to_save)} eval prompt batches to {config.rewards_save_path}')


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
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

    if config.reward_only:
        assert config.loss.name == "dpo", "for reward sampling, use loss = dpo"
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced', "trust_remote_code": True} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    if config.loss.name == 'dpo' and not config.sample_only and not config.save_as_hf:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name == 'dpo' and not config.sample_only and not config.save_as_hf:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    if config.save_as_hf is not None:
        assert config.trainer == 'BasicTrainer', "save with BasicTrainer"
        print(f"saving hf format to {config.save_as_hf}")
        policy.save_pretrained(config.save_as_hf)
        print("done saving, exiting")
        return

    if config.sample_only:
        print(f'not training, just sampling (saving to {config.sample_path})')
        worker_sample(0, 1, config, policy)
        return
    
    if config.reward_only:
        if config.policy_archive is not None:
            state_dict = torch.load(config.policy_archive, map_location='cpu')
            step, metrics = state_dict['step_idx'], state_dict['metrics']
            print(f'loading pre-trained policy weights at step {step} from {config.policy_archive} with metrics {json.dumps(metrics, indent=2)}')
            policy.load_state_dict(state_dict['state'])
            print('loaded pre-trained weights')
        print(f'not training, just getting rewards (saving to {config.sample_path})')
        worker_rewards(0, 1, config, policy, reference_model)
        return
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
