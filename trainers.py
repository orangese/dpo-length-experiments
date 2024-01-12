import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

try:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.fsdp import (
        XlaFullyShardedDataParallel as FSDP,
        consolidate_sharded_model_checkpoints,
        checkpoint_module,
    )
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import torch_xla.debug.metrics as met
except (ModuleNotFoundError, ImportError):
    print("WARNING: torch_xla not found")

from preference_datasets import get_batch_iterator, xla_get_dataloader
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
    upload_to_gcp
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: torch.FloatTensor,
             alpha: torch.FloatTensor,
             chosen_len: torch.FloatTensor,
             rejected_len: torch.FloatTensor,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        alpha: Length penalty parameter for loss, typically between 0.1 and 1.0
        chosen_len: Number of tokens in the chosen sequence, ignoring padding
        rejected_len: Number of tokens in the rejected sequence, ignoring padding
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios
    unscaled = beta * logits - alpha * rejected_len - alpha * chosen_len

    losses = -F.logsigmoid(unscaled)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels = torch.where(~loss_mask, torch.zeros_like(labels), labels)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1)


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, no_train: bool = False):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.

           If TPUs are available (config.use_tpu = True), you must use FDSPTrainerXLA for training! BasicTrainer
           will not work, nor will TensorParallelTrainer.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        if config.use_tpu or config.gcp_bucket:
            assert not self.config.sample_during_eval, "sampling during training not supported with xla/gcp"
            assert self.config.batch_size == self.config.eval_batch_size, "train batch size and eval batch size must be the same on xla"

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model
        
        dataloader_fn = get_batch_iterator
        data_iterator_kwargs["silent"] = rank != 0
        data_iterator_kwargs["n_epochs"] = config.n_epochs
        data_iterator_kwargs["n_examples"] = config.n_examples

        if config.use_tpu:
            dataloader_fn = xla_get_dataloader
            data_iterator_kwargs["device"] = xm.xla_device()
            data_iterator_kwargs["silent"] = not xm.is_master_ordinal(local=False)
            del data_iterator_kwargs["n_epochs"]
            del data_iterator_kwargs["n_examples"]

        if not no_train:
            self.train_iterator = dataloader_fn(**data_iterator_kwargs, split='train', batch_size=config.batch_size, cache_dir=get_local_dir(config.local_dirs))
            rank0_print(f'Loaded train data iterator ({config.use_tpu=})')
        else:
            self.train_iterator = None
            rank0_print('Did not load train data iterator')

        self.eval_iterator = dataloader_fn(**data_iterator_kwargs, split='test', batch_size=config.eval_batch_size, cache_dir=get_local_dir(config.local_dirs))
        if not config.use_tpu:
            self.eval_batches = None
            rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')
        else:
            self.eval_batches = list(self.eval_iterator)
            rank0_print(f"Did not pre-load eval batches since running with XLA data loader")

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor], use_reference: bool = False) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if use_reference:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if use_reference:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        all_logits = model(batch['concatenated_input_ids'], attention_mask=batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:self.config.batch_size]
        rejected_logps = all_logps[self.config.batch_size:]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True, lazy=False):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""
        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name == 'dpo':
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            losses, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
                beta=loss_config.beta,
                alpha=loss_config.alpha,
                reference_free=loss_config.reference_free,
                chosen_len=batch["chosen_len"],
                rejected_len=batch["rejected_len"]
            )
            chosen_rewards = chosen_rewards.detach()
            rejected_rewards = rejected_rewards.detach()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            if not lazy:
                chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
                rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
                reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies
            metrics[f'rewards_{train_test}/margins'] = chosen_rewards - rejected_rewards

            policy_rejected_logps = policy_rejected_logps.detach()
            if not lazy:
                policy_rejected_logps = all_gather_if_needed(policy_rejected_logps, self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = policy_chosen_logps.detach()
        if not lazy:
            policy_chosen_logps = all_gather_if_needed(policy_chosen_logps, self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps

        losses_view = losses.detach()
        if not lazy:
            losses_view = all_gather_if_needed(losses_view, self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = losses_view
 
        if not lazy:
            # do NOT execute with XLA! no native lowering implemented, will use xla->cpu callback
            # unsure if detach() is even a good idea with xla?
            for k, v in metrics.items():
                metrics[k] = v.cpu().numpy().tolist()

        return losses.mean(), metrics

    def sample(self, n_per=10):
        """Samples from self.policy over the evaluation set. Bootstraps n_per samples per prompt."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.n_eval_model_samples < self.config.eval_batch_size:
            rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
            sample_batches = self.eval_batches[:1]
        else:
            n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
            sample_batches = self.eval_batches[:n_sample_batches]

        result = {}
        self.policy.eval()

        for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            for i in range(n_per):
                samples, _ = self.get_batch_samples(local_eval_batch, use_reference=False)

                for prompt, sample in zip(eval_batch['prompt'], samples):
                    try:
                        result[prompt].append(sample)
                    except KeyError:
                        result[prompt] = [sample]

        return result
    
    @torch.no_grad
    def do_eval(self):
        """Run evaluation on the evaluation set, gathering metrics from all processes and logging only on the master rank process."""
        rank0_print(f'Running evaluation after {self.example_counter} train examples')
        self.policy.eval()

        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples, all_reference_samples = [], []
            policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
            if self.config.loss.name == 'dpo':
                reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])
        
        if self.rank == 0:
            eval_iter = tqdm.tqdm(self.eval_iterator, desc="Computing eval metrics")
        else:
            eval_iter = self.eval_iterator
    
        for eval_batch in eval_iter:
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)
            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)
 
        if self.config.sample_during_eval:
            if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                sample_batches = self.eval_batches[:1]
            else:
                n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                sample_batches = self.eval_batches[:n_sample_batches]
            for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                policy_samples, reference_samples = self.get_batch_samples(
                    local_eval_batch,
                    use_reference=self.config.loss.name == 'dpo'
                )

                all_policy_samples.extend(policy_samples)
                all_reference_samples.extend(reference_samples)

                for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                    policy_text_table.add_data(self.example_counter, prompt, sample)
                if self.config.loss.name == 'dpo':
                    for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                        reference_text_table.add_data(self.example_counter, prompt, sample)

        mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
        rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
        if self.config.sample_during_eval:                    
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))
            if self.config.loss.name == 'dpo':
                rank0_print(json.dumps(all_reference_samples[:10], indent=2))

        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)
            
            if self.config.sample_during_eval:
                wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                if self.config.loss.name == 'dpo':
                    wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

        if self.example_counter > 0:
            if self.config.debug:
                rank0_print('skipping save in debug mode')
            else:
                output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                rank0_print(f'creating checkpoint to write to {output_dir}...')
                self.save(output_dir, mean_eval_metrics)

    def do_train(self, batch, last_log):
        """Perform one training step on batch, non-FSDP."""
        self.policy.train()

        start_time = time.time()
        batch_metrics = defaultdict(list)
        for microbatch_idx in range(self.config.gradient_accumulation_steps):
            global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
            local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
            loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
            (loss / self.config.gradient_accumulation_steps).backward()

            for k, v in metrics.items():
                batch_metrics[k].extend(v)

        grad_norm = self.clip_gradient()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        step_time = time.time() - start_time
        examples_per_second = self.config.batch_size / step_time
        batch_metrics['examples_per_second'].append(examples_per_second)
        batch_metrics['grad_norm'].append(grad_norm)

        self.batch_counter += 1
        self.example_counter += self.config.batch_size

        if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
            mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
            mean_train_metrics['counters/examples'] = self.example_counter
            mean_train_metrics['counters/updates'] = self.batch_counter
            rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

            if self.config.wandb.enabled and self.rank == 0:
                wandb.log(mean_train_metrics, step=self.example_counter)

            last_log = time.time()
        else:
            rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')

        return last_log

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        if self.config.loss.name == 'dpo':
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        if self.train_iterator is None:
            raise ValueError("No train iterator loaded, cannot train")

        n_epochs = 0
        while True:
            for batch in self.train_iterator:
                if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                    self.do_eval()
                last_log = self.do_train(batch, last_log)
            
                if self.config.max_examples is not None and self.example_counter >= self.config.max_examples:
                    rank0_print(f'reached max_examples ({self.config.max_examples}), exiting')
                    return
            
            n_epochs += 1
            if self.config.n_epochs is not None and n_epochs >= self.config.n_epochs:
                rank0_print(f'reached n_epochs ({self.config.n_epochs}), exiting')
                return

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')

        data = {
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }
        if self.config.use_tpu:
            xm.save(data, output_path, master_only=True)
        else:
            torch.save(data, output_path)

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name == 'dpo':
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
   
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()


class FSDPTrainerXLA(BasicTrainer):

    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module], rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple XLA devices.
        
           This trainer will shard both the policy and reference model across all available XLA devices.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, 0, 1)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'
        rank0_print("Initializing FSDPTrainerXLA")

        # Move models to the proper device
        device = xm.xla_device()
        policy = policy.to(device)
        if reference_model:
            reference_model = reference_model.to(device)

        # Wrap submodules with FSDP too (zero 3)
        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, 
            transformer_layer_cls={wrap_class},
        )

        # Apply activation checkpointing
        auto_wrap_callable = None
        if config.activation_checkpointing:
            rank0_print("Enabling activation checkpointing")
            auto_wrap_callable = lambda m, *args, **kwargs: FSDP(
                checkpoint_module(m), *args, **kwargs
            )

        # Wrap the base and reference models
        rank0_print('Sharding policy...')
        mp_dtype = None
        if config.model.fsdp_policy_mp is not None:
            mp_dtype = getattr(torch, config.model.fsdp_policy_mp)

        fsdp_wrap = lambda m: FSDP(
            m,
            compute_dtype=mp_dtype,
            auto_wrap_policy=auto_wrap_policy,
            auto_wrapper_callable=auto_wrap_callable
        )
        self.policy = fsdp_wrap(policy)

        if config.loss.name == "dpo":
            rank0_print('Sharding reference model...')
            self.reference_model = fsdp_wrap(reference_model)

        print('Loaded model on rank', device)
    
    @staticmethod
    def _reduce_dict(dict_list):
        """Given list of dicts with same key sets, reduces into one dict by taking the 
        average over each key. Each key in each dict will map to a 1-D tensor of values."""
        result = {}
        for k in dict_list[0].keys():
            result[k] = 0
            for d in dict_list:
                result[k] += d[k].mean()
        return {k: v.item() / len(dict_list) for k, v in result.items()}

    @torch.no_grad
    def do_eval(self):
        """Run evaluation on the evaluation set, gathering metrics from all processes and logging only on the master rank process."""
        self.policy.eval()
        rank0_print(f'Running evaluation after {self.example_counter} train examples')

        all_eval_metrics = defaultdict(list)
        nb = self.config.n_eval_examples / (self.config.eval_batch_size * xm.xrt_world_size())
        if xm.is_master_ordinal(local=False):
            eval_iter = tqdm.tqdm(self.eval_iterator, desc="Computing eval metrics", total=nb)
        else:
            eval_iter = self.eval_iterator
   
        for i, local_eval_batch in enumerate(eval_iter):
            if i > nb:
                rank0_print(f"finished {i} eval batches ({self.config.n_eval_examples} examples) with batch size {self.config.batch_size}")
                break
            # already sliced and moved to correct XLA device by the data loader
            _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, lazy=True)
            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)
            xm.master_print(met.metrics_report())
            xm.master_print(f"nb={nb}, {self.config.n_eval_examples=}, {self.config.eval_batch_size=}, {xm.xrt_world_size()=}")
            xm.master_print(all_eval_metrics)

        # now we can mesh reduce across all processes
        mean_eval_metrics = xm.mesh_reduce('eval_metrics', all_eval_metrics, FSDPTrainerXLA._reduce_dict)
        rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')

        # log if appropriate
        if self.config.wandb.enabled and xm.is_master_ordinal(local=False):
            wandb.log(mean_eval_metrics, step=self.example_counter)

        if self.example_counter > 0:
            if self.config.debug:
                rank0_print('skipping save in debug mode')
            else:
                output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                rank0_print(f'creating checkpoint to write to {output_dir}...')
                self.save(output_dir, mean_eval_metrics)

    def do_train(self, batch, last_log):
        """Perform one training step on batch, FSDP."""

        def log_train_metrics(device, example_counter, batch_counter, batch_metrics, wandb_enabled):
            mean_train_metrics = xm.mesh_reduce('train_metrics', batch_metrics, FSDPTrainerXLA._reduce_dict)
            mean_train_metrics['counters/examples'] = example_counter
            mean_train_metrics['counters/updates'] = batch_counter
            rank0_print(f'[{device}] train stats after {example_counter} examples: {formatted_dict(mean_train_metrics)}')

            if wandb_enabled and xm.is_master_ordinal(local=False):
                wandb.log(mean_train_metrics, step=example_counter)

        self.policy.train()
        start_time = time.time()
        batch_metrics = defaultdict(list)

        chunk_size = self.config.batch_size // self.config.gradient_accumulation_steps
        for idx in range(self.config.gradient_accumulation_steps):
            # No need to slice the batch for device movement, since it's already on the correct XLA device
            # Slicing is solely for the purpose of gradient accumulation here
            end = (idx + 1) * chunk_size
            if self.config.batch_size - end < chunk_size:
                end = self.config.batch_size

            microbatch = {k: v[slice(idx * chunk_size, end)] for k, v in batch.items()}
            loss, metrics_microbatch = self.get_batch_metrics(microbatch, self.config.loss, train=True, lazy=True)
            (loss / self.config.gradient_accumulation_steps).backward()

            for k, v in metrics_microbatch.items():
                batch_metrics[k].append(v)

        # clip gradients and update parameters
        # (do not record gradient norm since it requires explicit access of intermediate tensor via item()
        # which doesn't have a native XLA translation)
        self.clip_gradient()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # gather metrics and log them
        step_time = time.time() - start_time
        examples_per_second = self.config.batch_size / step_time
        batch_metrics['examples_per_second'].append(examples_per_second)

        self.batch_counter += 1
        self.example_counter += self.config.batch_size

        # should use add_step_closure so that performance doesn't take a hit from accessing intermediate tensors
        if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
            xm.add_step_closure(
                log_train_metrics,
                args=(xm.xla_device(), self.example_counter, self.batch_counter, batch_metrics, self.config.wandb.enabled),
            )
            last_log = time.time()

        return last_log
    
    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all XLA devices."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm)

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk."""
        # Save model checkpoint
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        ckpt_prefix = f"{output_dir}/iter-{self.example_counter}"

        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
        ckpt = {
            'model': self.policy.state_dict(),
            'shard_metadata': self.policy.get_shard_metadata(),
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        xm.save(ckpt, ckpt_path, master_only=False)
        print(f'checkpoint saved to {ckpt_path}\n', end='')

         # Consolidate the sharded model checkpoints
        if xm.is_master_ordinal(local=False):
            consolidate_sharded_model_checkpoints(
                ckpt_prefix=ckpt_prefix,
                ckpt_suffix="_rank-*-of-*.pth"
            )
            rank0_print(f'consolidated checkpoint saved to {ckpt_prefix}_consolidated.pth\n', end='')
        
        xm.rendezvous('ckpt_consolidation')

        # Fix formats and upload to GCP on master process
        if xm.is_master_ordinal(local=False):
            self.coerce_ckpt_format(ckpt_prefix, metrics, output_dir)
    
    def coerce_ckpt_format(self, ckpt_prefix, metrics, output_dir):
        # format consolidated checkpoint to match original format
        ckpt = torch.load(f"{ckpt_prefix}_consolidated.pth", map_location="cpu")
        self.write_state_dict(self.example_counter, ckpt["model"], metrics, 'policy.pt', output_dir)
        del ckpt
    
        # remove the sharded checkpoints and consolidated checkpoint
        world_size = xm.xrt_world_size()
        for rank in range(world_size):
            os.remove(f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth')
        os.remove(f'{ckpt_prefix}_consolidated.pth')
        rank0_print(f'removed local shards and consolidated checkpoint\n', end='')

        # save optimizer and scheduler state
        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        del scheduler_state_dict

        # upload to gcp if available
        if self.config.gcp_bucket is not None:
            paths = [os.path.join(output_dir, f) for f in ["policy.pt", "optimizer.pt", "scheduler.pt"]]
            cache_dir = get_local_dir(self.config.local_dirs)
            save_dir = output_dir.replace(cache_dir, os.path.join(self.config.gcp_bucket, "runs"))
            for path in paths:
                upload_to_gcp(path, os.path.join(save_dir, path))
                os.remove(path)
            rank0_print(f'uploaded consolidated checkpoints to gcp\n', end='')

 
class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        
        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name == 'dpo':
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        
