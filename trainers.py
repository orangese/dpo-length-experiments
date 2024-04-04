import json
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
import pandas as pd
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

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
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
             beta: float,
             alpha: float,
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

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios
    unscaled = beta * logits

    if alpha != 0:
        unscaled -= alpha * rejected_len - alpha * chosen_len

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
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, no_train: bool = False):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

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
            deduplicate=config.sample_only or config.reward_only
        )

        self.policy = policy
        self.reference_model = reference_model

        if not no_train:
            self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
            rank0_print(f'Loaded train data iterator')
        else:
            self.train_iterator = None
            rank0_print('Did not load train data iterator')

        if config.n_eval_examples is None:
            n_epochs = 1
            n_examples = None
        else:
            n_epochs = None
            n_examples = config.n_eval_examples
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=n_examples, n_epochs=n_epochs, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor], use_reference: bool = False, num_beams: int = None, repetition_penalty: float = 1.0,
                          top_k: int = 50, penalty_alpha: float = 0.0, temperature: float = 1.0,
                          top_p: float = 1.0, no_repeat_ngram_size: int = 0) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""
        do_sample = temperature != 0

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=do_sample, pad_token_id=self.tokenizer.pad_token_id,
                num_beams=num_beams, repetition_penalty=repetition_penalty, top_k=top_k, penalty_alpha=penalty_alpha, temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size, top_p=top_p)

        if use_reference:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=do_sample, pad_token_id=self.tokenizer.pad_token_id,
                    num_beams=num_beams, repetition_penalty=repetition_penalty, top_k=top_k, penalty_alpha=penalty_alpha, temperature=temperature,
                    no_repeat_ngram_size=no_repeat_ngram_size, top_p=top_p)

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
    
    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], return_z: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        batch_size = batch['chosen_input_ids'].shape[0]

        if return_z:
            with torch.no_grad():
                prompt_logits = model(
                    batch['prompt_input_ids'],
                    attention_mask=batch['prompt_attention_mask'],
                ).logits.to(torch.float32)[:, -1, :]
                z = self.config.loss.beta * torch.logsumexp(prompt_logits / self.config.loss.beta, dim=1)

            """
            # dims: [batch size, sequence position index, vocab word index]
            # note that [:, i, :] gives logits for (i + 1)-th token given all i prior
            _, seq_len, vocab_size = all_logits.size()

            # chosen + rejected
            first_response_token_idx = batch['prompt_len'].to(torch.long).repeat(2) - 2
            gather_idxs = first_response_token_idx.view(-1, 1).expand(-1, vocab_size)
 
            # [batch_size, vocab_size]
            first_token_logits = torch.gather(all_logits, 1, gather_idxs.unsqueeze(1)).squeeze(1)
            z = self.config.loss.beta * torch.logsumexp(first_token_logits / self.config.loss.beta, dim=1)
            chosen_z = z[:batch_size]
            rejected_z = z[batch_size:]
            """
            
            #curr = torch.gather(concatenated_batch["concatenated_input_ids"], 1, (first_response_token_idx + 1).unsqueeze(1)).squeeze(1)
            #decoded = self.tokenizer.batch_decode(curr, skip_special_tokens=True)
            #print(decoded, "NEXT TOKEN")
            #print(batch["chosen_response_only"])
            
            """
            print()
            print("PROMPT TKOENS FROM FIRST RESPONSE IDNEX", gather_idxs.shape, first_response_token_idx.shape)
            print(first_response_token_idx)
            print(gather_idxs)
            print(concatenated_batch["concatenated_input_ids"].shape, all_logits.shape)
            print(self.tokenizer.batch_decode(concatenated_batch["concatenated_input_ids"], skip_special_tokens=True))
            print(concatenated_batch["concatenated_input_ids"].shape, " CONCAT vs BATCH logits -> ", all_logits.shape)
            curr = torch.gather(concatenated_batch["concatenated_input_ids"], 1, first_response_token_idx.unsqueeze(1)).squeeze(1)
            plus_1 = torch.gather(concatenated_batch["concatenated_input_ids"], 1, (first_response_token_idx + 1).unsqueeze(1)).squeeze(1)
            minus_1 = torch.gather(concatenated_batch["concatenated_input_ids"], 1, (first_response_token_idx - 1).unsqueeze(1)).squeeze(1)
            ## NOTE: MINUS 1 is the CORRECT ONE (curr = batch['prompt_len'])
            print(self.tokenizer.batch_decode(curr, skip_special_tokens=True), " CURR")
            print(self.tokenizer.batch_decode(plus_1, skip_special_tokens=True), " PLUS 1")
            print(self.tokenizer.batch_decode(minus_1, skip_special_tokens=True), " MINUS 1")
            print("PROMPT TKOENS FROM FIRST RESPONSE IDNEX")
            print()
            print(batch["prompt"], "|", sep="")
            print(batch["chosen_response_only"], "|", sep="")
            print("ACGTUAL RESPONSES")

            print("TOKEN OLOGITS")
            print(torch.sort(first_token_logits[0], descending=True)[:50])
            print("TOKEN OLOGITS")
            print()
            #beta * log \sum_{i=1}^K exp(1/beta * l_i)
            #where l_i is the logit of the i-th token
            #for the first token generation after the prompt
            print("Z VALUES", z.shape)
            print(z[0])
            print("Z VALUES", z.shape)
            """

        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch_size]
        rejected_logps = all_logps[batch_size:]

        if return_z:
            return chosen_logps, rejected_logps, z
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name == 'dpo':
            policy_chosen_logps, policy_rejected_logps, policy_z = self.concatenated_forward(self.policy, batch, return_z=True)
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
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)
            policy_z = all_gather_if_needed(policy_z, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/z'] = policy_z.cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics
    
    def get_rewards(self):
        """Gets rewards under the policy and reference model for eval set. Only works with BasicTrainer."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        results = []
        self.policy.eval()

        with tqdm.tqdm(desc="Computing rewards", total=self.config.n_eval_model_samples) as pbar:
            for eval_batch in self.eval_batches:
                if len(results) >= self.config.n_eval_model_samples * 2:
                    break
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)

                with torch.no_grad():
                    policy_chosen_logps, policy_rejected_logps, policy_z = self.concatenated_forward(
                        self.policy, local_eval_batch, return_z=True
                    )
                    reference_chosen_logps, reference_rejected_logps, reference_z = self.concatenated_forward(
                        self.reference_model, local_eval_batch, return_z=True
                    )

                    _, chosen_rewards, rejected_rewards = dpo_loss(
                        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
                        beta=self.config.loss.beta,
                        alpha=self.config.loss.alpha,
                        reference_free=self.config.loss.reference_free,
                        chosen_len=local_eval_batch["chosen_len"],
                        rejected_len=local_eval_batch["rejected_len"]
                    )
                    for i in range(self.config.eval_batch_size):
                        results.append({
                            "type": "chosen",
                            "policy_logps": policy_chosen_logps.cpu().numpy().tolist()[i],
                            "reference_logps": reference_chosen_logps.cpu().numpy().tolist()[i],
                            "policy_z": policy_z.cpu().numpy().tolist()[i],
                            "reference_z": reference_z.cpu().numpy().tolist()[i],
                            "rewards": chosen_rewards.cpu().numpy().tolist()[i],
                            "lengths": local_eval_batch["chosen_len_real"][i],
                            "completion": local_eval_batch["chosen_response_only"][i],
                            "prompt": local_eval_batch["prompt"][i]
                        })
                        results.append({
                            "type": "rejected",
                            "policy_logps": policy_rejected_logps.cpu().numpy().tolist()[i],
                            "reference_logps": reference_rejected_logps.cpu().numpy().tolist()[i],
                            "policy_z": policy_z.cpu().numpy().tolist()[i],
                            "reference_z": reference_z.cpu().numpy().tolist()[i],
                            "rewards": rejected_rewards.cpu().numpy().tolist()[i],
                            "lengths": local_eval_batch["rejected_len_real"][i],
                            "completion": local_eval_batch["rejected_response_only"][i],
                            "prompt": local_eval_batch["prompt"][i]
                        })
                        pbar.update(2)

        return pd.DataFrame(results)

    def sample(self, n_per=1, num_beams=None, repetition_penalty=1, top_k=50, penalty_alpha=0.0, temperature=1.0,
               top_p=1, no_repeat_ngram_size=0):
        """Samples from self.policy over the evaluation set."""
        if n_per != 1:
            print("warning: ignoring n_per sample argument")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        result = {}
        self.policy.eval()

        with tqdm.tqdm(desc="Sampling", total=self.config.n_eval_model_samples) as pbar:
            for eval_batch in self.eval_batches:
                if len(result) >= self.config.n_eval_model_samples:
                    break
            
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)

                samples, _ = self.get_batch_samples(local_eval_batch, use_reference=False, num_beams=num_beams, temperature=temperature,
                                                    repetition_penalty=repetition_penalty, top_k=top_k, penalty_alpha=penalty_alpha,
                                                    top_p=top_p, no_repeat_ngram_size=no_repeat_ngram_size)
                for prompt, sample in zip(local_eval_batch['prompt'], samples):
                    result[prompt] = sample
                    pbar.update(1)

        return result

    def train(self, example_counter_start=0, batch_counter_start=0):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        if self.config.optimizer_archive:
            self.load_optimizer_checkpoint(self.config.optimizer_archive)
            print('loaded optimzer from archive')
        if self.config.scheduler_archive:
            self.load_scheduler_checkpoint(self.config.scheduler_archive)
            print('loaded scheduler from archive')

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        if self.config.loss.name == 'dpo':
            self.reference_model.eval()

        self.example_counter = example_counter_start
        self.batch_counter = batch_counter_start
        last_log = None

        print(f"starting from example = {self.example_counter}, batch = {self.batch_counter}")

        if self.train_iterator is None:
            raise ValueError("No train iterator loaded, cannot train")

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name == 'dpo':
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
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
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
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
            #### END TRAINING ####

    def load_optimizer_checkpoint(self, optimizer_ckpt_path):
        state_dict = torch.load(optimizer_ckpt_path, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading optimizer at step {step} from {optimizer_ckpt_path} with metrics {json.dumps(metrics, indent=2)}')
        self.optimizer.load_state_dict(state_dict["state"])

    def load_scheduler_checkpoint(self, scheduler_ckpt_path):
        state_dict = torch.load(scheduler_ckpt_path, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading lr scheduler at step {step} from {scheduler_ckpt_path} with metrics {json.dumps(metrics, indent=2)}')
        self.scheduler.load_state_dict(state_dict["state"])

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
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None, only_policy: bool = False):
        """Save policy, optimizer, and scheduler state to disk."""
    
        try:
            n_examples = self.example_counter
        except:
            n_examples = 0

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(n_examples, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        if not only_policy:
            try:
                optimizer_state_dict = self.optimizer.state_dict()
                self.write_state_dict(n_examples, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
                del optimizer_state_dict

                scheduler_state_dict = self.scheduler.state_dict()
                self.write_state_dict(n_examples, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
            except:
                pass


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

    def load_optimizer_checkpoint(self, optimizer_ckpt_path):
        full_osd = None
        if self.rank == 0:
            full_osd = torch.load(optimizer_ckpt_path)["state"]
            print(f"[fsdp:MASTER] loaded optimizer checkpoint from {optimizer_ckpt_path}")
        sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, self.policy)
        print(f"successfully sharded optimizer for fsdp on local rank {self.rank}")

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
        
