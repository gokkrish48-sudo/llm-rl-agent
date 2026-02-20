"""
GRPO (Group Relative Policy Optimization) trainer for LLM reasoning.

GRPO eliminates the value network by computing advantages relative to
a group of sampled outputs for the same prompt. This is more stable
than PPO at LLM scale and was used in DeepSeek-R1.

Reference: DeepSeekMath (Shao et al., 2024)
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    model_name: str = "meta-llama/Llama-3-8B"
    learning_rate: float = 1e-6
    kl_coeff: float = 0.04               # KL divergence penalty weight
    group_size: int = 8                   # G: number of outputs sampled per prompt
    max_new_tokens: int = 512
    temperature: float = 0.9
    clip_epsilon: float = 0.2            # PPO-style clipping
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0


class GRPOTrainer:
    """
    GRPO trainer for LLM policy optimization.

    Key idea:
        For each prompt q, sample G outputs {o1, ..., oG} from the policy.
        Compute rewards {r1, ..., rG} from the reward model.
        Normalize advantages: A_i = (r_i - mean(r)) / std(r)
        Update policy to increase probability of high-advantage outputs.
        Add KL penalty to prevent policy from drifting too far from reference.
    """

    def __init__(self, config: GRPOConfig, reward_fn):
        self.config = config
        self.reward_fn = reward_fn

        logger.info(f"Loading policy model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Reference model (frozen) for KL penalty
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

    def sample_group(self, prompt: str) -> List[str]:
        """
        Sample G outputs from the current policy for a given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.policy.device)
        G = self.config.group_size

        with torch.no_grad():
            outputs = self.policy.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                num_return_sequences=G,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        completions = [
            self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            for out in outputs
        ]
        return completions

    def compute_log_probs(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for a batch of sequences."""
        with torch.no_grad() if model is self.ref_policy else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        # outputs.loss is mean NLL; we want per-token log probs
        logits = outputs.logits[:, :-1, :]  # (B, T-1, V)
        targets = labels[:, 1:]             # (B, T-1)

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        mask = (targets != -100).float()
        return (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp(min=1)

    def grpo_loss(
        self,
        prompt: str,
        completions: List[str],
        rewards: List[float],
    ) -> torch.Tensor:
        """
        Compute GRPO loss for a group of completions.

        Loss = -E[A_i * log π(o_i|q)] + β * KL(π || π_ref)

        where A_i = (r_i - mean(r)) / (std(r) + ε)  [group-normalized advantage]
        """
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # Group-normalized advantages
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        total_loss = torch.tensor(0.0, requires_grad=True)

        for completion, advantage in zip(completions, advantages.tolist()):
            full_text = prompt + completion
            enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.policy.device)

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            # Mask prompt tokens in labels (only train on completion)
            prompt_len = len(self.tokenizer(prompt)["input_ids"])
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100

            # Policy log prob
            log_prob_policy = self.compute_log_probs(
                self.policy, input_ids, attention_mask, labels
            )

            # Reference log prob (for KL penalty)
            with torch.no_grad():
                log_prob_ref = self.compute_log_probs(
                    self.ref_policy, input_ids, attention_mask, labels
                )

            kl = log_prob_policy - log_prob_ref  # per-sequence KL estimate
            policy_loss = -advantage * log_prob_policy
            loss = policy_loss + self.config.kl_coeff * kl
            total_loss = total_loss + loss

        return total_loss / len(completions)

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        One GRPO training step over a batch of prompts.
        """
        self.policy.train()
        total_loss = 0.0
        total_reward = 0.0

        for prompt in prompts:
            completions = self.sample_group(prompt)
            rewards = [self.reward_fn(prompt + c) for c in completions]

            loss = self.grpo_loss(prompt, completions, rewards)
            (loss / self.config.gradient_accumulation_steps).backward()

            total_loss += loss.item()
            total_reward += sum(rewards) / len(rewards)

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "loss": total_loss / len(prompts),
            "mean_reward": total_reward / len(prompts),
        }

    def save(self, path: str):
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")
