import torch
import torch.nn.functional as F

class GRPOTrainer:
    """
    Implements Group Relative Policy Optimization (GRPO).
    Eliminates the need for a critic by using group-based relative rewards.
    """
    def __init__(self, model, optimizer, kl_coeff: float = 0.01):
        self.model = model
        self.optimizer = optimizer
        self.kl_coeff = kl_coeff

    def compute_loss(self, log_probs, ref_log_probs, rewards):
        """
        rewards: Tensor of shape (batch_size, group_size)
        """
        # Calculate group relative advantages
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - mean_r) / std_r # Shape: (batch_size, group_size)

        # Policy Ratio
        ratio = torch.exp(log_probs - log_probs.detach()) 
        
        # Surrogates
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL Divergence from reference model (regularization)
        kl_loss = F.kl_div(log_probs, ref_log_probs, reduction='batchmean', log_target=True)
        
        return policy_loss + self.kl_coeff * kl_loss

    def train_step(self, states, actions, rewards):
        self.optimizer.zero_grad()
        # simplified forward pass logic
        log_probs = self.model(states, actions)
        ref_log_probs = self.model.get_ref_log_probs(states, actions)
        
        loss = self.compute_loss(log_probs, ref_log_probs, rewards)
        loss.backward()
        self.optimizer.step()
        return loss.item()
