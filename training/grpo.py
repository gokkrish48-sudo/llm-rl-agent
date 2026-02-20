import torch
import math
from typing import List, Optional

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def select_child(self, exploration_weight: float = 1.41):
        """Standard UCB1 selection."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            score = (child.value / (child.visits + 1e-6)) + \
                    exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

class GRPO_Optimizer:
    """Group Relative Policy Optimization implementation stub."""
    def __init__(self, model, lr=1e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def calculate_loss(self, log_probs, rewards):
        """
        Computes loss by comparing rewards within a group.
        rewards: shape (batch_size, group_size)
        """
        # Calculate mean/std within groups
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True) + 1e-8
        
        # Calculate advantages
        advantages = (rewards - mean_rewards) / std_rewards
        
        # Policy gradient loss
        loss = -(log_probs * advantages.detach()).mean()
        return loss

    def step(self, log_probs, rewards):
        self.optimizer.zero_grad()
        loss = self.calculate_loss(log_probs, rewards)
        loss.backward()
        self.optimizer.step()
        return loss.item()
