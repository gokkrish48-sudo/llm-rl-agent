import math
import random
from typing import List, Optional, Dict

class MCTSNode:
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None, action: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.reward = 0.0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def ucb1(self, exploration_weight: float = 1.41):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

class MCTSSearch:
    """Orchestrates the search process."""
    def __init__(self, policy_llm, prm_model):
        self.policy = policy_llm
        self.prm = prm_model

    def search(self, initial_state: str, iterations: int = 100):
        root = MCTSNode(state=initial_state)
        
        for _ in range(iterations):
            node = self._select(root)
            reward = self._simulate(node)
            self._backpropagate(node, reward)
            
        return self._best_action(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded():
            node = max(node.children, key=lambda c: c.ucb1())
        return self._expand(node)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        # Generate possible next reasoning steps from LLM
        actions = self.policy.generate_actions(node.state)
        for action in actions:
            new_state = node.state + "\n" + action
            child = MCTSNode(state=new_state, parent=node, action=action)
            node.children.append(child)
        return random.choice(node.children) if node.children else node

    def _simulate(self, node: MCTSNode) -> float:
        # Use PRM to score the current reasoning step
        return self.prm.score_step(node.state, node.action)

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _best_action(self, root: MCTSNode):
        return max(root.children, key=lambda c: c.visits).action
