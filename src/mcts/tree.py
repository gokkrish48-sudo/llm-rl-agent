"""
Monte Carlo Tree Search for LLM multi-step reasoning.
Uses UCB1 selection and Process Reward Model (PRM) for value estimation.
"""

import math
import random
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """
    A node in the MCTS reasoning tree.
    Each node represents a partial reasoning chain (sequence of steps).
    """
    state: str                          # reasoning chain so far
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0                  # cumulative PRM reward
    prior: float = 0.0                  # prior probability from policy
    step: str = ""                      # the reasoning step this node adds
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Mean value estimate."""
        return self.value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, exploration_constant: float = 1.4) -> float:
        """
        UCB1 score for node selection.
        Balances exploitation (q_value) and exploration (visit ratio).
        """
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return self.q_value + exploration

    def is_fully_expanded(self, max_children: int = 4) -> bool:
        return len(self.children) >= max_children or self.is_terminal

    def best_child(self, exploration_constant: float = 1.4) -> "MCTSNode":
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))

    def __repr__(self):
        return (f"MCTSNode(visits={self.visits}, q={self.q_value:.3f}, "
                f"step='{self.step[:40]}...')")


class MCTSReasoner:
    """
    MCTS-guided reasoning for LLMs.

    Algorithm:
        1. Selection  — traverse tree using UCB1 until unexpanded node
        2. Expansion  — sample new reasoning steps from LLM policy
        3. Simulation — roll out to completion, score with PRM
        4. Backprop   — update visit counts and values up the tree
    """

    def __init__(
        self,
        policy_fn: Callable[[str], List[str]],     # LLM: state → [candidate steps]
        reward_fn: Callable[[str], float],          # PRM: reasoning chain → score
        terminal_fn: Callable[[str], bool],         # is_terminal: state → bool
        num_simulations: int = 32,
        max_depth: int = 8,
        exploration_constant: float = 1.4,
        num_candidates: int = 4,                    # steps to sample per expansion
    ):
        self.policy_fn = policy_fn
        self.reward_fn = reward_fn
        self.terminal_fn = terminal_fn
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c = exploration_constant
        self.num_candidates = num_candidates

    def search(self, query: str) -> Dict[str, Any]:
        """
        Run MCTS from the query root and return the best reasoning chain.

        Returns:
            dict with keys: chain, steps, value, visits
        """
        root = MCTSNode(state=query)

        for sim in range(self.num_simulations):
            node = self._select(root)

            if not node.is_terminal and not node.is_fully_expanded(self.num_candidates):
                node = self._expand(node)

            value = self._simulate(node)
            self._backpropagate(node, value)

            if sim % 10 == 0:
                logger.debug(f"Sim {sim}: root visits={root.visits}, best_q={root.best_child(0).q_value:.3f}")

        best = self._best_path(root)
        return {
            "chain": best[-1].state if best else query,
            "steps": [n.step for n in best if n.step],
            "value": best[-1].q_value if best else 0.0,
            "visits": root.visits,
        }

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree using UCB1 until a node that can be expanded."""
        depth = 0
        while (node.children and
               node.is_fully_expanded(self.num_candidates) and
               not node.is_terminal and
               depth < self.max_depth):
            node = node.best_child(self.c)
            depth += 1
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Sample a new reasoning step from the policy and add as child."""
        candidate_steps = self.policy_fn(node.state)

        # Exclude steps already taken by existing children
        existing_steps = {c.step for c in node.children}
        new_steps = [s for s in candidate_steps if s not in existing_steps]

        if not new_steps:
            node.is_terminal = True
            return node

        step = random.choice(new_steps)
        new_state = node.state + "\n" + step
        is_terminal = self.terminal_fn(new_state)

        child = MCTSNode(
            state=new_state,
            parent=node,
            step=step,
            is_terminal=is_terminal,
        )
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """
        Score the current reasoning chain using the Process Reward Model.
        For terminal nodes, score directly. Otherwise do a light rollout.
        """
        if node.is_terminal or self.terminal_fn(node.state):
            return self.reward_fn(node.state)

        # Light rollout: extend greedily up to max_depth steps
        state = node.state
        depth = 0
        while not self.terminal_fn(state) and depth < 3:
            steps = self.policy_fn(state)
            if not steps:
                break
            state = state + "\n" + steps[0]
            depth += 1

        return self.reward_fn(state)

    def _backpropagate(self, node: MCTSNode, value: float):
        """Update visits and cumulative value from leaf to root."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def _best_path(self, root: MCTSNode) -> List[MCTSNode]:
        """Extract the path with highest Q-value from root to best leaf."""
        path = []
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.q_value)
            path.append(node)
        return path
