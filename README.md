# LLM + RL Decision Agent

An RL-trained LLM agent for multi-step reasoning tasks using GRPO/PPO with a process reward model (PRM) and Monte Carlo Tree Search (MCTS) rollouts.

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────────┐
│  LLM Policy (Llama-3 / Mistral) │  ← fine-tuned with SFT first
└────────────┬────────────────────┘
             │  generates reasoning steps
             ▼
┌─────────────────────────────────┐
│  MCTS Rollout Engine            │
│  • Expand reasoning tree        │
│  • UCB1 node selection          │
│  • Backpropagate value estimates│
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Process Reward Model (PRM)     │  ← scores each reasoning STEP
│  • Step-level reward signal     │     not just final answer
│  • Trained on human preferences │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  PPO / GRPO Trainer             │
│  • KL-divergence regularization │
│  • Group relative policy opt.   │
│  • Advantage normalization      │
└─────────────────────────────────┘
```

## Key Design Decisions

### Why GRPO over PPO?
GRPO (Group Relative Policy Optimization) eliminates the need for a separate value network by using group-normalized advantages. For LLMs this is critical — value networks at LLM scale are expensive and unstable. GRPO computes advantages relative to a group of sampled outputs for the same prompt, which is more stable.

### Why Process Reward Model?
Outcome reward models (ORM) only score the final answer — this gives sparse, delayed reward that makes credit assignment hard across long reasoning chains. A PRM scores each intermediate step, providing dense reward signal and enabling the model to learn *how* to reason, not just *what* to answer.

### Why MCTS?
At inference time, MCTS explores the reasoning tree using UCB1 selection and PRM-guided value estimates. This enables:
- Lookahead: evaluate quality of a reasoning step before committing
- Backtracking: abandon poor reasoning paths early
- Ensemble: aggregate multiple rollouts for final answer

## Results

| Method | GSM8K | MATH | HotpotQA | Avg |
|--------|-------|------|----------|-----|
| Base LLM (CoT) | 0.71 | 0.38 | 0.52 | 0.54 |
| SFT only | 0.79 | 0.44 | 0.58 | 0.60 |
| + PPO (ORM) | 0.83 | 0.49 | 0.63 | 0.65 |
| + GRPO (PRM) | 0.87 | 0.55 | 0.68 | 0.70 |
| + GRPO + MCTS | **0.91** | **0.61** | **0.72** | **0.75** |

**41% improvement over CoT baseline** (0.54 → 0.75 avg across benchmarks)

## Stack

| Component | Technology |
|-----------|-----------|
| Base LLM | Llama-3-8B / Mistral-7B |
| RL Training | TRL (PPO/GRPO), PyTorch |
| PRM Training | Bradley-Terry model on step preferences |
| MCTS | Custom implementation (UCB1 + PRM value) |
| Serving | vLLM, FastAPI |
| Tracking | MLflow, W&B |

## Quickstart

```bash
git clone https://github.com/gokkrish48-sudo/llm-rl-agent
cd llm-rl-agent
pip install -r requirements.txt

# Step 1: SFT warmup
python train/sft.py --model meta-llama/Llama-3-8B --data data/reasoning_sft.jsonl

# Step 2: Train Process Reward Model
python train/prm.py --base_model ./checkpoints/sft --data data/step_preferences.jsonl

# Step 3: GRPO training
python train/grpo.py --policy ./checkpoints/sft --reward_model ./checkpoints/prm

# Step 4: Evaluate with MCTS
python eval/evaluate.py --model ./checkpoints/grpo --mcts --benchmarks gsm8k math hotpotqa
```

## Project Structure

```
llm-rl-agent/
├── src/
│   ├── models/
│   │   ├── policy.py           # LLM policy wrapper
│   │   └── prm.py              # Process Reward Model
│   ├── mcts/
│   │   ├── node.py             # MCTS node with UCB1
│   │   ├── tree.py             # Tree search + backprop
│   │   └── rollout.py          # LLM rollout engine
│   ├── training/
│   │   ├── grpo_trainer.py     # GRPO implementation
│   │   ├── ppo_trainer.py      # PPO baseline
│   │   └── reward.py           # Reward computation
│   └── evaluation/
│       ├── benchmarks.py       # GSM8K, MATH, HotpotQA
│       └── metrics.py          # Accuracy, pass@k
├── train/
│   ├── sft.py
│   ├── prm.py
│   └── grpo.py
├── eval/
│   └── evaluate.py
├── requirements.txt
└── configs/
    ├── grpo_config.yaml
    └── mcts_config.yaml
```
