# RLFighter

Simple combat-logic RL playground for 1vN / NvN agents, trained with a custom PPO implementation in PyTorch.

## Features

- **Combat simulation**: 2D top-down continuous arena with hitboxes, toughness/stagger system, and windup/recovery frames.
- **Actions**: vertical slash, horizontal sweep (friendly fire), thrust, dodge, and limited heals.
- **Three agent types**:
  - **Human**: keyboard control for testing and feeling the combat.
  - **Scripted**: rule-based FSM mimicking classic game AI.
  - **RL**: trained with a from-scratch PPO + GAE + self-play pipeline.
- **Curriculum**: 1v1 → 1vN → NvN, with shared-policy multi-agent training.
- **Visualization**: real-time pygame rendering with HP/toughness bars, hitboxes, and action phase indicators.

## Quick start

Install dependencies (preferably in a virtual environment):
```bash
pip install -e ".[dev]"
```

### Human vs Scripted
```bash
python -m scripts.play --p1 human --p2 scripted
```
Controls (Player 1):
- `WASD` — move
- `J` — vertical slash
- `K` — horizontal sweep
- `L` — thrust
- `Space` — dodge
- `H` — heal (3 uses, 50% HP each)

### Train RL agent (1v1 vs scripted)
```bash
python -m scripts.train --teams 1,1 --opponent scripted --total-steps 1000000
```

### Evaluate
```bash
python -m scripts.eval --checkpoint checkpoints/latest.pt --opponent scripted --episodes 100
```

View training metrics:
```bash
tensorboard --logdir runs/
```

## Project structure

```
rlfighter/
  core/        # simulation engine (actions, agents, hitboxes, combat, world, observations)
  env/         # multi-agent arena environment (reset/step/rewards)
  agents/      # human, scripted, and RL controllers
  rl/          # custom PPO training framework
  render/      # pygame visualization
scripts/
  play.py      # local gameplay with any agent combination
  train.py     # training entry point
  eval.py      # evaluation entry point
tests/         # pytest unit tests
```

## License

MIT
