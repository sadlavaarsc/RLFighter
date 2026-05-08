# RLFighter Design Document

## Overview

RLFighter is a simplified top-down combat simulator designed for reinforcement learning research. It supports 1v1, 1vN, and NvN scenarios with three agent types: human (keyboard), scripted (rule-based FSM), and RL (custom PPO).

## Combat System

### Coordinate System

- Continuous 2D top-down arena: 20 × 20 units
- Y-axis points north (up in world coordinates, down in pygame screen coordinates)
- Agents are circles with radius 0.25
- Logic runs at 30 frames per second

### Actions

Each action has windup, active, and recovery phases measured in logic frames:

| Action | Windup | Active | Recovery | Damage | Impact | Hitbox | Friendly Fire |
|--------|--------|--------|----------|--------|--------|--------|---------------|
| Vertical Slash | 18 | 4 | 14 | 25 | 35 | Rect 0.4×1.6 | No |
| Horizontal Sweep | 10 | 8 | 12 | 18 | 25 | Sector 110° R=1.4 | Yes |
| Thrust | 8 | 6 | 10 | 20 | 30 | Rect 0.3×2.0 | No |
| Dodge | 0 | 10 | 8 | — | — | — | Invincible during active |
| Heal | 6 | 0 | 18 | — | — | — | Restores 50% HP at windup end |

- **Movement** is only possible during `IDLE` phase
- Actions entered during `IDLE` immediately start their windup
- During `RECOVERY` (last 4 frames only), a new action can be buffered and will execute immediately after recovery ends
- All animation phases are non-interruptible except by death or stagger

### Hit Detection

Agents are treated as point targets for hitbox intersection (sufficient given small agent radius relative to attack sizes):

- **Rectangle attacks** (vertical, thrust): oriented rectangle extending forward from the attacker's position
- **Sector attacks** (horizontal): circular sector centered on attacker with given radius and arc angle
- During `DODGE_INVINCIBLE`, agents are immune to all damage and impact

### Toughness and Stagger

- Each agent has a toughness bar (default 100)
- When hit, toughness decreases by the attack's `impact` value
- If toughness drops to 0 or below: agent enters `STAGGER` for 30 frames, toughness fully restores
- If toughness stays above 0 but impact > 0: short flinch (`STAGGER` for 5 frames) if currently `IDLE` or `RECOVERY`
- Toughness regenerates slowly when not taking damage: +0.3% of max per logic frame

### Heal

- 3 charges per life
- Each heal restores 50% of max HP
- Has 6-frame windup and 18-frame recovery
- Cannot heal if at full HP or out of charges

## Observation Space

Fixed-length vector per agent (84 dimensions):

### Self (20 dims)
- HP ratio, toughness ratio, heal charges ratio
- Action type one-hot (6)
- Phase one-hot (6)
- Phase progress [0, 1]
- Facing (cos, sin)
- Normalized position (x, y)

### K=4 Nearest Others (16 dims each)
- Mask (1 = valid agent)
- Is enemy
- Alive
- Relative position (dx, dy) normalized by arena size
- Distance normalized by arena diagonal
- Relative angle (cos, sin) from self-facing
- Action type one-hot (6)
- HP ratio, toughness ratio

Missing slots are zeroed with mask=0.

## Action Space

Two discrete categorical outputs:
- **Action type** (6): NOOP/move, vertical, horizontal, thrust, dodge, heal
- **Move direction** (9): 8 cardinal directions + stationary

During non-IDLE phases, the provided action is ignored (or buffered in late recovery).

## Reward Function

Per-agent reward:
- `+1.0` per HP damage dealt to enemies
- `-1.0` per HP damage taken
- `+50` per enemy kill
- `-100` on death
- `-0.001` per step (encourages decisive play)
- `+0.3 ×` mean teammate reward (soft team coordination)

## Training Pipeline

### Architecture

- Shared MLP trunk: 84 → 128 → 128 (ReLU)
- Action head: 128 → 6 (Categorical)
- Move head: 128 → 9 (Categorical)
- Value head: 128 → 1

### Algorithm

Vanilla PPO with:
- Clipped surrogate objective (ε = 0.2)
- GAE advantage estimation (γ = 0.99, λ = 0.95)
- Value loss coefficient: 0.5
- Entropy bonus coefficient: 0.01
- KL divergence early stopping at 0.03
- Gradient clipping: 0.5
- Adam optimizer, lr = 3e-4

### Curriculum

1. **1v1 vs Scripted**: Train RL agent (team 0) against hardcoded FSM opponent
2. **1v1 Self-Play**: All agents use shared policy; natural emergence of attack/defense patterns
3. **1vN / NvN**: Expand team sizes, inherit weights from earlier stages

### Vectorization

- Multiple arena instances run in parallel (default 4)
- Rollouts collected across all envs and all alive agents
- Parameter sharing: one policy network controls all agents, with team relationships encoded in observations
