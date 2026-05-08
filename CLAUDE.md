# RLFighter — 项目上下文

## 概述

简化战斗逻辑的 RL 训练平台，支持 1v1/1vN/NvN。三种智能体：人工键盘、硬编码 FSM、自实现 PPO。

## 技术栈

- Python 3.12, PyTorch, pygame, numpy, tensorboard, pytest
- 无外部 RL 库（SB3/RLlib），PPO 自实现

## 项目结构

```
rlfighter/
  core/         # 战斗模拟引擎
    constants.py, action.py, agent.py, hitbox.py, combat.py, world.py, observation.py
  env/arena.py  # 多智能体环境（reset/step/reward/done）
  agents/       # 控制器
    base.py, human.py, scripted.py, rl.py
  rl/           # 自实现 PPO
    policy.py, rollout.py, ppo.py, train.py
  render/       # pygame 可视化
scripts/{play,train,eval}.py
tests/
checkpoints/    # .gitignore
runs/           # tensorboard logs, .gitignore
```

## 核心战斗系统

- 2D 连续 top-down, 20×20 竞技场, 30 逻辑帧/秒
- 5 种动作（已重新平衡）：
  - **突刺** (THRUST): 快/轻 — windup 6 / active 4 / recovery 8, dmg 10, impact 15, 范围 2.2×0.2
  - **横扫** (HORIZONTAL): 中/中 — windup 10 / active 6 / recovery 10, dmg 20, impact 25, 扇形 120° R=1.6, **友伤**
  - **竖劈** (VERTICAL): 慢/重 — windup 15 / active 4 / recovery 16, dmg 35, impact 40, 范围 1.6×0.4
  - **闪避** (DODGE): 0 / 10 / 8, 无敌帧
  - **回血** (HEAL): 3 次, 恢复 50% HP, windup 6 / recovery 18
- 状态机：IDLE → WINDUP → ACTIVE → RECOVERY；STAGGER 结束自动回到 IDLE
- **霸体与打断**：
  - 每个攻击动作有霸体等级（Poise）和打断等级（Impact）：**竖劈 3 ＞ 横扫 2 ＞ 突刺 1**
  - 攻击者 Impact ＞ 被击者 Poise → **打断**对方，强制进入 5 帧 STAGGER 并重置动作
  - 攻击者 Impact ≤ 被击者 Poise → 对方**霸体生效**，继续完成动作，仅承受 HP/韧性 伤害
  - 非攻击状态（IDLE / RECOVERY / HEAL 等，Poise=0）挨打仍进短 STAGGER（5 帧）
  - 韧性被击破（toughness ≤ 0）→ 无论霸体等级，强制进入 30 帧 STAGGER
- 韧性系统：impact 累计 → 击破进入 30 帧 STAGGER；未受击时缓慢恢复
- **单动作单命中**：同一攻击的 multi-frame ACTIVE 期间，每个目标最多受击一次（通过 `_hit_targets` 追踪，新动作开始时清空）
- 闪避：检查敌人攻击类型和实际范围，不再盲目闪避

## 动作空间（RL）

- action_type: 6 类 Categorical（NOOP/move, 竖劈, 横扫, 突刺, 闪避, 回血）
- move_dir: 9 类 Categorical（8 方向 + 静止）

## 观测空间

固定 84 维：自身 20 维 + K=4 最近他者 × 16 维（含 mask/team/pos/action/HP 等）

## 奖励

- 对敌伤害 +1/HP, 自身受伤 -1/HP
- 击杀 +50, 死亡 -100
- 步时惩罚 -0.001
- 队伍软奖励 +0.3 × 队友均值

## 硬编码 AI 设计（当前）

优先级式 FSM，per-agent 交互跟踪（HP 快照）避免卡死：
1. 安全回血（HP<40%, 无人在 2.5 内）
2. 范围感知闪避（敌人 WINDUP 且攻击实际能打到）
3. 突刺（≤2.2, 面向差<15°; 否则 IDLE 原地精确转向）
4. 横扫（≥2 敌人在 1.6 内, 扇形内无队友; 否则转向）
5. 竖劈（≤1.6, 面向差<30°; 否则转向）
6. 移动靠近（八向离散）

**鲁莽模式**：连续 90 帧无交互则跳过闪避，但攻击前仍会检查面向并精确转向。

## 训练进度

- ✅ 核心战斗模拟 + 24 项 pytest
- ✅ pygame 渲染 + 人工控制器
- ✅ 硬编码 AI（含鲁莽模式、范围感知闪避、连续精确转向）
- ✅ 观测/环境/PPO 框架
- ✅ 1v1 scripted 对手训练完成（100 updates, 204,800 steps）
- ✅ 数值重平衡（突刺快轻/横扫中中/竖劈慢重）
- ✅ 修复帧伤 bug（单动作单命中）+ STAGGER 结束 transition 修复
- 🔄 **下一步**：self-play 训练、1vN/NvN 扩展

## 关键文件

| 文件 | 说明 |
|---|---|
| `rlfighter/core/world.py` | 主模拟循环，动作状态机 |
| `rlfighter/core/combat.py` | 命中判定、伤害、韧性、击倒 |
| `rlfighter/agents/scripted.py` | 硬编码 AI，持续迭代中 |
| `rlfighter/env/arena.py` | 奖励计算、episode 终止 |
| `rlfighter/rl/train.py` | Trainer 类，支持 scripted/self_play 对手 |
| `scripts/play.py` | 本地对战，支持 human/scripted/rl |

## 常用命令

```bash
# 运行测试
uv run pytest tests/ -q

# 人工 vs 硬编码
uv run python -m scripts.play --p1 human --p2 scripted

# 训练 1v1 vs scripted
uv run python -m scripts.train --teams 1,1 --opponent scripted --total-steps 204800

# 训练 self-play
uv run python -m scripts.train --teams 1,1 --opponent self_play --total-steps 500000

# 评估
uv run python -m scripts.eval --checkpoint checkpoints/1v1_scripted/latest.pt --opponent scripted --episodes 100

# 加载 RL 对战
uv run python -m scripts.play --p1 rl --p2 scripted --checkpoint checkpoints/1v1_scripted/latest.pt

# tensorboard
uv run tensorboard --logdir runs/
```

## 已知注意事项

- `move_dir` 只有 9 个离散值，移动时 facing 精度受限；硬编码 AI 在 IDLE 攻击前通过直接修改 `me.facing` 实现连续精确转向
- 多 agent 共享 ScriptedController 实例时，`_hp_snapshots` 和 `_idle_frames` 用 `agent_id` 做 key 隔离
- GitHub push 偶尔因 SSL 超时失败，可稍后重试
