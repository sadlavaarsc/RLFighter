import argparse

from rlfighter.rl.train import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", default="1,1", help="team sizes, e.g., 1,1 or 1,2")
    parser.add_argument("--opponent", default="scripted", choices=["scripted", "self_play"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-kl", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--logdir", default="runs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()

    team_sizes = [int(x) for x in args.teams.split(",")]

    trainer = Trainer(
        team_sizes=team_sizes,
        opponent=args.opponent,
        num_envs=args.num_envs,
        steps_per_update=args.steps_per_update,
        total_steps=args.total_steps,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_kl=args.max_kl,
        seed=args.seed,
        save_interval=args.save_interval,
        logdir=args.logdir,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
