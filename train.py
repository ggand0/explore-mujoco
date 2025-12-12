"""Train a pick-and-place policy using SAC."""
import argparse
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.pick_cube import PickCubeEnv


def make_env():
    """Create the environment."""
    return PickCubeEnv(render_mode=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=10000, help="Checkpoint save frequency")
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create environments
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_pick_cube",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )

    # Create SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    print(f"\nStarting training for {args.timesteps} timesteps...")
    print(f"Output directory: {output_dir}")

    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model and normalization stats
    model.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
