"""Train a lift policy using SAC with staged rewards.

Uses lift_cube.py environment with continuous lift gradient:
- Small lift reward without grasp (exploration signal)
- Stronger lift reward when grasping (correct behavior signal)
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from callbacks.plot_callback import PlotLearningCurveCallback
from envs.lift_cube import LiftCubeCartesianEnv


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict):
    return LiftCubeCartesianEnv(
        render_mode=None,
        max_episode_steps=env_cfg.get("max_episode_steps", 200),
        action_scale=env_cfg.get("action_scale", 0.02),
        lift_height=env_cfg.get("lift_height", 0.08),
        hold_steps=env_cfg.get("hold_steps", 10),
        reward_type=env_cfg.get("reward_type", "dense"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lift_500k.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override timesteps from config (useful for resuming)")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    env_cfg = config["env"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directory - always create new timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            raise ValueError(f"Resume directory not found: {resume_dir}")
        # Create new directory for resumed training
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / f"{timestamp}_resumed"
    else:
        resume_dir = None
        output_dir = Path(exp_cfg["base_dir"]) / exp_cfg["name"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to output directory
    shutil.copy(args.config, output_dir / "config.yaml")

    # If resuming, document what we're resuming from
    if args.resume:
        with open(output_dir / "RESUME_INFO.txt", "w") as f:
            f.write(f"Resumed from: {resume_dir}\n")
            f.write(f"Timestamp: {timestamp}\n")

    # Create environments
    env = DummyVecEnv([lambda: make_env(env_cfg)])

    # Load normalization stats from resume directory if resuming
    if args.resume:
        resume_vec_normalize_path = resume_dir / "vec_normalize.pkl"
        if resume_vec_normalize_path.exists():
            env = VecNormalize.load(resume_vec_normalize_path, env)
            env.training = True
            print(f"Loaded normalization stats from {resume_vec_normalize_path}")
        else:
            raise ValueError(f"vec_normalize.pkl not found in {resume_dir}")
    else:
        env = VecNormalize(
            env,
            norm_obs=env_cfg["normalize_obs"],
            norm_reward=env_cfg["normalize_reward"],
        )

    eval_env = DummyVecEnv([lambda: make_env(env_cfg)])
    if args.resume:
        # Load same normalization stats for eval (but with training=False)
        eval_env = VecNormalize.load(resume_vec_normalize_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=env_cfg["normalize_obs"],
            norm_reward=False,
            training=False,
        )

    # Create or load model
    resume_step = 0
    if args.resume:
        checkpoints = list((resume_dir / "checkpoints").glob("*.zip"))
        # Sort numerically by step number (not alphabetically!)
        def get_step_number(path):
            # Extract step number from filename like "sac_lift_500000_steps.zip"
            name = path.stem
            parts = name.split("_")
            for part in parts:
                if part.isdigit():
                    return int(part)
            return 0
        checkpoints = sorted(checkpoints, key=get_step_number)
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            resume_step = get_step_number(latest_checkpoint)
            model = SAC.load(latest_checkpoint, env=env, device=device)
            model.tensorboard_log = str(output_dir / "tensorboard")
            print(f"Resumed from {latest_checkpoint} (step {resume_step})")
            # Update RESUME_INFO with checkpoint details
            with open(output_dir / "RESUME_INFO.txt", "a") as f:
                f.write(f"Checkpoint: {latest_checkpoint}\n")
                f.write(f"Resume step: {resume_step}\n")
        else:
            raise ValueError(f"No checkpoints found in {resume_dir / 'checkpoints'}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_cfg["learning_rate"],
            buffer_size=sac_cfg["buffer_size"],
            learning_starts=sac_cfg["learning_starts"],
            batch_size=sac_cfg["batch_size"],
            tau=sac_cfg["tau"],
            gamma=sac_cfg["gamma"],
            train_freq=sac_cfg["train_freq"],
            gradient_steps=sac_cfg["gradient_steps"],
            verbose=1,
            seed=train_cfg["seed"],
            device=device,
            tensorboard_log=str(output_dir / "tensorboard"),
        )

    # Callbacks (created after model loading so resume_step is known)
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["save_freq"],
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_lift",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=train_cfg["eval_freq"],
        deterministic=True,
        render=False,
    )

    plot_callback = PlotLearningCurveCallback(
        run_dir=output_dir,
        save_freq=train_cfg["save_freq"],
        verbose=1,
        resume_step=resume_step,
    )

    # Use CLI timesteps if provided, otherwise use config
    timesteps = args.timesteps if args.timesteps is not None else train_cfg["timesteps"]

    # When resuming, we want to continue the timestep counter from where we left off
    # reset_num_timesteps=False tells SB3 to:
    # 1. NOT reset num_timesteps to 0
    # 2. ADD num_timesteps to total_timesteps internally (so we pass just the additional steps)
    if args.resume:
        # Model already has num_timesteps from checkpoint, verify it matches
        print(f"Loaded model num_timesteps: {model.num_timesteps}")
        reset_num_timesteps = False
        # Pass ONLY the additional steps - SB3 will add num_timesteps internally
        learn_timesteps = timesteps
        target_total = model.num_timesteps + timesteps
        print(f"\nResuming Lift training from step {model.num_timesteps}...")
        print(f"Training for {timesteps} additional timesteps (target: {target_total} total)")
    else:
        reset_num_timesteps = True
        learn_timesteps = timesteps
        target_total = timesteps
        print(f"\nStarting Lift training for {timesteps} timesteps...")

    print(f"Action space: delta XYZ + gripper (4 dims)")
    print(f"Output directory: {output_dir}")

    model.learn(
        total_timesteps=learn_timesteps,
        callback=[checkpoint_callback, eval_callback, plot_callback],
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(output_dir / "final_model")
    env.save(output_dir / "vec_normalize.pkl")

    print(f"\nTraining complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
