"""Evaluate and visualize a trained policy."""
import argparse
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.pick_cube import PickCubeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/best_model/best_model.zip", help="Path to model")
    parser.add_argument("--vec-normalize", type=str, default="runs/vec_normalize.pkl", help="Path to VecNormalize stats")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render with MuJoCo viewer")
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Train a model first with: python train.py")
        return

    model = SAC.load(model_path)
    print(f"Loaded model from {model_path}")

    # Create environment
    env = PickCubeEnv(render_mode=None)

    # Load normalization stats if available
    vec_normalize_path = Path(args.vec_normalize)
    if vec_normalize_path.exists():
        # Wrap the same env instance so viewer stays in sync
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        use_vec_env = True
        print(f"Loaded normalization stats from {vec_normalize_path}")
    else:
        use_vec_env = False
        print("No normalization stats found, using raw observations")

    # Run episodes
    successes = 0
    total_rewards = []

    for ep in range(args.episodes):
        if use_vec_env:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        done = False
        episode_reward = 0
        step = 0

        # Optional: launch viewer
        viewer = None
        if args.render:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            if use_vec_env:
                obs, reward, dones, infos = vec_env.step(action)
                done = dones[0]
                info = infos[0]
                reward = reward[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            episode_reward += reward
            step += 1

            # Update viewer with real-time pacing
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.02)  # ~50fps for visible motion

        if viewer is not None:
            viewer.close()

        total_rewards.append(episode_reward)
        if info.get("is_success", False):
            successes += 1

        print(f"Episode {ep + 1}: reward={episode_reward:.2f}, steps={step}, success={info.get('is_success', False)}")

    # Summary
    print(f"\n{'='*40}")
    print(f"Results over {args.episodes} episodes:")
    print(f"  Success rate: {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    print(f"  Mean reward: {sum(total_rewards)/len(total_rewards):.2f}")

    env.close()
    if use_vec_env:
        vec_env.close()


if __name__ == "__main__":
    main()
