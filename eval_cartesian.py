"""Evaluate a trained Cartesian lift policy with video recording."""
import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.lift_cube_cartesian import LiftCubeCartesianEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip file")
    parser.add_argument("--normalize", type=str, default=None, help="Path to vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=str, default="eval_cartesian.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Create env
    env = LiftCubeCartesianEnv(render_mode="rgb_array", max_episode_steps=200)
    vec_env = DummyVecEnv([lambda: env])

    if args.normalize:
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded normalization from {args.normalize}")

    # Load model
    model = SAC.load(args.model)
    print(f"Loaded model from {args.model}")

    frames = []
    total_rewards = []
    successes = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        ep_reward = 0
        ep_frames = []

        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]

            # Log gripper/contact info periodically
            if step % 50 == 0:
                i = info[0]
                print(f"  step={step}: gripper={i.get('gripper_state', 0):.3f}, "
                      f"dist={i.get('gripper_to_cube', 0):.3f}, "
                      f"cube_z={i.get('cube_z', 0):.3f}, "
                      f"grasp={i.get('is_grasping', False)}, "
                      f"contacts=({i.get('has_gripper_contact', False)}, {i.get('has_jaw_contact', False)})")

            # Render and collect frame
            frame = env.render()
            if frame is not None:
                ep_frames.append(frame)

            if done[0]:
                break

        # Get final info
        final_info = info[0]
        is_success = final_info.get("is_success", False)
        successes.append(is_success)
        total_rewards.append(ep_reward)

        print(f"Episode {ep + 1}: reward={ep_reward:.2f}, success={is_success}, "
              f"cube_z={final_info.get('cube_z', 0):.3f}, "
              f"gripper_to_cube={final_info.get('gripper_to_cube', 0):.3f}")

        frames.extend(ep_frames)

    # Save video
    if frames:
        imageio.mimsave(args.output, frames, fps=args.fps)
        print(f"\nSaved video to {args.output}")

    print(f"\nSummary:")
    print(f"  Mean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  Success rate: {100 * np.mean(successes):.1f}%")

    env.close()


if __name__ == "__main__":
    main()
