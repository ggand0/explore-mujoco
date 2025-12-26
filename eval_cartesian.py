"""Evaluate a trained Cartesian lift policy with video recording."""
import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.lift_cube import LiftCubeCartesianEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip file")
    parser.add_argument("--normalize", type=str, default=None, help="Path to vec_normalize.pkl")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path. Defaults to eval.mp4 in model's run directory")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--curriculum-stage", type=int, default=0,
                        help="Curriculum stage (0=normal, 1=grasped lifted, 2=grasped table, 3=near cube)")
    parser.add_argument("--lock-wrist", action="store_true",
                        help="Lock wrist joints for stable top-down grasping")
    parser.add_argument("--action-scale", type=float, default=0.02,
                        help="Action scale for delta XYZ movements")
    args = parser.parse_args()

    # Determine output path - default to model's run directory
    if args.output is None:
        model_path = Path(args.model)
        # Model is typically in checkpoints/ subdirectory, so go up to run dir
        run_dir = model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent
        # Extract step count from model filename for video name
        step_str = ""
        for part in model_path.stem.split("_"):
            if part.isdigit():
                step_str = f"_{part}"
                break
        output_path = run_dir / f"eval{step_str}.mp4"
    else:
        output_path = Path(args.output)

    # Create env
    env = LiftCubeCartesianEnv(
        render_mode="rgb_array",
        max_episode_steps=200,
        curriculum_stage=args.curriculum_stage,
        lock_wrist=args.lock_wrist,
        action_scale=args.action_scale,
    )
    vec_env = DummyVecEnv([lambda: env])

    if args.normalize:
        vec_env = VecNormalize.load(args.normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded normalization from {args.normalize}")

    # Load model
    model = SAC.load(args.model)
    print(f"Loaded model from {args.model}")

    frames_closeup = []
    frames_wide = []
    frames_wide2 = []
    total_rewards = []
    successes = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        ep_reward = 0

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

            # Render all camera views
            frame_closeup = env.render(camera="closeup")
            frame_wide = env.render(camera="wide")
            frame_wide2 = env.render(camera="wide2")
            if frame_closeup is not None:
                frames_closeup.append(frame_closeup)
            if frame_wide is not None:
                frames_wide.append(frame_wide)
            if frame_wide2 is not None:
                frames_wide2.append(frame_wide2)

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

    # Save all videos
    if frames_closeup:
        closeup_path = output_path.with_stem(output_path.stem + "_closeup")
        imageio.mimsave(str(closeup_path), frames_closeup, fps=args.fps)
        print(f"\nSaved closeup video to {closeup_path}")
    if frames_wide:
        wide_path = output_path.with_stem(output_path.stem + "_wide")
        imageio.mimsave(str(wide_path), frames_wide, fps=args.fps)
        print(f"Saved wide video to {wide_path}")
    if frames_wide2:
        wide2_path = output_path.with_stem(output_path.stem + "_wide2")
        imageio.mimsave(str(wide2_path), frames_wide2, fps=args.fps)
        print(f"Saved wide2 video to {wide2_path}")

    # Save combined video (all 3 views horizontally concatenated)
    if frames_closeup and frames_wide and frames_wide2:
        frames_combined = [
            np.concatenate([c, w, w2], axis=1)
            for c, w, w2 in zip(frames_closeup, frames_wide, frames_wide2)
        ]
        combined_path = output_path.with_stem(output_path.stem + "_combined")
        imageio.mimsave(str(combined_path), frames_combined, fps=args.fps)
        print(f"Saved combined video to {combined_path}")

    print(f"\nSummary:")
    print(f"  Mean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  Success rate: {100 * np.mean(successes):.1f}%")

    env.close()


if __name__ == "__main__":
    main()
