"""Callback to auto-save learning curves during training."""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from plot_learning_curves import plot_single_run


class PlotLearningCurveCallback(BaseCallback):
    """Callback that plots learning curves at regular intervals.

    Plots are saved to the run directory whenever:
    - A checkpoint is saved (based on save_freq)
    - Training completes
    """

    def __init__(
        self,
        run_dir: str | Path,
        save_freq: int = 50000,
        verbose: int = 0,
        resume_step: int = 0,
    ):
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.save_freq = save_freq
        self._last_plot_step = resume_step

    def _on_step(self) -> bool:
        # Plot at save_freq intervals
        if self.num_timesteps - self._last_plot_step >= self.save_freq:
            self._plot()
            self._last_plot_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        """Plot final learning curves when training ends."""
        self._plot()

    def _plot(self) -> None:
        """Generate and save learning curve plot."""
        try:
            output_path = self.run_dir / "learning_curves.png"
            plot_single_run(self.run_dir, output_path)
            if self.verbose > 0:
                print(f"Saved learning curves to {output_path}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Failed to plot learning curves: {e}")
