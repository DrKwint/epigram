from flax import nnx
from tensorboardX import SummaryWriter  # or torch.utils.tensorboard
import os


class TensorboardLogger:
    def __init__(self, log_dir: str, run_name: str):
        # Clean up old run if exists to avoid mixing plots
        path = os.path.join(log_dir, run_name)
        idx = 0
        while os.path.exists(path):
            idx += 1
            path = os.path.join(log_dir, f"{run_name}_{idx}")
        if idx != 0:
            run_name = f"{run_name}_{idx}"

        self.writer = SummaryWriter(log_dir=path)
        print(f"Tensorboard writing to: {path}")

    def log_dict(self, metrics: dict[str, float], step: int):
        """Logs a dictionary of scalars to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, float(value), step)

    def close(self):
        self.writer.close()


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state = None  # To store the best NNX state

    def update(self, val_loss: float, model: nnx.Module):
        """
        Returns True if we should stop.
        Also snapshots the model state if it's the best so far.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Snapshot the BEST model parameters
            # In NNX, nnx.state(model) returns a lightweight State object (Pytree)
            self.best_state = nnx.state(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best(self, model: nnx.Module):
        """Reverts the model to the best state found during training."""
        if self.best_state is not None:
            print(f"Restoring best model with loss: {self.best_loss:.6f}")
            nnx.update(model, self.best_state)
        else:
            print("No best state to restore (did training run?).")
