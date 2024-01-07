import torch
import copy

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        # Early stopping hyper-parameters
        self.patience = patience
        self.min_delta = min_delta

        # Best Performance
        self.best_loss = torch.inf
        self.best_epoch = 0
        self.best_model = None

        # Current Performance
        self.current_epoch = -1
        self.current_loss = None

        # Patience counter
        self.patience_counter = 0

    def __call__(self, epoch_loss: float, model: torch.nn.Module) -> bool:
        self.current_epoch += 1
        self.current_loss = epoch_loss

        # Check for improvement
        if (epoch_loss - self.min_delta) < self.best_loss:
            # Improvement found
            self.best_loss = epoch_loss
            self.best_epoch = self.current_epoch
            self.best_model = copy.deepcopy(model.state_dict())
            self.patience_counter = 0  # Reset patience counter
        else:
            # No improvement
            self.patience_counter += 1  # Increment patience counter

        # Check if patience exceeded
        if self.patience_counter >= self.patience:
            print(f"Early Stopping: Epoch {self.stopped_epoch} - loss: {self.current_loss:.4f} - best loss: {self.best_loss:.4f} - delta: {self.best_loss - self.current_loss:.4f}")
            return True
        else:
            return False