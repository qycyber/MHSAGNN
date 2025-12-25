
import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional

class CosineAnnealingWarmupLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 max_epochs: int,
                 warmup_start_lr: float = 1e-6,
                 warmup_target_lr: Optional[float] = None,
                 eta_min: float = 1e-6,
                 warmup_type: str = 'linear',
                 last_epoch: int = -1,
                 verbose: bool = False):

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.warmup_type = warmup_type.lower()
        self.verbose = verbose

        if warmup_target_lr is None:
            if len(optimizer.param_groups) == 0:
                raise ValueError("Optimizer must have at least one param group")
            self.warmup_target_lr = optimizer.param_groups[0]['lr']
        else:
            self.warmup_target_lr = warmup_target_lr

        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.max_epochs <= self.warmup_epochs:
            raise ValueError("max_epochs must be greater than warmup_epochs")
        if self.warmup_start_lr <= 0 or self.warmup_target_lr <= 0 or self.eta_min <= 0:
            raise ValueError("All learning rates must be positive")
        if self.warmup_type not in ['linear', 'exponential', 'polynomial']:
            raise ValueError("warmup_type must be one of 'linear', 'exponential', 'polynomial'")

        self.current_phase = "warmup" if self.warmup_epochs > 0 else "annealing"
        self.total_lr_reductions = 0
        self.phase_transitions = []

        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def _get_warmup_lr(self, epoch: int) -> float:

        if self.warmup_epochs == 0:
            return self.warmup_target_lr

        progress = epoch / self.warmup_epochs
        progress = min(1.0, max(0.0, progress))

        if self.warmup_type == 'linear':
            lr = self.warmup_start_lr + (self.warmup_target_lr - self.warmup_start_lr) * progress
        elif self.warmup_type == 'exponential':

            lr = self.warmup_start_lr * (self.warmup_target_lr / self.warmup_start_lr) ** progress
        elif self.warmup_type == 'polynomial':

            lr = self.warmup_start_lr + (self.warmup_target_lr - self.warmup_start_lr) * (progress ** 2)
        else:
            raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

        return lr

    def _get_annealing_lr(self, epoch: int) -> float:

        annealing_epoch = epoch - self.warmup_epochs
        total_annealing_epochs = self.max_epochs - self.warmup_epochs

        if total_annealing_epochs <= 0:
            return self.eta_min

        cosine_factor = 0.5 * (1 + math.cos(math.pi * annealing_epoch / total_annealing_epochs))
        lr = self.eta_min + (self.warmup_target_lr - self.eta_min) * cosine_factor

        return lr

    def get_lr(self) -> List[float]:

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == -1:

            return [self.warmup_start_lr] * len(self.optimizer.param_groups)

        if self.last_epoch < self.warmup_epochs:

            lr = self._get_warmup_lr(self.last_epoch)
            if self.current_phase != "warmup":
                self.current_phase = "warmup"
                self.phase_transitions.append(('warmup_start', self.last_epoch, lr))
        else:

            lr = self._get_annealing_lr(self.last_epoch)
            if self.current_phase != "annealing":
                self.current_phase = "annealing"
                self.phase_transitions.append(('annealing_start', self.last_epoch, lr))

        return [lr] * len(self.optimizer.param_groups)

    def get_current_phase(self) -> str:

        return self.current_phase

    def get_phase_progress(self) -> float:

        if self.current_phase == "warmup":
            if self.warmup_epochs == 0:
                return 1.0
            return min(1.0, self.last_epoch / self.warmup_epochs)
        else:

            total_annealing_epochs = self.max_epochs - self.warmup_epochs
            if total_annealing_epochs <= 0:
                return 1.0
            current_annealing_epoch = max(0, self.last_epoch - self.warmup_epochs)
            return min(1.0, current_annealing_epoch / total_annealing_epochs)

    def get_scheduler_info(self) -> dict:

        current_lr = self.get_last_lr()[0] if self.last_epoch >= 0 else self.warmup_start_lr

        return {
            'current_epoch': self.last_epoch,
            'current_lr': current_lr,
            'current_phase': self.current_phase,
            'phase_progress': self.get_phase_progress(),
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'warmup_target_lr': self.warmup_target_lr,
            'eta_min': self.eta_min,
            'warmup_type': self.warmup_type,
            'phase_transitions': self.phase_transitions.copy(),
            'total_lr_reductions': self.total_lr_reductions
        }

    def _print_lr_update(self, is_warmup_end: bool = False, is_annealing_end: bool = False):

        current_lr = self.get_last_lr()[0]

        if is_warmup_end:
            print(f"CosineWarmupLR: Warm-up completed (epoch {self.last_epoch})")
            print(f"  Learning rate: {current_lr:.2e} → Starting cosine annealing phase")
        elif is_annealing_end:
            print(f"CosineWarmupLR: Training completed (epoch {self.last_epoch})")
            print(f"  Final learning rate: {current_lr:.2e}")
        elif self.verbose:
            phase_desc = f"{self.current_phase.capitalize()} phase"
            progress = self.get_phase_progress()
            print(f"CosineWarmupLR ({phase_desc}, {progress*100:.1f}% complete): lr = {current_lr:.2e}")

    def step(self, epoch: Optional[int] = None):

        prev_phase = self.current_phase

        super(CosineAnnealingWarmupLR, self).step(epoch)

        if prev_phase == "warmup" and self.current_phase == "annealing":
            self._print_lr_update(is_warmup_end=True)
        elif self.last_epoch >= self.max_epochs - 1:
            self._print_lr_update(is_annealing_end=True)
        elif self.verbose:
            self._print_lr_update()

class AdaptiveCosineAnnealingWarmupLR(CosineAnnealingWarmupLR):

    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 max_epochs: int,
                 warmup_start_lr: float = 1e-6,
                 warmup_target_lr: Optional[float] = None,
                 eta_min: float = 1e-6,
                 warmup_type: str = 'linear',
                 enable_restarts: bool = False,
                 restart_epochs: int = 0,
                 restart_factor: float = 1.0,
                 last_epoch: int = -1,
                 verbose: bool = False):

        super().__init__(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=warmup_start_lr,
            warmup_target_lr=warmup_target_lr,
            eta_min=eta_min,
            warmup_type=warmup_type,
            last_epoch=last_epoch,
            verbose=verbose
        )

        self.enable_restarts = enable_restarts
        self.restart_epochs = restart_epochs
        self.restart_factor = restart_factor
        self.restarts_completed = 0
        self.adaptive_eta_min = eta_min

        self.loss_history = []
        self.lr_adaptation_history = []

    def register_loss(self, loss: float):

        self.loss_history.append(loss)

        max_history = 100
        if len(self.loss_history) > max_history:
            self.loss_history = self.loss_history[-max_history:]

    def adapt_eta_min(self, recent_losses: List[float]):

        if len(recent_losses) < 10:
            return

        recent_avg = sum(recent_losses[-10:]) / 10
        earlier_avg = sum(recent_losses[-20:-10]) / 10 if len(recent_losses) >= 20 else recent_avg

        if recent_avg < earlier_avg * 0.95:

            self.adaptive_eta_min = min(self.eta_min * 1.5, self.warmup_target_lr * 0.01)
        elif recent_avg > earlier_avg * 1.05:
            self.adaptive_eta_min = max(self.eta_min * 0.5, self.warmup_start_lr)

    def get_adaptive_info(self) -> dict:

        base_info = super().get_scheduler_info()
        base_info.update({
            'enable_restarts': self.enable_restarts,
            'restart_epochs': self.restart_epochs,
            'restarts_completed': self.restarts_completed,
            'adaptive_eta_min': self.adaptive_eta_min,
            'loss_history_length': len(self.loss_history)
        })
        return base_info

def create_cosine_warmup_scheduler(optimizer,
                                 warmup_epochs: int = 7,
                                 max_epochs: int = 100,
                                 warmup_start_lr: float = 1e-6,
                                 warmup_target_lr: Optional[float] = None,
                                 eta_min: float = 1e-6,
                                 warmup_type: str = 'linear',
                                 adaptive: bool = False,
                                 **kwargs):

    if adaptive:
        return AdaptiveCosineAnnealingWarmupLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=warmup_start_lr,
            warmup_target_lr=warmup_target_lr,
            eta_min=eta_min,
            warmup_type=warmup_type,
            **kwargs
        )
    else:
        return CosineAnnealingWarmupLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            warmup_start_lr=warmup_start_lr,
            warmup_target_lr=warmup_target_lr,
            eta_min=eta_min,
            warmup_type=warmup_type
        )

if __name__ == "__main__":

    print("Cosine Annealing with Warm-up Scheduler")
    print("=" * 50)

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nTesting standard CosineAnnealingWarmupLR:")
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        warmup_epochs=7,
        max_epochs=50,
        warmup_start_lr=1e-6,
        warmup_target_lr=0.01,
        eta_min=1e-6,
        verbose=True
    )

    print(f"Warm-up: {scheduler.warmup_start_lr:.2e} → {scheduler.warmup_target_lr:.2e}")
    print(f"Annealing: {scheduler.warmup_target_lr:.2e} → {scheduler.eta_min:.2e}")

    lr_history = []
    for epoch in range(50):
        lr = scheduler.get_last_lr()[0]
        lr_history.append(lr)
        scheduler.step()

    print(f"\nLR range: {min(lr_history):.2e} → {max(lr_history):.2e}")
    print(f"Phase transitions: {scheduler.phase_transitions}")

    print("\nTesting AdaptiveCosineAnnealingWarmupLR:")
    adaptive_scheduler = AdaptiveCosineAnnealingWarmupLR(
        optimizer,
        warmup_epochs=7,
        max_epochs=50,
        enable_restarts=True,
        verbose=False
    )

    print("Scheduler implementation complete!")
