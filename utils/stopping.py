from typing import Any, Optional


class StopCondition:
    """
    Stop condition for model optimization.
    """
    def __init__(
        self,
        growing_is_good: bool = True,
        patience: int = 15,
        min_improvement: int = 0,
        # abs_tol: float = None
        ignore_first: bool = False,
    ):
        """
        Initialize early stopping instance
        :param patience: how much to wait even with no improvement.
        :param min_improvement: minimum accountable metric improvement.
        :param growing_is_good: growing metric means improving the model.
        """
        self.patience: int = patience
        self.min_improvement: float = min_improvement
        self.growing_is_good: bool = growing_is_good
        self.ignore_next: bool = True if ignore_first else False
        # self.abs_tol: float = abs_tol
        self.counter: int = 0
        self.norm_best_metric: float = 0.0
        self.best_metric: float = None
        self.best_checkpoint: Any = None
        self.stop = False

    def update(self, metric: float, checkpoint: Any = None):
        """
        Update early stopping counter.
        :param metric: metric value.
        :checkpoint: anything that can identify the current parameters.
        """
        norm_metric = metric if self.growing_is_good else -metric
        if self.best_metric is None:
            if self.ignore_next:
                self.ignore_next = False
            else:
                self.norm_best_metric = norm_metric
                self.best_metric = metric
                self.best_checkpoint = checkpoint
        elif norm_metric <= self.norm_best_metric + self.min_improvement:
            self.counter += 1
            if self.counter > self.patience:
                self.stop = True
        else:
            self.counter = 0
            self.norm_best_metric = norm_metric
            self.best_metric = metric
            self.best_checkpoint = checkpoint
        return self.stop
