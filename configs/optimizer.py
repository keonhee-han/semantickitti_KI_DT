"""
Defines an AdamOptimizer class that inherits from torch.optim.Adam.
Initializes the class with standard Adam optimizer hyperparameters.
Optionally, you can add a method load_config to load hyperparameters like learning rate from a configuration file (implementation details omitted here).
"""

import torch


class AdamOptimizer(torch.optim.Adam):
  """Wrapper class for Adam optimizer with custom configuration loading."""

  def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    """
    Args:
      params (iterable): An iterable of parameters to optimize or dicts defining parameter groups.
      lr (float, optional): Learning rate. Defaults to 0.001.
      betas (Tuple[float, float], optional): Coefficients used for moving average calculations. Defaults to (0.9, 0.999).
      eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
      weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.
    """
    super(AdamOptimizer, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

  def load_config(self, config):
    """Loads hyperparameters from a configuration object."""
    # Assuming the configuration object has attributes for optimizer hyperparameters
    self.lr = config.optimizer.learning_rate
    self.betas = (config.optimizer.beta1, config.optimizer.beta2)
    self.eps = config.optimizer.epsilon
    self.weight_decay = config.optimizer.weight_decay


# # Example usage (assuming config object is available)
# optimizer = AdamOptimizer(model.parameters())
# optimizer.load_config(config)