import torch, torch.nn as nn

class UncertaintyWeighting(nn.Module):
    """Kendall & Gal 2018: learnable loss balancing.
       L_total = Σ_i exp(-s_i) * L_i + 0.5 * s_i,  s_i = log σ_i² (learnable).
       s_i large → weight small, but pays log penalty; auto-balances scales.
    """
    def __init__(self, num_terms: int, init: float = 0.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.full((num_terms,), float(init)))

    def forward(self, losses):
        # losses: list of scalar tensors
        w = torch.exp(-self.log_var)
        total = sum(w[i] * losses[i] + 0.5 * self.log_var[i] for i in range(len(losses)))
        # for logging: effective weights
        weights = {f"w{i}": w[i].item() for i in range(len(losses))}
        return total, weights
