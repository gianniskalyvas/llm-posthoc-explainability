import torch

from rationalizers.explainers.base import BaseExplainer
from rationalizers.modules.sparsemap import seq_budget_smap


class SparseMAPExplainer(BaseExplainer):
    """
    The Generator takes an input text and returns samples from p(z|x)
    """

    def __init__(self, h_params: dict, enc_size):
        super().__init__()
        self.self_scorer = torch.nn.Linear(enc_size, 1)
        self.init = h_params.get('sparsemap_init', False)
        self.max_iter = h_params.get('sparsemap_max_iter', 100)
        self.transition = h_params.get('sparsemap_transition', 0)
        self.budget = h_params.get('sparsemap_budget', 0)
        self.temperature = h_params.get('sparsemap_temperature', 0.01)

    def forward(self, h, mask=None, **kwargs):
        batch_size, target_size, enc_size = h.shape
        lengths = mask.long().sum(1)

        # compute attention scores
        # [B, T, H] -> [B, T, 1]
        h1 = self.self_scorer(h)

        z = []
        num_states = 2

        for k in range(batch_size):
            scores = h1[k].view(-1)
            if isinstance(self.budget, (int, float)):
                budget = torch.round(self.budget / 100 * lengths[k])
            else:
                budget = torch.round(self.budget[k].item() / 100 * lengths[k])
            length = scores.shape[0]

            # Set unary scores for valid positions
            x = torch.cat(
                (
                    scores.unsqueeze(-1) / self.temperature,
                    torch.zeros((length, 1), device=scores.device),
                ),
                dim=-1,
            )
            # x[lengths[k]:, 0] = -1e12
            x[:, 0].masked_fill_(~mask[k], -1e12)

            # Set transition scores for valid positions
            # Create transition scores for the actual sequence length
            transition_scores = torch.full((length + 1,), float(self.transition), device=scores.device)
            transition = torch.zeros(
                (length + 1, num_states, num_states), device=scores.device
            )
            # Set transitions only for valid positions (up to sequence length)
            valid_length = min(lengths[k].item(), length)
            transition[:valid_length, 0, 0] = transition_scores[:valid_length] / self.temperature
            # SequenceBudget expects flattened transition matrix without first 2 and last 2 elements
            # Only one state in the beginning and in the end for start / stop symbol
            transition = transition.reshape(-1)[2:-2]

            # H:SeqBudget consists of a single factor so, in this particular case, the LP-SparseMAP solution is
            # indeed the SparseMAP solution and it can be found within a single iteration.
            self.max_iter = 1
            self.step_size = 0.0

            if self.training:
                z_probs = seq_budget_smap(
                    x,
                    transition,
                    budget=budget,
                    temperature=self.temperature,
                    init=self.init,
                    max_iter=self.max_iter,
                    step_size=self.step_size,
                )
            else:
                test_temperature = 1e-3
                z_probs = seq_budget_smap(
                    x / test_temperature,
                    transition / test_temperature,
                    budget=budget,
                    temperature=test_temperature,
                    init=self.init,
                    max_iter=self.max_iter,
                    step_size=self.step_size,
                    )

            z_probs.cuda()
            z.append(z_probs)

        z = torch.stack(z, dim=0).squeeze(-1)  # [B, T]
        z = z.to(mask.device)
        z = torch.where(mask, z, z.new_zeros([1]))
        self.z = z

        return z, None
