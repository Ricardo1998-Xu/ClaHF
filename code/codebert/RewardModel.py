import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List


class RewardModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.scorer = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_token_index = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.size(0)
        last_hidden = output.last_hidden_state
        cls_output = last_hidden[torch.arange(batch_size), last_token_index]
        score = self.scorer(cls_output)
        return score.squeeze(-1)


def compute_combined_rank_loss(rank_rewards_list: List[List[torch.Tensor]], alpha: float = 0.0, margin: float = 0.0, device: str = 'cpu') -> torch.Tensor:

    top1_losses = []
    pairwise_losses = []
    for rewards in rank_rewards_list:
        # shape: [k]
        k = rewards.size(0)
        if k > 2:
            top1 = rewards[0]
            others = rewards[1:]

            # === Top1 Loss ===
            # top1 - others: shape [k - 1]
            top1_diffs = top1 - others
            top1_loss = -F.logsigmoid(top1_diffs - margin).mean()
            top1_losses.append(top1_loss)

            # === Pairwise Loss (i < j) ===
            # Create diff matrix: [k, k]
            diff_matrix = others.unsqueeze(1) - others.unsqueeze(0)
            # Mask upper triangle (i < j)
            mask = torch.triu(torch.ones_like(diff_matrix), diagonal=1)
            diffs = diff_matrix[mask == 1]  # shape [k*(k-1)/2]
            pairwise_loss = -F.logsigmoid(diffs - margin).mean()
            pairwise_losses.append(pairwise_loss)
        else:
            diff = rewards[0] - rewards[1]
            pairwise_loss = -F.logsigmoid(diff).mean()
            pairwise_losses.append(pairwise_loss)

    # === Aggregate final loss ===
    top1_loss = torch.stack(top1_losses).mean() if top1_losses else torch.tensor(0.0, device=device)
    pairwise_loss = torch.stack(pairwise_losses).mean() if pairwise_losses else torch.tensor(0.0, device=device)
    loss = alpha * top1_loss + (1 - alpha) * pairwise_loss
    return loss
