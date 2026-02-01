import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List

class RewardModel(nn.Module):
    def __init__(self, encoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.scorer = nn.Linear(config.hidden_size, 1)
        self.args = args

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        # decoder_hidden_states[-1]: [batch_size, seq_len, hidden_size]
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]

        score = self.scorer(vec)  # shape: [B, 1]
        return score.squeeze(-1)  # shape: [B]


def compute_rank_list_loss(rank_rewards_list: List[List[torch.tensor]], device='cpu') -> torch.Tensor:

    loss = torch.tensor(0.0, device=device)
    add_count = 0

    for rewards in rank_rewards_list:
        diff_matrix = rewards.unsqueeze(1) - rewards.unsqueeze(0)  # shape: [k, k]
        mask = torch.triu(torch.ones_like(diff_matrix), diagonal=1)
        diffs = diff_matrix[mask == 1]  # shape: [k * (k - 1) / 2]

        loss += F.logsigmoid(diffs).sum()
        add_count += diffs.numel()

    return -loss / add_count


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
            pairwise_loss = -F.logsigmoid(diff - margin).mean()
            pairwise_losses.append(pairwise_loss)

    # === Aggregate final loss ===
    top1_loss = torch.stack(top1_losses).mean() if top1_losses else torch.tensor(0.0, device=device)
    pairwise_loss = torch.stack(pairwise_losses).mean() if pairwise_losses else torch.tensor(0.0, device=device)
    loss = alpha * top1_loss + (1 - alpha) * pairwise_loss
    return loss
