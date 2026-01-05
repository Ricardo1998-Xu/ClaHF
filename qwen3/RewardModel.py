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
        # 使用最后一个非PAD token的 hidden state
        last_token_index = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.size(0)
        last_hidden = output.last_hidden_state
        cls_output = last_hidden[torch.arange(batch_size), last_token_index]
        score = self.scorer(cls_output)
        return score.squeeze(-1)


def compute_rank_list_loss(rank_rewards_list: List[List[torch.tensor]], device='cpu') -> torch.Tensor:
    """
    通过给定的有序（从高到低）的ranklist的reward列表，计算rank loss。
    所有排序高的句子的得分减去排序低的句子的得分差的总和，并取负。

    Args:
        rank_rewards_list (torch.tensor): 有序（从高到低）排序句子的reward列表，e.g. ->
                                        [
                                            [torch.tensor([0.3588]), torch.tensor([0.2481]), ...],
                                            [torch.tensor([0.5343]), torch.tensor([0.2442]), ...],
                                            ...
                                        ]
        device (str): 使用设备

    Returns:
        loss (torch.tensor): tensor([0.4891], grad_fn=<DivBackward0>)
    """
    loss = torch.tensor(0.0, device=device)
    add_count = 0

    for rewards in rank_rewards_list:
        # 向量化所有 i < j 的 reward 差值
        diff_matrix = rewards.unsqueeze(1) - rewards.unsqueeze(0)  # shape: [k, k]
        mask = torch.triu(torch.ones_like(diff_matrix), diagonal=1)  # 上三角掩码
        diffs = diff_matrix[mask == 1]  # shape: [k * (k - 1) / 2]

        loss += F.logsigmoid(diffs).sum()
        add_count += diffs.numel()

    return -loss / add_count  # 要最大化分差，所以要取负数


def compute_combined_rank_loss(rank_rewards_list: List[List[torch.Tensor]], alpha: float = 0.0, margin: float = 0.0, device: str = 'cpu') -> torch.Tensor:
    """
    组合排序损失函数：
    - Top1强化项：强调 top-1 与其他候选的区分。
    - Pairwise Margin项：为所有 pair 引入 margin。
    二者比例由 alpha 控制。

    Args:
        rank_rewards_list: 每条样本的奖励值（已按偏好排序，从好到差）。
        alpha: 控制 Top1 强调与 Margin 损失的加权比例，范围 [0, 1]。
        margin: Margin值，设为0即为logsigmoid，无明确间隔要求。
        device: 计算设备。

    Returns:
        最终损失值。
    """

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
