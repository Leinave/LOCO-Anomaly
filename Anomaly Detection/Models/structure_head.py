"""Structure-aware extensions for VCP-CLIP.

This module is deliberately independent of a particular CLIP repository.  It
expects dense patch features and a global CLIP feature, so it can be inserted
after the existing VCP-CLIP image encoder / Pre-VCP / Post-VCP code.

Inputs
------
global_feat: [B, D]
patch_feat:  [B, N, D]

The module exposes four interpretable signals:
  part_score      local part appearance anomaly
  count_score     component-count anomaly
  layout_score    spatial-layout anomaly
  relation_score  part-to-part relation anomaly

Only normal samples are required to fit the reference statistics.  Optional
normal/abnormal text embeddings can be supplied for CLIP semantic scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def corrupt_part_layout(part_tokens, centers):
    """
    保持部件特征不变，只交换部件的空间位置。
    用于构造逻辑布局异常。
    """

    B, C, D = part_tokens.shape
    device = part_tokens.device

    # 每张图像生成一个随机排列
    perm = torch.argsort(
        torch.rand(B, C, device=device),
        dim=1
    )

    batch_index = torch.arange(
        B,
        device=device
    ).unsqueeze(1)

    corrupted_tokens = part_tokens.clone()

    # 将原部件中心随机分配给其他部件
    corrupted_centers = centers[
        batch_index,
        perm
    ]

    return corrupted_tokens, corrupted_centers


def make_patch_grid(num_tokens: int, device: torch.device) -> torch.Tensor:
    """Return normalized [N, 2] (x, y) coordinates for a square patch grid."""
    side = int(num_tokens ** 0.5)
    if side * side != num_tokens:
        raise ValueError("patch tokens must form a square grid")
    y, x = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    xy = torch.stack((x, y), dim=-1).float().reshape(num_tokens, 2)
    return xy / max(side - 1, 1)


def cosine_text_score(
    feat: torch.Tensor,
    normal_text: torch.Tensor,
    abnormal_text: torch.Tensor,
) -> torch.Tensor:
    """CLIP-style abnormal-minus-normal score for [B,D] or [B,N,D]."""
    feat = F.normalize(feat, dim=-1)
    normal_text = F.normalize(normal_text, dim=-1)
    abnormal_text = F.normalize(abnormal_text, dim=-1)
    # Support text embeddings shaped [D] or [B,D] for dense features [B,N,D].
    if feat.dim() == 3 and normal_text.dim() == 2:
        normal_text = normal_text.unsqueeze(1)
        abnormal_text = abnormal_text.unsqueeze(1)
    normal = (feat * normal_text).sum(dim=-1)
    abnormal = (feat * abnormal_text).sum(dim=-1)
    return abnormal - normal


class PartAssignment(nn.Module):
    """Soft assignment of patch features to C pseudo-part prototypes."""

    def __init__(self, dim: int, num_parts: int, temperature: float = 0.07):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_parts, dim))
        self.temperature = temperature

    def forward(self, patch_feat: torch.Tensor) -> torch.Tensor:
        x = F.normalize(patch_feat, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        logits = torch.einsum("bnd,cd->bnc", x, p)
        return F.softmax(logits / self.temperature, dim=-1)


def build_part_representation(
    patch_feat: torch.Tensor,
    part_prob: torch.Tensor,
    patch_xy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build part tokens, centers, normalized counts, and raw counts."""
    eps = 1e-6
    weights = part_prob.transpose(1, 2)  # [B,C,N]
    raw_counts = part_prob.sum(dim=1)    # [B,C]
    part_tokens = torch.bmm(weights, patch_feat)
    part_tokens = part_tokens / (raw_counts.unsqueeze(-1) + eps)
    centers = torch.bmm(weights, patch_xy.expand(patch_feat.size(0), -1, -1))
    centers = centers / (raw_counts.unsqueeze(-1) + eps)
    counts = raw_counts / (raw_counts.sum(dim=1, keepdim=True) + eps)
    return part_tokens, centers, counts, raw_counts


class PartRelationEncoder(nn.Module):
    """Relation encoder over part tokens, conditioned on global context."""

    def __init__(self, dim: int, heads: int = 8, layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.position = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        part_tokens: torch.Tensor,
        centers: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = part_tokens + self.position(centers)
        if global_feat is not None:
            x = x + global_feat.unsqueeze(1)
        x = self.norm(self.encoder(x))
        return x, x.mean(dim=1)


@dataclass
class ReferenceStats:
    """Robust normal references fitted from normal training images."""

    relation_mean: Optional[torch.Tensor] = None
    relation_inv_cov: Optional[torch.Tensor] = None
    count_mean: Optional[torch.Tensor] = None
    count_std: Optional[torch.Tensor] = None
    layout_mean: Optional[torch.Tensor] = None
    layout_std: Optional[torch.Tensor] = None


def mahalanobis(x: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    d = x - mean
    return torch.einsum("bi,ij,bj->b", d, inv_cov, d).clamp_min(0).sqrt()


def fit_reference_stats(
    relation_features: torch.Tensor,
    counts: torch.Tensor,
    layouts: torch.Tensor,
    eps: float = 1e-4,
) -> ReferenceStats:
    """Fit normal references. Inputs are [M,D], [M,C], [M,L]."""
    rel_mean = relation_features.mean(0)
    rel_d = relation_features - rel_mean
    cov = rel_d.T @ rel_d / max(relation_features.size(0) - 1, 1)
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    rel_inv_cov = torch.linalg.pinv(cov + eps * eye)
    return ReferenceStats(
        relation_mean=rel_mean,
        relation_inv_cov=rel_inv_cov,
        count_mean=counts.mean(0),
        count_std=counts.std(0).clamp_min(eps),
        layout_mean=layouts.mean(0),
        layout_std=layouts.std(0).clamp_min(eps),
    )


class StructureAwareVCP(nn.Module):
    """Feature-level structural head for VCP-CLIP."""

    def __init__(
        self,
        dim: int,
        num_parts: int = 16,
        relation_heads: int = 8,
        relation_layers: int = 2,
        layout_grid: int = 4,
    ):
        super().__init__()
        self.layout_grid = layout_grid
        self.assignment = PartAssignment(dim, num_parts)
        self.relation = PartRelationEncoder(
            dim, heads=relation_heads, layers=relation_layers
        )

    def spatial_layout(self, part_prob: torch.Tensor, patch_xy: torch.Tensor) -> torch.Tensor:
        b, n, c = part_prob.shape
        g = self.layout_grid
        gx = (patch_xy[:, 0] * g).long().clamp(0, g - 1)
        gy = (patch_xy[:, 1] * g).long().clamp(0, g - 1)
        cell = gy * g + gx
        out = torch.zeros(b, g * g, c, device=part_prob.device, dtype=part_prob.dtype)
        out.scatter_add_(1, cell.view(1, n, 1).expand(b, -1, c), part_prob)
        return out.flatten(1)

    def forward(
        self,
        global_feat: torch.Tensor,
        patch_feat: torch.Tensor,
        stats: Optional[ReferenceStats] = None,
        text_normal_local: Optional[torch.Tensor] = None,
        text_abnormal_local: Optional[torch.Tensor] = None,
        text_normal_global: Optional[torch.Tensor] = None,
        text_abnormal_global: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, n, _ = patch_feat.shape
        xy = make_patch_grid(n, patch_feat.device)
        part_prob = self.assignment(patch_feat)
        part_tokens, centers, counts, raw_counts = build_part_representation(
            patch_feat, part_prob, xy
        )
        relation_tokens, relation_global = self.relation(
            part_tokens, centers, global_feat
        )

        # Local part appearance score: retain high-response patches.
        if text_normal_local is not None and text_abnormal_local is not None:
            patch_score = cosine_text_score(
                patch_feat, text_normal_local, text_abnormal_local
            )
            k = max(1, n // 20)
            part_score = patch_score.topk(k, dim=1).values.mean(1)
        else:
            patch_score = torch.zeros(b, n, device=patch_feat.device)
            part_score = patch_score.mean(1)

        layout = self.spatial_layout(part_prob, xy)
        count_score = torch.zeros(b, device=patch_feat.device)
        layout_score = torch.zeros(b, device=patch_feat.device)
        relation_score = torch.zeros(b, device=patch_feat.device)

        if stats is not None:
            if stats.count_mean is not None:
                count_score = ((counts - stats.count_mean) / stats.count_std).pow(2).mean(1).sqrt()
            if stats.layout_mean is not None:
                layout_score = ((layout - stats.layout_mean) / stats.layout_std).pow(2).mean(1).sqrt()
            if stats.relation_mean is not None:
                relation_score = mahalanobis(
                    relation_global, stats.relation_mean, stats.relation_inv_cov
                )

        # Project relation deviation back to pseudo-parts and then to patches.
        # This is an interpretable pseudo-part map rather than a pixel-accurate
        # segmentation mask; it shows which regions participate in the abnormal
        # global configuration.
        if stats is not None and stats.relation_mean is not None:
            part_relation_score = (
                relation_tokens - stats.relation_mean.view(1, 1, -1)
            ).pow(2).mean(dim=-1).sqrt()
            part_relation_score = part_relation_score / (
                part_relation_score.amax(dim=1, keepdim=True) + 1e-6
            )
            logic_patch_score = torch.einsum(
                "bnc,bc->bn", part_prob, part_relation_score
            )
        else:
            part_relation_score = torch.zeros(
                b, part_prob.size(-1), device=patch_feat.device
            )
            logic_patch_score = torch.zeros(
                b, n, device=patch_feat.device
            )

        result = {
            "part_score": part_score,
            "patch_score": patch_score,
            "count_score": count_score,
            "layout_score": layout_score,
            "relation_score": relation_score,
            "part_prob": part_prob,
            "part_tokens": part_tokens,
            "centers": centers,
            "counts": counts,
            "raw_counts": raw_counts,
            "layout": layout,
            "relation_tokens": relation_tokens,
            "relation_global": relation_global,
            "part_relation_score": part_relation_score,
            "logic_patch_score": logic_patch_score,
        }

        if text_normal_global is not None and text_abnormal_global is not None:
            result["semantic_score"] = cosine_text_score(
                global_feat, text_normal_global, text_abnormal_global
            )
        else:
            result["semantic_score"] = torch.zeros_like(part_score)

        # The global semantic score is intentionally low-weight: it is context,
        # not a direct logical-anomaly score.
        result["final_score"] = (
            0.30 * result["part_score"]
            + 0.20 * result["count_score"]
            + 0.20 * result["layout_score"]
            + 0.25 * result["relation_score"]
            + 0.05 * result["semantic_score"]
        )
        return result


def normal_relation_loss(
    relation_global: torch.Tensor,
    global_feat: torch.Tensor,
) -> torch.Tensor:
    """Normal-only consistency loss; detach CLIP to protect VCP alignment."""
    return 1.0 - F.cosine_similarity(
        relation_global, global_feat.detach(), dim=-1
    ).mean()


def logic_corruption_loss(
    clean_relation: torch.Tensor,
    corrupted_relation: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Margin loss for synthetic swap/delete/move logical corruptions."""
    clean_norm = clean_relation.pow(2).mean(dim=-1)
    corrupted_norm = corrupted_relation.pow(2).mean(dim=-1)
    return F.relu(margin + clean_norm - corrupted_norm).mean()
