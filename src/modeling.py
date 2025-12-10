
import torch
import torch.nn as nn
import torch.nn.functional as F

def sampled_loss(final, y, emb, neg=32):
    pos = (final * emb[y]).sum(dim=1)
    V = emb.size(0); B = final.size(0)
    neg_idx = torch.randint(0, V, (B, neg), device=final.device)
    negW = emb[neg_idx]
    neg_scores = (negW * final.unsqueeze(1)).sum(dim=2)
    logits = torch.cat([pos.unsqueeze(1), neg_scores], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=final.device)
    return F.cross_entropy(logits, labels)
