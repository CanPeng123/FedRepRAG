import torch
import torch.nn.functional as F


@torch.no_grad()
def token_accuracy_and_supervised_ratio_causal(logits: torch.Tensor, labels: torch.Tensor,   ignore_index: int = -100) -> tuple[float, float]:
    """
    logits: [B, T, V], labels: [B, T]
    Returns:
      token_acc: next-token accuracy over supervised positions (labels != ignore_index),
                 aligned to causal LM shift (logits[t] -> labels[t+1]).
      sup_ratio: fraction of supervised label positions in the original labels tensor.
    """
    # supervised ratio on original labels
    sup = (labels != ignore_index).sum().float()
    tot = labels.numel()
    sup_ratio = float((sup / (tot + 1e-9)).item())

    # next-token alignment: logits at t predict labels at t+1
    pred_next = logits[:, :-1].argmax(dim=-1)  # [B, T-1]
    labels_next = labels[:, 1:]                # [B, T-1]

    mask = (labels_next != ignore_index)
    if mask.sum() == 0:
        return 0.0, sup_ratio

    correct = (pred_next == labels_next) & mask
    token_acc = float((correct.sum().float() / mask.sum().float()).item())

    return token_acc, sup_ratio


@torch.no_grad()
def median_l2_norm(x: torch.Tensor) -> float:
    # x: [..., D]
    n = x.norm(dim=-1)  # [...]
    return float(n.median().item())

@torch.no_grad()
def cosine_pos_neg_margin(e_a: torch.Tensor, e_b: torch.Tensor) -> tuple[float, float, float]:
    """
    e_a, e_b: [N, D]
    pos = cosine(e_a[i], e_b[i])
    neg = cosine(e_a[i], e_b[perm[i]])
    """
    ea = F.normalize(e_a, dim=-1)
    eb = F.normalize(e_b, dim=-1)
    cos_pos = (ea * eb).sum(dim=-1)  # [N]
    perm = torch.randperm(eb.size(0), device=eb.device)
    cos_neg = (ea * eb[perm]).sum(dim=-1)
    return float(cos_pos.mean().item()), float(cos_neg.mean().item()), float((cos_pos - cos_neg).mean().item())
