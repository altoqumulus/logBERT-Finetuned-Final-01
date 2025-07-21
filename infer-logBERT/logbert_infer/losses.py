import torch, torch.nn as nn

class MLKPLoss(nn.Module):                       # masked log-key prediction
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, labels):
        # logits: [B,L,V], labels: [B,L]
        return self.ce(logits.view(-1, logits.size(-1)),
                       labels.view(-1))

class VHMLoss(nn.Module):
    def __init__(self, embed_dim, alpha=0.1, ema_gamma=0.01):
        super().__init__()
        # ⇩ changes here
        self.register_buffer("center", torch.zeros(embed_dim))
        self.alpha   = alpha
        self.ema_g   = ema_gamma

    def forward(self, h_dist):
        # ensure both tensors are on same device automatically
        loss = ((h_dist - self.center) ** 2).mean()
        with torch.no_grad():
            batch_mean = h_dist.mean(dim=0)
            self.center.mul_(1 - self.ema_g).add_(batch_mean * self.ema_g)
        return self.alpha * loss

