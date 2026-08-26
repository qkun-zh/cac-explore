"""MFU Queue for visual prompt augmentation (VQCounter Alg.1 adapted).

Stores exemplar embeddings per class, evicts Most Frequently Used,
samples with P(v) ∝ 1/(freq+eps) favoring rare prompts.
"""
import random
import torch


class MFUQueue:
    def __init__(self, capacity=32, dim=256, eps=1e-6, device="cuda"):
        self.E = capacity
        self.dim = dim
        self.eps = eps
        self.device = device
        # class_key -> list[Tensor[D]]
        self.queues = {}
        self.freqs = {}

    def _ensure(self, k):
        if k not in self.queues:
            self.queues[k] = []
            self.freqs[k] = []

    def enqueue(self, class_ids, embs):
        """class_ids: list[str] len B, embs: [B,K,D]"""
        if isinstance(class_ids, torch.Tensor):
            class_ids = [str(int(x)) for x in class_ids]
        B, K, D = embs.shape
        for b, cid in enumerate(class_ids):
            self._ensure(cid)
            q = self.queues[cid]
            f = self.freqs[cid]
            for k in range(K):
                v = embs[b, k].detach().clone()
                if len(q) >= self.E:
                    idx = max(range(len(f)), key=lambda i: f[i])
                    q.pop(idx); f.pop(idx)
                q.append(v)
                f.append(0)

    def sample(self, class_ids, m, device=None):
        """Sample m vectors per sample. Returns [B,m,D]. Empty queues -> zeros."""
        if isinstance(class_ids, torch.Tensor):
            class_ids = [str(int(x)) for x in class_ids]
        B = len(class_ids)
        dev = device or self.device
        # infer device from any queue entry if available
        for cid in class_ids:
            if cid in self.queues and len(self.queues[cid]) > 0:
                dev = self.queues[cid][0].device
                break
        out = []
        for cid in class_ids:
            self._ensure(cid)
            q = self.queues[cid]
            f = self.freqs[cid]
            if len(q) == 0:
                out.append(torch.zeros(m, self.dim, device=dev))
                continue
            probs = torch.tensor([1.0 / (fi + self.eps) for fi in f], dtype=torch.float32, device=dev)
            probs = probs / probs.sum()
            k = min(m, len(q))
            idx = torch.multinomial(probs, k, replacement=False)
            for i in idx:
                f[int(i)] += 1
            sampled = torch.stack([q[int(i)] for i in idx])
            if k < m:
                pad = torch.zeros(m - k, self.dim, device=dev)
                sampled = torch.cat([sampled, pad], dim=0)
            out.append(sampled)
        return torch.stack(out, dim=0)  # [B,m,D]

    def __len__(self):
        return sum(len(v) for v in self.queues.values())
