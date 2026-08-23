"""N0016_dino_seqcount — champion encoder verbatim -> SeqCount causal decoder over 14x14 integer count tokens."""
import math

import torch
import torch.nn as nn

BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"
PATCH = 14


class PromptEncoderV2(nn.Module):
    def __init__(self, freqs=8, hidden=256, out_dim=384):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(freqs) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(4 * freqs * 2 + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, bboxes, size):
        b = bboxes / float(size)
        w = (b[:, 2] - b[:, 0]).clamp_min(1e-4)
        h = (b[:, 3] - b[:, 1]).clamp_min(1e-4)
        cxywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, w, h], dim=1)
        ang = cxywh[..., None] * self.freqs
        fourier = torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)
        log_area = torch.log(w * h).unsqueeze(1).clamp(-13.8, 0.0)
        return self.mlp(torch.cat([fourier, log_area], dim=1))


class SeqCountDino(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch = 384
        dim = int(cfg.get("adapter_dim", 768))
        drop = float(cfg.get("dropout", 0.1))
        d = int(cfg.get("dec_dim", 256))
        self.backbone = timm.create_model(cfg.get("backbone", BACKBONE), pretrained=True,
                                          dynamic_img_size=True, features_only=True, out_indices=(6, 11))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.patch = PATCH
        self.t6_proj = nn.Linear(ch, ch)
        self.t11_proj = nn.Linear(ch, ch)
        self.layer_logits = nn.Parameter(torch.zeros(2))
        self.prompt_enc = PromptEncoderV2(out_dim=ch)
        self.adapter = nn.Sequential(nn.Linear(ch, dim), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(dim, ch))
        self.mem_proj = nn.Linear(ch, d)
        self.seq_len = int(cfg.get("seq_grid", 14)) ** 2
        self.vocab = int(cfg.get("seq_vocab", 64))
        self.tok_emb = nn.Embedding(self.vocab, d)
        self.pos_emb = nn.Embedding(self.seq_len + 1, d)
        self.start_emb = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerDecoderLayer(d_model=d, nhead=int(cfg.get("dec_heads", 4)),
                                           dim_feedforward=int(cfg.get("dec_ffn", 512)), dropout=drop,
                                           norm_first=True, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(cfg.get("dec_layers", 4)),
                                             norm=nn.LayerNorm(d))
        self.head = nn.Linear(d, self.vocab)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        with torch.no_grad():
            taps = self.backbone(imgs)
        ps = S // self.patch
        f6, f11 = taps[0].float(), taps[1].float()
        if f6.ndim == 3:
            f6 = f6.transpose(1, 2).reshape(f6.shape[0], f6.shape[2], ps, ps)
            f11 = f11.transpose(1, 2).reshape(f11.shape[0], f11.shape[2], ps, ps)
        gate = torch.softmax(self.layer_logits, dim=0)
        z6 = self.t6_proj(f6.flatten(2).transpose(1, 2))
        z11 = self.t11_proj(f11.flatten(2).transpose(1, 2))
        tokens = gate[0] * z6 + gate[1] * z11
        prompt = self.prompt_enc(bboxes, S)
        adapted = self.adapter(tokens)
        return self.mem_proj(adapted) + self.mem_proj(prompt)[:, None, :]

    @staticmethod
    def causal(l, device):
        return torch.triu(torch.full((l, l), float("-inf"), device=device), diagonal=1)

    def forward(self, imgs, bboxes, targets=None):
        mem = self.encode(imgs, bboxes)
        B, dev = mem.shape[0], mem.device
        if self.training and targets is not None:
            inp = torch.cat([self.start_emb.expand(B, -1, -1), self.tok_emb(targets[:, :-1])], dim=1)
            x = inp + self.pos_emb(torch.arange(inp.shape[1], device=dev))[None]
            h = self.decoder(x, mem, tgt_mask=self.causal(x.shape[1], dev))
            return {"logits": self.head(h)}
        xs = self.start_emb.expand(B, -1, -1)
        steps = []
        for i in range(self.seq_len):
            x = xs + self.pos_emb(torch.arange(xs.shape[1], device=dev))[None]
            h = self.decoder(x, mem, tgt_mask=self.causal(xs.shape[1], dev))
            step = self.head(h[:, -1])
            steps.append(step)
            if i < self.seq_len - 1:
                xs = torch.cat([xs, self.tok_emb(step.argmax(-1)).unsqueeze(1)], dim=1)
        return {"logits": torch.stack(steps, dim=1)}


def build_model(cfg):
    return SeqCountDino(cfg)
