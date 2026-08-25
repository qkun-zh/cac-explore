from abc import ABC, abstractmethod
import os, torch

class Backbone(ABC, torch.nn.Module):
    """Abstraction: image -> patch tokens."""
    @abstractmethod
    def forward_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """return [B, M, C] patch tokens, M = (S/patch)^2."""
        ...

class DINOv3HFBackbone(Backbone):
    """HF DINOv3 ViT-S/16. Frozen by default; optionally unfreeze last N blocks at lr×mult."""
    def __init__(self, cfg):
        super().__init__()
        self.S = cfg.image_size
        self.patch = cfg.patch_size
        os.environ.setdefault("HF_HOME", "/data/asset/hf")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        token = None
        for p in ["/tmp/hf_token.txt", "/root/.cache/huggingface/token"]:
            if os.path.exists(p):
                token = open(p).read().strip(); break
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(cfg.hf_model, token=token, trust_remote_code=True)
        # freeze all first
        for p in self.net.parameters(): p.requires_grad_(False)
        # optionally unfreeze last N blocks + final norm
        unfreeze_n = int(getattr(cfg, "unfreeze_last_n_blocks", 0))
        backbone_lr_mult = float(getattr(cfg, "backbone_lr_mult", 0.1))
        if unfreeze_n > 0:
            blocks = self.net.model.layer
            total = len(blocks)
            for blk in blocks[total - unfreeze_n:]:
                for p in blk.parameters(): p.requires_grad_(True)
            if hasattr(self.net, "layernorm"):
                for p in self.net.layernorm.parameters(): p.requires_grad_(True)
            print(f"[DINOv3] unfroze last {unfreeze_n}/{total} blocks @lr×{backbone_lr_mult}")
        self.hidden = self.net.config.hidden_size
        self.num_reg = getattr(self.net.config, "num_register_tokens", 4)
        self.backbone_lr_mult = backbone_lr_mult

    def param_groups(self, base_lr, weight_decay):
        """Split params: unfrozen backbone blocks get lr×mult."""
        bb_params, head_params = [], []
        for name_, p in self.named_parameters():
            if not p.requires_grad: continue
            if name_.startswith("net.") and "encoder.layer" in name_:
                bb_params.append(p)
            else:
                head_params.append(p)
        groups = []
        if bb_params:
            groups.append({"params": bb_params, "lr": base_lr * self.backbone_lr_mult})
        if head_params:
            groups.append({"params": head_params, "lr": base_lr})
        return groups

    def forward_tokens(self, pixel_values):
        h = self.net(pixel_values=pixel_values).last_hidden_state  # [B, L, C]
        return h[:, 1 + self.num_reg :, :]  # drop cls+registers -> [B, M, C]
