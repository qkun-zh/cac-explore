from abc import ABC, abstractmethod
import os, torch

class Backbone(ABC, torch.nn.Module):
    """Abstraction: image -> patch tokens."""
    @abstractmethod
    def forward_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """return [B, M, C] patch tokens, M = (S/patch)^2."""
        ...

class DINOv3HFBackbone(Backbone):
    """HF DINOv3 ViT-S/16, all frozen per current spec."""
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
        for p in self.net.parameters(): p.requires_grad_(False)
        self.hidden = self.net.config.hidden_size
        self.num_reg = getattr(self.net.config, "num_register_tokens", 4)

    def forward_tokens(self, pixel_values):
        h = self.net(pixel_values=pixel_values).last_hidden_state  # [B, L, C]
        return h[:, 1 + self.num_reg :, :]  # drop cls+registers -> [B, M, C]
