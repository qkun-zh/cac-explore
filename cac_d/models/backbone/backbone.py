import os, torch
from abc import ABC, abstractmethod

class Backbone(ABC, torch.nn.Module):
    @abstractmethod
    def forward_feature_map(self, x): ...

class ConvNeXtBackbone(Backbone):
    def __init__(self, cfg):
        super().__init__()
        os.environ.setdefault("HF_HOME","/data/asset/hf")
        os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
        tok=None
        for p in ["/tmp/hf_token.txt","/root/.cache/huggingface/token"]:
            if os.path.exists(p): tok=open(p).read().strip(); break
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(cfg.hf_model, token=tok, trust_remote_code=True)
        for p in self.net.parameters(): p.requires_grad_(False)
        self.out_channels = self.net.config.hidden_sizes[2] if hasattr(self.net.config,"hidden_sizes") else 384

    def forward_feature_map(self, x):
        out = self.net(pixel_values=x, output_hidden_states=True)
        # stage3 is index 2
        return out.hidden_states[3] if len(out.hidden_states)>3 else out.last_hidden_state
