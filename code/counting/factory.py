import torch.nn as nn
from .backbone import DinoV3Backbone
from .prompt import PPPEPrompt
from .fusion import GatedFusion
from .head import SoftplusHead
from .criterion import MSECri, GLCri
from .post import SumPost

def build_backbone(cfg): return DinoV3Backbone(img_size=int(cfg.get("img_size",384)))
def build_prompt(cfg): return PPPEPrompt(dim=384)
def build_fusion(cfg): return GatedFusion(dim=384, hidden=int(cfg.get("adapter_dim",768)), dropout=float(cfg.get("dropout",0.1)))
def build_head(cfg): return SoftplusHead(in_dim=384)
def build_criterion(cfg):
    if str(cfg.get("loss", "mse")).lower() == "gl":
        return GLCri(blur=float(cfg.get("gl_blur", 0.01)), reach=float(cfg.get("gl_reach", 0.5)), scaling=float(cfg.get("gl_scaling", 0.5)), tau=float(cfg.get("gl_tau", 0.1)), cost=str(cfg.get("gl_cost", "per")))
    return MSECri()
def build_post(cfg): return SumPost()
