"""Precompute backbone features for FSC147.
Saves h2, h3, bboxes, points per sample → .pt files.
Backbone is frozen; this runs once and eliminates backbone from training loop.

Usage:
  python -m cac_d.precompute_features --split train
  python -m cac_d.precompute_features --split val
"""
import argparse, json, os, sys
from pathlib import Path

import torch
from PIL import Image as PILImage
from datasets import load_dataset, Image
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToTensor

REPO = "isentropic/FSC147"
REV = "refs/convert/parquet"
SPLIT_FILE = "Train_Test_Val_FSC_147.json"
ANNO_FILE = "annotation_FSC147_384.json"
DEFAULT_CACHE = "/data/cache/fsc147_features"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args()

    # -- HF env --
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", "/data/asset/hf")
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from cac_d.common import hf_token
    from cac_d.configs.config import Config

    tok = hf_token()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load model (backbone + exemplar encoder) --
    cfg = Config()
    from cac_d.models.backbone.backbone import ConvNeXtBackbone
    from cac_d.models.prompt.prompt import ExemplarEncoder
    backbone = ConvNeXtBackbone(cfg).to(device).eval()
    exemplar = ExemplarEncoder(
        backbone.out_channels[1], cfg.embed_dim, cfg.exemplar_layers, roi_size=cfg.roi_size
    ).to(device).eval()
    # Load trained exemplar weights if available
    ckpt_path = "/data/runs/cac_d_redesign/best_v21.pth"
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "exemplar" in {k.split(".")[0] for k in sd}:
            # extract exemplar.* keys
            ex_keys = {k: v for k, v in sd.items() if k.startswith("exemplar.")}
            if ex_keys:
                clean = {k.replace("exemplar.", "", 1): v for k, v in ex_keys.items()}
                exemplar.load_state_dict(clean, strict=False)
                print(f"Loaded exemplar weights from {ckpt_path}")
    exemplar.requires_grad_(False)

    # -- Load dataset --
    print(f"Loading {args.split} split...")
    ds = load_dataset(REPO, revision=REV, token=tok)["train"]
    raw = ds.cast_column("image", Image(decode=False)).data["image"]
    chunks = raw.chunks if hasattr(raw, "chunks") else [raw]
    paths = [os.path.basename(p) for c in chunks for p in c.field("path").to_pylist()]
    anno = json.load(open(hf_hub_download(REPO, ANNO_FILE, repo_type="dataset", token=tok)))
    ids = set(json.load(open(hf_hub_download(REPO, SPLIT_FILE, repo_type="dataset", token=tok)))[args.split])

    # -- Load HF processor --
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(cfg.hf_model, token=tok)

    # -- Prepare output dir --
    out_dir = Path(args.cache_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    to_tensor = ToTensor()

    # -- Precompute --
    count = 0
    skipped = 0
    for row, im_id in enumerate(paths):
        if im_id not in ids:
            continue
        ann = anno[im_id]
        sx, sy = args.image_size / float(ann["W"]), args.image_size / float(ann["H"])
        boxes = torch.tensor(
            [[min(x for x, _ in c) * sx, min(y for _, y in c) * sy,
              max(x for x, _ in c) * sx, max(y for _, y in c) * sy]
             for c in ann["box_examples_coordinates"][:3]], dtype=torch.float32).reshape(-1, 4)
        pts = torch.tensor([[x * sx, y * sy] for x, y in ann["points"]],
                           dtype=torch.float32).reshape(-1, 2)

        # Load and preprocess image
        img = ds[row]["image"]
        pixel = processor(images=img, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            h2, h3 = backbone.forward_feature_map(pixel)     # [1,192,48,48], [1,384,24,24]
            e = exemplar(h3, boxes.unsqueeze(0).to(device), args.image_size)  # [1,K,256]

        # Save as CPU tensors (float32 for precision)
        sample = {
            "h2": h2.squeeze(0).cpu(),          # [192,48,48]
            "h3": h3.squeeze(0).cpu(),          # [384,24,24]
            "e": e.squeeze(0).cpu(),            # [K,256]
            "bboxes": boxes,                     # [3,4]
            "points": pts,                       # [N,2]
        }
        torch.save(sample, out_dir / f"{count:05d}.pt")
        count += 1
        if count % 500 == 0:
            print(f"  {count} samples done...")

    print(f"Done: {count} samples saved to {out_dir} (skipped {skipped})")


if __name__ == "__main__":
    main()
