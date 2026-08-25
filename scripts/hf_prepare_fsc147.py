"""Prepare isentropic/FSC147 via real HF datasets stack.
Fast path: official auto-converted parquet (single file) instead of 6146 loose files.
Downloads: train parquet + Train_Test_Val_FSC_147.json + annotation_FSC147_384.json (via hf_hub_download).
Cache lands in HF_HOME=/data/asset/hf. Prints column/path info for the trainer to bind ids.
"""
import os, json, sys

os.environ.setdefault("HF_HOME", "/data/asset/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download

def main():
    token = None
    for p in ["/tmp/hf_token.txt", "/root/.cache/huggingface/token"]:
        if os.path.exists(p):
            token = open(p).read().strip(); break
    print("[prepare] token:", (token or "NONE")[:8], flush=True)

    from datasets import load_dataset
    ds = None
    # Fast path 1: parquet convert branch
    try:
        ds = load_dataset("isentropic/FSC147", revision="refs/convert/parquet", token=token)
        print("[prepare] loaded via refs/convert/parquet:", ds, flush=True)
    except Exception as e:
        print("[prepare] parquet branch failed:", e, flush=True)
    # Fast path 2: explicit parquet api url
    if ds is None:
        try:
            data_files = {"train": "https://huggingface.co/api/datasets/isentropic/FSC147/parquet/default/train/0.parquet"}
            ds = load_dataset("parquet", data_files=data_files, token=token)
            print("[prepare] loaded via parquet api url:", ds, flush=True)
        except Exception as e:
            print("[prepare] parquet url failed:", e, flush=True)
    # Slow fallback: native imagefolder (many files)
    if ds is None:
        ds = load_dataset("isentropic/FSC147", token=token)
        print("[prepare] loaded native imagefolder:", ds, flush=True)

    split_name = list(ds.keys())[0]
    ex = ds[split_name][0]
    print("[prepare] columns:", ds[split_name].column_names, flush=True)
    for k, v in ex.items():
        desc = type(v).__name__
        if hasattr(v, "size"):
            desc += f" size={v.size}"
        if isinstance(v, dict):
            desc += f" keys={list(v.keys())} path={v.get('path')}"
        if isinstance(v, str):
            desc += f" val={v[:80]}"
        print(f"[prepare] col {k}: {desc}", flush=True)

    # side-car JSONs
    p_splits = hf_hub_download("isentropic/FSC147", "Train_Test_Val_FSC_147.json", token=token)
    p_anno = hf_hub_download("isentropic/FSC147", "annotation_FSC147_384.json", token=token)
    sp = json.load(open(p_splits)); an = json.load(open(p_anno))
    print(f"[prepare] splits: {[{k: len(v)} for k, v in sp.items()]}", flush=True)
    some = list(an.keys())[0]
    print(f"[prepare] anno[{some}] keys={list(an[some].keys())}", flush=True)
    json.dump({"splits_json": p_splits, "anno_json": p_anno,
               "split_name": split_name, "columns": ds[split_name].column_names},
              open("/tmp/fsc147_hf_meta.json", "w"))
    print("[prepare] META_WRITTEN", flush=True)

if __name__ == "__main__":
    main()
