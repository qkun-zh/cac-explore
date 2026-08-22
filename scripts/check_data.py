"""数据集与加载器健全性检查：python scripts/check_data.py [root]"""
import sys

sys.path.insert(0, "code")
from data.fsc147 import FSC147Density

root = sys.argv[1] if len(sys.argv) > 1 else "/data/dataset/FSC147"
for split in ("train", "val", "test"):
    ds = FSC147Density(root, img_size=384, split=split)
    s = ds[0]
    assert s["imgs"].shape == (3, 384, 384), s["imgs"].shape
    assert s["density"].shape == (1, 384, 384), s["density"].shape
    diff = abs(float(s["counts"]) - float(s["density"].sum()))
    assert diff < 0.01, f"计数守恒失败 {diff}"
    print(f"{split}: n={len(ds)} 首样本count={float(s['counts']):.1f} imgs{tuple(s['imgs'].shape)} OK")
print("数据集检查全部通过")
