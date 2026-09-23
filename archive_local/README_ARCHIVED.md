# Archived / sealed (2026-09-22)

Training-free work is the active focus: **tree/N0029_tf_champion** (local + server).

## Sealed old tree
- `tree/N0001_champion` — **read-only sealed**, training-era champion tree.
- Checkpoints (`*.pth`) **deleted** (were ~3.3G on server).
- Metadata archive: `N0001_champion_local_20260922.tgz`
- Server metadata: `cac-server:/data/cac/archive/N0001_champion_meta_20260922.tgz`

## Removed local side projects (tars in this dir)
- `cac_backup_small_20260922.tgz` — old N0015 ckpt dir (pth dropped)
- `cac_explore_20260922.tgz`
- `cac_lightning_20260922.tgz`

## Server cleanup same day
- `/data/repro/{det_*,baseline_*,precision_*,innov_*}` pth removed
- `/data/.rootcache` pip cache removed (~3.2G)
- superseded `val_subset100/200` removed (locked: `val_subset286.json`)
- broken `convolutional_counting.py.{broken,bad}` removed
- `/data` used: 36G → **29G** (40% → 32%)

## Active only
- Local: `/home/qkun/cac/tree/N0029_tf_champion`
- Server: `/data/cac/tree/N0029_tf_champion`, `/data/cdino_run`, `/data/repro` (scripts+subset286)
