"""Combo search for F1/F3/F4 with early stopping. Runs each combo for up to 5 epochs, stops early if not promising."""
import os, sys, json, time, subprocess, signal

# combos to try (F1, F3, F4) — already tried: 0,0,0 (clean 36.5) and 1,0,0 (single F1 36.89 rho~0)
# remaining in priority order per root-cause: F4 alone, F1+F4, F3 alone, F1+F3, F4+F3, all three (already did, failed)
COMBOS = [
    (0,0,1),  # F4 only
    (1,0,1),  # F1+F4
    (0,1,0),  # F3 only
    (1,1,0),  # F1+F3
    (0,1,1),  # F3+F4
    (1,1,1),  # all three (already did, but re-run with lower anchor weight)
]

def patch_config(f1,f3,f4):
    p="/data/repo/cac_uot/configs/uot_config.py"
    t=open(p).read()
    # box_anchor_weight
    t=t.replace("box_anchor_weight: float = 1.0", f"box_anchor_weight: float = {1.0 if f1 else 0.0}")
    t=t.replace("box_anchor_weight: float = 0.0", f"box_anchor_weight: float = {1.0 if f1 else 0.0}")
    # loss_normalize
    t=t.replace('loss_normalize: str = "none"', f'loss_normalize: str = "{"demand_size" if f3 else "none"}"')
    t=t.replace('loss_normalize: str = "demand_size"', f'loss_normalize: str = "{"demand_size" if f3 else "none"}"')
    # gate
    t=t.replace("use_standardized_gate: bool = False", f"use_standardized_gate: bool = {str(bool(f4))}")
    t=t.replace("use_standardized_gate: bool = True", f"use_standardized_gate: bool = {str(bool(f4))}")
    open(p,"w").write(t)
    print(f"patched F1={f1} F3={f3} F4={f4}")

def run_one(combo, max_ep=5):
    f1,f3,f4 = combo
    patch_config(f1,f3,f4)
    # also need to scp to server? This script runs on server via ssh? For now assume local->server has been scp'd? We'll run via ssh from local?
    # Instead, this script is intended to run ON SERVER as a controller that launches training sub-processes
    import subprocess, time, json, os
    log=f"/tmp/cac_uot_combo_{f1}{f3}{f4}.log"
    hist="/tmp/cac_uot_hist.jsonl"
    if os.path.exists(hist): os.remove(hist)
    cmd=["/data/miniconda/envs/cac/bin/python","-u","/data/repo/cac_uot/train.py"]
    # train.py reads UOTConfig directly, no args needed, but we can pass --epochs
    # For combo search we want 5 epochs max
    env=dict(os.environ, HF_TOKEN=open("/tmp/hf_token.txt").read().strip(), HF_ENDPOINT="https://hf-mirror.com", HF_HOME="/data/asset/hf")
    # patch train.py to respect max_ep? For now just run full 40 but we will kill early via monitoring
    # Instead launch and monitor hist
    proc=subprocess.Popen(cmd, env=env, stdout=open(log,"w"), stderr=subprocess.STDOUT)
    print(f"started combo {combo} pid {proc.pid} log {log}")
    # monitor every 70s (per epoch)
    for ep in range(1, max_ep+1):
        time.sleep(80)  # ~1 epoch + val
        # check hist
        try:
            rows=[json.loads(l) for l in open("/tmp/god_hist.jsonl")]
            # Actually new trainer writes to /tmp/cac_uot_hist? Check trainer.py: it writes to /tmp/god_hist.jsonl? No, new trainer writes to ... let's check
            # For cac_uot, trainer writes to /tmp/cac_uot_hist? No, check trainer.py: it writes to ... actually it doesn't write hist yet, only prints
            # So we need to parse log for MAE
            import re
            txt=open(log).read()
            maes=re.findall(r"closed MAE=([0-9.]+)", txt)
            rhos=re.findall(r"rho=([0-9.\-]+)", txt)
            if maes:
                print(f" ep{ep} closed {maes[-1]} rho {rhos[-1] if rhos else '?'}")
                # early stop: if ep>=3 and mae>35 and rho<0.5
                if ep>=3 and float(maes[-1])>35 and float(rhos[-1])<0.5:
                    print(f"early stop combo {combo} at ep{ep} (mae {maes[-1]} rho {rhos[-1]})")
                    proc.terminate(); proc.wait(timeout=10)
                    return False
        except Exception as e:
            print(f"monitor err {e}")
        if proc.poll() is not None:
            print(f"combo {combo} finished early with code {proc.returncode}")
            break
    # let it run to max_ep then kill
    if proc.poll() is None:
        proc.terminate(); proc.wait(timeout=10)
    return True

if __name__=="__main__":
    for combo in COMBOS:
        print(f"\n=== COMBO {combo} ===")
        run_one(combo)
        time.sleep(5)
