# N0021_dino_fullft — Full backbone fine-tuning with differential LR
Champion recipe verbatim but backbone UNFROZEN. Differential LR: backbone at 0.1x base.
H0030: IF backbone is fully fine-tuned THEN MAE <= 18 BECAUSE instance-boundary features emerge.
DISPROVED IF MAE > 21.53 (= no gain from unfreezing).
