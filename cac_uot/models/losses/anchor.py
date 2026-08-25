import torch
def box_mass_anchor(w, p, bboxes3, weight=1.0):
    """F1: each exemplar box should contain ~1 mass. p [B,M,2], w [B,M], bboxes3 [B,3,4]."""
    B = w.shape[0]
    loss = 0.0
    for b in range(w.shape[0]):
        for k in range(3):
            x1,y1,x2,y2 = bboxes3[b,k]
            mask = (p[b,:,0]>=x1)&(p[b,:,0]<=x2)&(p[b,:,1]>=y1)&(p[b,:,1]<=y2)
            mass = (w[b]*mask.float()).sum()
            loss = loss + (mass-1.0)**2
    return weight * loss / (B*3)
