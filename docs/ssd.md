
- [ssd]()

- prior box
    - base area
    - area ratio, aspect ration
    - for every feature layer grid
    - num_anchors * h * w * 4
    - [cx, cy, w, h] belong to [0, 1] 
    - returns cat [num_anchors * h * w, 4(x, y, w, h)]

- match 
    - ious [N, M], n: number anchors, m: number targets
    - gt <-> prior, every gt has a best-matched prior. max(0)
    - prior box <-> gt_box: every prior box has a best-matched target. max(1)
    - and make sure every gt_box asigned to at least 1 best matched prior box
    - foreach j best_gt_overlap[best_prior_idx[j]] = j 

- target encoding
    - ( gt_xy - anchor_xy ) / anchor_wh
    - log( gt_wh / anchor_wh )

- loss
    - l1 loss for bbox
    - cross entropy for cls
    
- hard neg minning
    - softmax
        - logsumexp
    - sigmoid
        - -log p

