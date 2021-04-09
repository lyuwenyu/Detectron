
- [ssd]()

- prior box
    - num_anchors * h * w * 4
    - for every layer 
- match 
    - gt <-> prior  
    - prior box <-> gt_box: every prior box has a target
    - and make sure gt_box asigned to at least 0 prior box
- hard neg minning
    - softmax
        - logsumexp
    - sigmoid
        - -log p
- TODO
    - obj mask
