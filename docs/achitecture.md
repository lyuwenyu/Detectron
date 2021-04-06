
# Detector 

## Settings
- Default
    - images (tensor, shape=(n, c, h, w))
    - targets (tensor, shape=(n, 6)): [[im-idx, cls-id, x, y, w, h,], ...], where (x, y, w, h) belong to [0, 1], im-idx is image index in batch 
    - anchors (tensor, shape=[layer_num, anchor_num]): ((w, h), ...), ...), where layer_num is equal to features num for detection, (w, h) is value respecting to origin image size
    - strides (list, ): [8, 16, 32], where strides is in ascending order.
    - **covn** in new modules initialized with kaiming_init

- Dataset-Labling:
    - image: jpg
    - label: txt
        [cls, x, y, x, y],
    - 1 vs. 1
    
- Backbone
    - forward return feats is Tuple[feat, ...], if len(feats) > 1, in ascending order align with strides. 

- Encoder
    - processing features obtained from backbone
    - forward returns new feats: Tuple[Tensor] with format is same as above
    - FPNEncoder
        - output features after processing with (conv_**bn_relu**) 
        - bn relu problem after|before lateral and top_down features merging ? 

- Decoder
    - processing feats obtain from encoder
    - forward returns preds: 
        - Tuple[Tensor], len(preds) == feat_layers, tensor shape is [n c h w], where c == [4(x, y, w, h) + 1(obj) + num_cls] * num_anchor. 
        - Tuple[Tuple[Tensor]], 

- Target
    - training
        - build_target
            - yolo
                - v3
                - v5
        - compute_loss
            - losses
            
    - inference
        - returns 
            - [N * M * C], where C == 4(bbox) + 1(obj) + num_classes
        - postprocess
        - bboxes
        - obj
        - classes


## Examples

- yolov3  
```python 
# yolov3



```
