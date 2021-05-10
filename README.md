# Low-Illumination-Detector
- Object Detection in the Dark (YoLo, SSD-based)
- Robustness for Low-illumination Environments


### TODO
- [x] Set the Dataset and DataLoader (LOL dataset, COCO, ExDark)
- [x] Split the Backbone into two-parts (robust-extractor / detector)
- [x] Patch Selection Strategy from CUT (Contrastive Unpaired Translation) 
- [x] Generate the Training Dataset (processing coco-d and ex-dark dataset)
- [x] Set the SSD and YoLo Detector
- [x] Training Schemes for 1) Learn Robustness for Extractor, 2) Learn to Detect
- [ ] Load the pre-trained Weights (Backbone, SSD-detector)



### Architecture






### Train Options
- Train Type
    - robust: pre-training only the base network for illumination robustness 
    - coco-d: pre-training SSD network with COCO-d dataset
    - ex-d: training SSD network with ex-dark dataset
    - robust_ex-d: training backbone with "robust" and SSD with "ex-d"
    - robust_coco-d_ex-d: training backbone with "robust" and SSD with "coco-d" and "ex-d"
    - robust_coco-d: training backbone with "robust" and SSD with "coco-d"
    - coco-d_ex-d: training SSD network with "coco-d" and "ex-d" type
    
    
- Pre-trained Option
    - backbone_path : "None" or "backbone-dir" 
    - coco_path : "None" or "coco-dir"



### Test




### Inference




### Reference




