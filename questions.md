1) 21
2) Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Conv2d, Linear
3) 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 37
4) ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, None, ReLU, ReLU, ReLU, ReLU, None, ReLU, ReLU, ReLU, ReLU, None, ReLU, ReLU, None
5) 11195493
6) Cross-entropy (label smoothing 0.1)
7) Muon for the 20 convolution weights, AdamW for the classifier, BatchNorm and biases
8) [4.14e-04, 9.14e-04, 1.68e-03, 2.62e-03, 3.62e-03, 4.56e-03, 5.33e-03, 5.83e-03, 6.00e-03, 5.97e-03, 5.87e-03, 5.70e-03, 5.48e-03, 5.20e-03, 4.87e-03, 4.50e-03, 4.09e-03, 3.67e-03, 3.22e-03, 2.77e-03, 2.33e-03, 1.90e-03, 1.50e-03, 1.13e-03, 8.00e-04, 5.20e-04, 2.96e-04, 1.33e-04, 3.32e-05, 2.46e-08]
9) 30
10) 16
11) ToImage, RandomResizedCrop, ColorJitter, RandomRotation, ToDtype, Normalize, plus an alternating horizontal flip in the training loop
12) 3680
13) 0
14) 100.00%
15) 82.01%

Notes: the model is ResNet-18 (model.py). Layers 1-4 are in forward order: the 7x7
stem convolution, then each residual block's two 3x3 convolutions, with the 1x1
shortcut projection after the first block's pair in stages 2-4 (the 18 in
ResNet-18 excludes these three). A block's second ReLU is applied after its
shortcut is added. Testing averages the logits of two centre crops (resize 248
and 256, crop 224). Previous answers, for the submitted CNN, are at the
`submission` tag.
