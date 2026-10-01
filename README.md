# Pet breed classification from scratch

A ResNet-18 trained from random initialisation on the [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset, reaching 82% test accuracy in 30 epochs using only the 3,680-image training split.

Uses a custom implementation of the [Muon](https://kellerjordan.github.io/posts/muon/) optimiser for the convolutional layers, with AdamW for the rest, instead of AdamW alone. This improved test accuracy by 18 percentage points over a tuned AdamW baseline (82% vs 64%). (AdamW essentially underfit every learning rate)

## Results

Accuracy of a few models on the 3,669-image test split. These averages were taken for models trained on seeds 1-5.

| Recipe                  | Mean test acc ± std | Mean train acc |
|-------------------------|---------------------|----------------|
| Muon / ResNet-18        | 82.25 ± 0.28%       | 99.98%         |
| AdamW / ResNet-18       | 64.01 ± 1.16%       | 85.39%         |
| AdamW / CNN             | 67.67 ± 0.72%       | 97.93%         |

## Method

**Model.** ResNet-18 implemented in `model.py`: basic residual blocks, widths 64-512, Kaiming initialisation.
**Optimiser.** `MuonWithAdamW` in `train.py`:
    - Muon for the convolutional weights: Nesterov momentum, then 5 Newton-Schulz iterations to orthogonalise each update (coefficients from Jordan et al.), scaled by 0.2·√max(m, n) so it can share a learning rate with AdamW.
    - AdamW for the classifier, BatchNorm parameters and biases.
    - Decoupled weight decay of 0.02 for both.
**Training.** Batch size 16; one-cycle schedule peaking at 6e-3 with 30% warm-up; label smoothing 0.1; gradient clipping at 5.
**Augmentation.** Random resized crops (scale 0.35-1), colour jitter, rotation up to 15°, and alternating horizontal flips where each image is mirroed on every other epoch rather than at random (Jordan, 2024).
**Evaluation.** Logits averaged over two centre crops, with images resized to 248 and 256 then cropped to 224.
**Hyperparameters.** Tuned on 15% of trainval; final model was retrained on 3,680 images.

## Reproducing

```
conda env create -f environment-gpu.yml   # or environment.yml without a GPU
conda activate muon-pets
python train.py   # trains and saves model.pth
python test.py    # [evaluates model.pth on the test split]
```
Set `Config.muon = False` in `train.py` to train with AdamW. Set `Config.seed` to set the seed.
