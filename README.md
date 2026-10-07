# HW2: CNNs

## 1. Models

- A MLP
- A CNN
- A ManualConv2d implementation

The models classify the CIFAR-10 subset containing:

- Cat (3)
- Deer (4)
- Dog (5)

## 2. Accuracy

- MLP test acc: 61.07%
- CNN best test acc: 81.00%

The CNN achieves greater than the required 75% test accuracy, and the MLP achieves greater than the required 58% test accuracy.

During testing, the CNN uses the custom ManualConv2d implementation for one convolutional layer when `is_testing=True`.

## 3. CNN Ablation Results

| Configuration | Best Test Accuracy |
|---|---:|
| ReLU baseline | 79.07% |
| ReLU + Dropout 0.3 | 79.87% |
| ReLU + Dropout 0.3 + Random Crop/Flip | 80.47% |
| SiLU + Dropout 0.3, lr=3e-4 | 75.57% |
| SiLU + Dropout 0.3, lr=6e-4 | 78.33% |
| SiLU + Dropout 0.3, lr=1e-3 | 80.03% |
| SiLU + Dropout 0.3, lr=2e-3 | 79.50% |
| ReLU + Dropout 0.3 + Crop/Flip + BatchNorm | 79.57% |
| ReLU + Dropout 0.3 + Crop/Flip + Extra MaxPool | 80.57% |
| ReLU + Dropout 0.3 + Crop/Flip + Extra MaxPool + L2 | 79.87% |
| ReLU + Dropout 0.3 + Crop/Flip + Extra MaxPool + LR Decay | 81.00% |

The final CNN uses ReLU activations, dropout of 0.3, random crop and horizontal flip augmentation, an additional max pooling layer, and learning-rate decay from `3e-4` to `1e-4` after epoch 18.

## 4. Known Bugs

None known.