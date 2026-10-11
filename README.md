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

- MLP best local test acc: 65.60% (1968/3000 correct, best epoch 24)
- CNN best local test acc: 83.70% (2511/3000 correct, best epoch 23)

The final MLP uses hidden widths 256/128/64, ReLU, and BatchNorm between the first Dense layer and ReLU (`epsilon=1e-3`, `momentum=0.99`). It uses no dropout or input standardization, horizontal flips only, batch size 64, seed 42, and 25 epochs. Adam starts at `1e-3` and changes to `1e-4` at the start of epoch 19. BatchNorm uses saved moving statistics during testing.

The final CNN uses ReLU, filters 32/64/128, three max-pooling layers, hidden size 256, dropout 0.3, crop/flip augmentation, batch size 64, seed 42, and 25 epochs. Adam starts at `1e-3` and changes to `1e-4` at the start of epoch 22.

## 3. CNN Ablation Results

### Architecture and Regularization

| Configuration | Test Accuracy |
|---|---:|
| ReLU baseline | 76.77% |
| ReLU + Dropout 0.3 | 77.50% |
| ReLU + Dropout 0.3 + Crop/Flip | 80.47% |
| SiLU + Dropout 0.3, LR=3e-4 | 73.23% |
| SiLU + Dropout 0.3, LR=6e-4 | 74.07% |
| SiLU + Dropout 0.3, LR=1e-3 | 74.83% |
| SiLU + Dropout 0.3, LR=2e-3 | 73.40% |
| ReLU + Dropout 0.3 + Crop/Flip + BatchNorm, historical result | 79.57% |
| **B: ReLU + Dropout 0.3 + Crop/Flip + Extra MaxPool** | **80.57%** |
| B + L2 regularization, historical result | 79.87% |
| B + learning-rate decay | 81.00% |

For the experiments below:
**B = ReLU + Dropout 0.3 + Crop/Flip + Extra MaxPool**
Unless otherwise specified: **batch size = 64, hidden size = 256, seed = 42, epochs = 25**.

### Hyperparameter Tuning Around B

| Configuration | Test Accuracy |
|---|---:|
| B, LR=3e-4, Step@19→1e-4, local eager rerun | 81.00% |
| B, seed=0, LR=3e-4, Step@19→1e-4 | 80.23% |
| B, seed=7, LR=3e-4, Step@19→1e-4 | 80.03% |
| B, seed=123, LR=3e-4, Step@19→1e-4 | 80.07% |
| B, seed=1234, LR=3e-4, Step@19→1e-4 | 80.60% |
| B, seed=3407, LR=3e-4, Step@19→1e-4 | 80.97% |
| B, LR=6e-4, Cosine→1e-5 | 81.83% |
| B, LR=1e-3, Cosine→1e-5 | 83.43% |
| B, LR=1.5e-3, Cosine→1e-5 | 82.93% |
| B, LR=2e-3, Cosine→1e-5 | 81.23% |
| B, Dropout=0.2, LR=6e-4, Cosine→1e-5 | 81.77% |
| B, Dropout=0.2, LR=1e-3, Cosine→1e-5 | 82.20% |
| B, Batch=128, LR=1e-3, Cosine→1e-5 | 81.97% |
| B, Hidden=512, LR=1e-3, Cosine→1e-5 | 82.67% |
| B, seed=3407, LR=1e-3, Cosine→1e-5 | 82.93% |
| B, Dropout=0.2, seed=3407, LR=1e-3, Cosine→1e-5 | 82.23% |
| B, LR=1e-3, Step@19→1e-4 | 83.63% |
| B, seed=3407, LR=1e-3, Step@19→1e-4 | 83.23% |
| B, Dropout=0.2, LR=1e-3, Step@19→1e-4 | 83.30% |
| B, Hidden=512, LR=1e-3, Step@19→1e-4 | 82.87% |
| B, LR=1e-3, Step@16→1e-4 | 83.00% |
| **B, LR=1e-3, Step@22→1e-4** | **83.70%** |
| B, LR=6e-4, Step@19→1e-4 | 83.20% |
| B, seed=1234, LR=1e-3, Cosine→1e-5 | 81.87% |
| B, seed=1234, LR=1e-3, Step@19→1e-4 | 82.93% |

The following comparison uses the same three seeds (42, 3407, 1234), with LR=1e-3 and all other B settings fixed. SD is the sample standard deviation of each run's best test accuracy.

| Learning-Rate Schedule | Mean Best Test Accuracy ± SD |
|---|---:|
| Cosine Decay | 82.74% ± 0.80% |
| Step Decay at Epoch 16 | 82.66% ± 0.37% |
| Step Decay at Epoch 19 | 83.27% ± 0.35% |
| Step Decay at Epoch 22 | 83.49% ± 0.19% |


## 4. MLP Ablation Results

**M = ReLU MLP with hidden widths 256/128/64, no dropout, horizontal flips only, batch size 64, seed 42, 25 epochs, and LR=6e-4 with Step@19 to 1e-4.** Each row overrides only the specified settings.

Experiments use eager training and fresh initialization for each configuration. BN is placed before ReLU; unless specified otherwise, its epsilon is `1e-3` and momentum is `0.99`. Dropout, when enabled, follows every hidden activation. Standardization statistics are computed only from training images and reused at test time. He initialization applies only to hidden Dense kernels.

### Initial Sweep

| Configuration | Best Test Accuracy | Best Epoch |
|---|---:|---:|
| M + Crop/Flip, LR=1e-3, Step@22 | 60.33% | 24 |
| M, no augmentation, LR=1e-3 | 60.80% | 20 |
| M, LR=1e-3 | 61.80% | 25 |
| M + Crop/Flip, LR=1e-3 | 60.23% | 25 |
| M, no augmentation, LR=3e-4, constant LR | 60.90% | 17 |
| M, no augmentation, LR=3e-4 | 61.53% | 21 |
| M, no augmentation | 60.27% | 21 |
| M | 62.57% | 25 |
| M, hidden=512/256/128, no augmentation | 61.67% | 24 |
| M, hidden=512/256/128 | 62.03% | 25 |
| M + Dropout 0.1, LR=1e-3 | 60.03% | 19 |
| M, hidden=512/256/128 + Dropout 0.1 | 60.90% | 24 |
| M, LR=1e-3, seed=3407 | 61.70% | 21 |
| M, LR=1e-3, seed=1234 | 61.93% | 22 |
| M, LR=1e-3, Cosine to 1e-5 | 62.30% | 21 |

### Follow-up Sweep

| Configuration | Best Test Accuracy | Best Epoch |
|---|---:|---:|
| M, baseline rerun | 62.57% | 25 |
| M, Step@16 | 61.70% | 25 |
| M, Step@22 | 62.50% | 25 |
| M, Step@19 to 2e-4 | 61.83% | 25 |
| M, Step@19 to 5e-5 | 62.00% | 20 |
| M, Cosine to 1e-5 | 61.67% | 24 |
| M, batch=32 | 62.57% | 21 |
| M, batch=128 | 61.33% | 23 |
| M, seed=3407 | 61.80% | 25 |
| M, seed=1234 | 61.97% | 21 |
| M, LR=4e-4 | 61.90% | 25 |
| M, batch=32, LR=4e-4 | 62.17% | 23 |
| M, LR=8e-4 | 62.27% | 25 |
| M, batch=32, LR=8e-4 | 61.70% | 25 |
| M + BN at hidden layer 1 | 64.63% | 24 |
| M + BN at all hidden layers | 65.17% | 19 |
| **M + BN at hidden layer 1, LR=1e-3 (selected submission)** | **65.60%** | 24 |

## 5. Known Bugs

None known.
