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

- MLP test acc 61%
- CNN test acc 78%

The CNN achieves greater than the required 75% test accuracy, and the MLP achieves greater than the required 58% test accuracy.

During testing, the CNN uses the custom ManualConv2d implementation for one convolutional layer when is_testing=True.

## 3. Known Bugs

None known.