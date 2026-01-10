# Micro-PyTorch : Moteur d'Autodifferenciation

## Description

Implementation from scratch d'un moteur d'autodifferenciation en Python avec NumPy. Ce projet reproduit les mecanismes fondamentaux de PyTorch pour comprendre le deep learning de l'interieur.

## Fonctionnalites

- Differentiation automatique (Reverse Mode)
- Classe Tensor avec graphe de calcul dynamique
- Couches : Linear, Conv2d, MaxPool2d, BatchNorm1d, Dropout
- Activations : ReLU, Tanh, Sigmoid, Softmax
- Optimiseurs : SGD, Adam, AdamW, RMSProp, Adagrad
- Fonctions de perte : CrossEntropyLoss, MSELoss
- Learning Rate Schedulers

## Structure du projet

```
prog_diff/
├── autodiff_engine.py          # Moteur d'autodifferenciation
├── micro_pytorch_project.ipynb # Notebook avec exemples et MNIST
├── requirements.txt            # Dependances
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```python
from autodiff_engine import Tensor, Linear, Adam

# Creation d'un tenseur
x = Tensor([1.0, 2.0, 3.0])

# Forward pass
y = x ** 2
loss = y.sum()

# Backward pass
loss.backward()
print(x.grad)  # Gradients calcules automatiquement
```

## Contributeurs

- Wassim Badraoui
- Alexandra Da Silva
