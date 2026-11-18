# NeuralCore

**NeuralCore** is a compact research laboratory for studying deep, fully connected
(multilayer perceptron, MLP) neural networks. Its numerical core is implemented
from scratch with **Python and NumPy**: forward propagation, backpropagation,
losses, optimizers, mini-batch training, metrics, checkpointing, and gradient
checks do not rely on a deep-learning framework.

It is intended for learning and controlled experiments, not as a replacement
for mature production frameworks.

## What is included

- Arbitrarily deep dense networks with configurable hidden widths.
- ReLU, tanh, sigmoid, and linear activations.
- Stable softmax and cross-entropy classification.
- Mean-squared-error regression.
- Backpropagation with L2 regularization.
- SGD, momentum SGD, and Adam optimizers.
- Mini-batch training, deterministic shuffling, validation monitoring, and
  early stopping.
- Classification and regression metrics.
- Numerical finite-difference gradient checking.
- Matplotlib learning curves, confusion matrices, and decision-boundary plots.
- Reproducible XOR and synthetic classification experiments.
- Unit tests for shapes, gradients, losses, optimizers, training, and metrics.

## Quick start

Python 3.9+ is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/NeuralCore.git
cd NeuralCore
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
python examples/xor_classification.py
```

For plotting, install the optional visualization dependency (included in `.[dev]`).

## Minimal example

```python
import numpy as np
from neuralcore import MLP, Trainer, TrainingConfig

rng = np.random.default_rng(7)
X = rng.normal(size=(500, 4))
y = (X[:, 0] + 0.7 * X[:, 1] > 0).astype(int)

model = MLP(
    input_dim=4,
    hidden_dims=(32, 16),
    output_dim=2,
    task="classification",
    hidden_activation="relu",
    seed=7,
)
trainer = Trainer(model, TrainingConfig(
    epochs=100, batch_size=32, learning_rate=0.01,
    optimizer="adam", patience=15,
))
history = trainer.fit(X, y, validation_data=(X, y))
probabilities = model.predict_proba(X[:5])
predictions = model.predict(X[:5])
print(history.best_epoch, predictions)
```

For classification, labels may be integer class IDs with shape `(n,)` or
one-hot rows with shape `(n, classes)`. For regression, targets should have
shape `(n, output_dim)`; a one-dimensional target is accepted for a
single-output model.

## Mathematical model

For a batch represented as row vectors, a dense layer computes

\[
Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)}, \qquad
A^{(l)} = \phi(Z^{(l)}).
\]

The input is \(A^{(0)}=X\). Each weight matrix has shape
`(fan_in, fan_out)` and each bias has shape `(1, fan_out)`.

For softmax classification,

\[
p_{ic} = \frac{\exp(z_{ic}-m_i)}
{\sum_j \exp(z_{ij}-m_i)}, \quad m_i=\max_j z_{ij},
\]

and the mean cross-entropy is

\[
L_{\mathrm{CE}}=-\frac1N\sum_i\log p_{i,y_i}.
\]

For mean-squared error,

\[
L_{\mathrm{MSE}}=\frac{1}{N D}\sum_{i,d}(\hat y_{id}-y_{id})^2.
\]

Backpropagation applies the chain rule from the output toward the input.
For a softmax-plus-cross-entropy output, the derivative with respect to logits
is `(probabilities - one_hot_targets) / batch_size`. For hidden layers,

\[
\delta^{(l)} =
(\delta^{(l+1)}(W^{(l+1)})^T)
\odot \phi'(Z^{(l)}).
\]

Then

\[
\nabla_{W^{(l)}}L=(A^{(l-1)})^T\delta^{(l)},\qquad
\nabla_{b^{(l)}}L=\sum_i\delta_i^{(l)}.
\]

L2 regularization adds \(\lambda\sum_l\|W^{(l)}\|_F^2/2\) to the objective,
so its weight-gradient contribution is \(\lambda W^{(l)}\). Biases are not
regularized.

## Project layout

```text
NeuralCore/
├── neuralcore/
│   ├── __init__.py
│   ├── activations.py
│   ├── losses.py
│   ├── metrics.py
│   ├── layers.py
│   ├── optimizers.py
│   ├── model.py
│   ├── trainer.py
│   ├── diagnostics.py
│   ├── visualization.py
│   └── data.py
├── examples/
│   ├── xor_classification.py
│   └── synthetic_benchmark.py
├── tests/
│   ├── test_activations_losses.py
│   ├── test_gradients.py
│   ├── test_model_training.py
│   └── test_metrics_optimizers.py
├── docs/
│   ├── MATHEMATICS.md
│   ├── EXPERIMENTS.md
│   └── RESEARCH_PROTOCOL.md
├── pyproject.toml
└── README.md
```

## Research use and limitations

NeuralCore is deliberately transparent. It is useful for checking derivations,
testing optimizer behavior, and benchmarking small dense networks under
controlled conditions. It is not optimized for GPUs, large datasets, automatic
differentiation, distributed training, or deployment. NumPy's BLAS backend,
CPU model, thread count, and package versions can affect timing.

A successful test suite is evidence that the tested cases pass; it is not proof
that the implementation is flawless. Before making a research claim, record
seeds, data-generation parameters, software versions, hardware, train/validation
splits, baseline models, hyperparameters, and uncertainty across repeated runs.
Do not interpret a single run as evidence of general superiority.

## Reproducibility

All examples accept a seed. The model and trainer use NumPy's `Generator` rather
than global random state. Save experiment outputs separately; generated outputs
are intentionally excluded from version control by `.gitignore`.

## License

MIT. See [LICENSE](LICENSE).
