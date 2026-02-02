# batch-susceptibility

**Model-free optimal batch size finder using susceptibility analysis.**

Stop guessing your batch size. This tool uses a physics-inspired method to find the critical batch size where your training dynamics undergo a phase transition — the point where batch statistics change from noisy (too small) to information-losing (too large).

```python
from batch_susceptibility.pytorch import find_optimal_batch_size

result = find_optimal_batch_size(model, dataset, loss_fn)
print(f"Use batch size: {result.K_c}")  # e.g., 128
```

## How It Works

For a sequence of per-step losses `L_1, L_2, ..., L_N`:

1. **Re-batch** at multiple scales `K`: group into batches of size `K` and compute batch means
2. **Measure** `V(K) = Var(batch means)` — how variable are the batch statistics?
3. **Differentiate**: `chi(K) = |d log V / d log K|` — the *susceptibility*
4. **Find the peak**: `K_c = argmax(chi)` — the *critical batch size*

For i.i.d. data, `V(K) ~ 1/K` (law of large numbers). Deviations from this scaling reveal temporal structure in your training dynamics, and the peak of `chi(K)` marks the characteristic correlation scale.

**Key metrics:**
- `K_c`: Optimal batch size (the scale where information density peaks)
- `kappa`: Peak sharpness (`max(chi)/median(chi)`). Higher = more confident. `>3` is significant.
- `alpha`: Scaling exponent. `-1` = i.i.d., `> -1` = correlated (drift), `< -1` = anti-correlated

Based on: M.C. Wurm, *"Batch-Size Susceptibility across Five Computational Domains"* (2024/2025).

## Installation

```bash
# Core (numpy + scipy only)
pip install batch-susceptibility

# With PyTorch integration
pip install batch-susceptibility[pytorch]

# With TensorFlow/Keras
pip install batch-susceptibility[tensorflow]

# With PyTorch Lightning
pip install batch-susceptibility[lightning]

# With plotting
pip install batch-susceptibility[plot]

# Everything
pip install batch-susceptibility[all]
```

## Quick Start

### PyTorch: Find Optimal Batch Size

```python
import torch.nn as nn
from torchvision import datasets, transforms, models
from batch_susceptibility.pytorch import find_optimal_batch_size

dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                           transform=transforms.ToTensor())
model = models.resnet18(num_classes=10)

result = find_optimal_batch_size(
    model, dataset, nn.CrossEntropyLoss(),
    batch_sizes=[16, 32, 64, 128, 256, 512, 1024],
    steps_per_size=100,
)
print(f"Optimal batch size: {result.K_c}")
```

### PyTorch: Monitor During Training

```python
from batch_susceptibility.pytorch import SusceptibilityCallback

monitor = SusceptibilityCallback(check_every=500)

for epoch in range(10):
    for batch in dataloader:
        loss = train_step(batch)
        monitor.on_step(loss)

# K_c is logged every 500 steps
final = monitor.result()
print(f"K_c = {final.K_c}, regime = {final.regime}")
```

### TensorFlow / Keras

```python
from batch_susceptibility.tensorflow import find_optimal_batch_size

result = find_optimal_batch_size(model, x_train, y_train)
print(f"Optimal batch size: {result.K_c}")
```

Or as a Keras callback:

```python
from batch_susceptibility.tensorflow import SusceptibilityCallback

cb = SusceptibilityCallback(check_every=500)
model.fit(x_train, y_train, epochs=10, callbacks=[cb])
print(cb.result().K_c)
```

### PyTorch Lightning

```python
from batch_susceptibility.lightning import SusceptibilityCallback

cb = SusceptibilityCallback(check_every=500)
trainer = pl.Trainer(callbacks=[cb])
trainer.fit(model, train_dataloader)
print(cb.result().K_c)
```

### HuggingFace Transformers

```python
from batch_susceptibility.examples.huggingface_transformers import SusceptibilityTrainerCallback

cb = SusceptibilityTrainerCallback(check_every=200)
trainer = Trainer(model=model, args=args, callbacks=[cb], ...)
trainer.train()
print(cb.result().K_c)
```

### Generic Data (any framework)

```python
from batch_susceptibility import BatchSusceptibility

# From per-step losses (any source)
bs = BatchSusceptibility()
bs.feed(losses)
result = bs.find_critical()
print(result.summary())
```

### Command Line

```bash
# Analyze a CSV of per-step losses
batch-susceptibility training_log.csv --column loss

# Analyze a batch-size sweep
batch-susceptibility sweep.csv --mode sweep --batch-col batch_size --metric-col loss

# JSON output
batch-susceptibility losses.csv -c loss --json

# With plot
batch-susceptibility losses.csv -c loss --plot result.png

# From stdin
cat losses.txt | batch-susceptibility -
```

## Interpreting Results

```
Critical batch size:  K_c = 128
Sharpness:            kappa = 5.42
Scaling exponent:     alpha = -0.87 +/- 0.02
i.i.d. test:          p = 0.0001
Regime:               weakly-correlated
Significant:          True
```

| Metric | Meaning |
|--------|---------|
| `K_c` | Optimal batch size. Use this for training. |
| `kappa > 3` | Confident detection. There IS a characteristic scale. |
| `kappa ~ 1` | No characteristic scale found. Data is scale-free. |
| `alpha ~ -1` | i.i.d. — no temporal correlations (normal). |
| `alpha > -1` | Positive correlations or drift (loss landscape structure). |
| `alpha < -1` | Anti-correlations (rare, indicates over-mixing). |

**Regime interpretation:**
- `iid`: Your data has no temporal structure. Any batch size works equally well.
- `correlated`: Training has drift/momentum. `K_c` marks where correlations decay.
- `weakly-correlated`: Mild structure. `K_c` is a soft optimum.
- `anti-correlated`: Unusual. Check for data preprocessing artifacts.

## Visualization

```python
from batch_susceptibility.plot import plot_susceptibility, plot_comparison

# Single result
fig = plot_susceptibility(result, title="My Model")
fig.savefig("susceptibility.png")

# Compare multiple experiments
fig = plot_comparison({
    "Adam lr=1e-3": result_adam,
    "SGD lr=0.1": result_sgd,
})
fig.savefig("comparison.png")
```

## API Reference

### `BatchSusceptibility`

Main interface class.

```python
bs = BatchSusceptibility(K_min=2, K_max=None, n_points=60, min_batches=10)
bs.feed(values)             # Feed 1D sequence of metric values
result = bs.find_critical() # Compute susceptibility, find K_c
```

### `SusceptibilityResult`

Dataclass with all results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `K_c` | `float` | Critical batch size |
| `kappa` | `float` | Peak sharpness |
| `alpha` | `float` | Scaling exponent |
| `alpha_se` | `float` | Standard error of alpha |
| `p_iid` | `float` | p-value for i.i.d. null |
| `K_values` | `ndarray` | Batch sizes tested |
| `V_values` | `ndarray` | Variance at each K |
| `chi_values` | `ndarray` | Susceptibility at each K |
| `is_significant` | `bool` | kappa > 3 and p < 0.05 |
| `regime` | `str` | iid / correlated / weakly-correlated / anti-correlated |

### Framework integrations

| Module | Class/Function | Description |
|--------|---------------|-------------|
| `batch_susceptibility.pytorch` | `find_optimal_batch_size()` | One-call PyTorch finder |
| `batch_susceptibility.pytorch` | `BatchSizeFinder` | Configurable PyTorch finder |
| `batch_susceptibility.pytorch` | `SusceptibilityCallback` | Training monitor |
| `batch_susceptibility.tensorflow` | `find_optimal_batch_size()` | One-call Keras finder |
| `batch_susceptibility.tensorflow` | `SusceptibilityCallback` | Keras callback |
| `batch_susceptibility.lightning` | `SusceptibilityCallback` | Lightning callback |
| `batch_susceptibility.lightning` | `BatchSizeFinder` | Lightning finder |
| `batch_susceptibility.plot` | `plot_susceptibility()` | Single result plot |
| `batch_susceptibility.plot` | `plot_comparison()` | Multi-result comparison |
| `batch_susceptibility.cli` | `main()` | CLI entry point |

## Examples

See the [`examples/`](examples/) directory:

- [`pytorch_cifar10.py`](examples/pytorch_cifar10.py) — CIFAR-10 with ResNet-18
- [`pytorch_monitor.py`](examples/pytorch_monitor.py) — Online monitoring during training
- [`tensorflow_mnist.py`](examples/tensorflow_mnist.py) — MNIST with Keras
- [`huggingface_transformers.py`](examples/huggingface_transformers.py) — HuggingFace Trainer integration
- [`generic_csv.py`](examples/generic_csv.py) — Any data source / CSV

## Theory

The method is based on the observation that for any stationary time series, the variance of batch means `V(K)` follows a power law `V(K) ~ K^alpha` where:

- `alpha = -1` for independent, identically distributed (i.i.d.) samples
- `alpha > -1` when consecutive samples are positively correlated
- `alpha < -1` when consecutive samples are anti-correlated

The *susceptibility* `chi(K) = |d log V / d log K|` measures how sensitive the variance scaling is to the batch size. A peak in `chi(K)` indicates a *critical scale* — the batch size at which the dominant correlation structure is resolved.

**For ML training:** This critical scale corresponds to the batch size where:
- Smaller batches: gradient estimates are dominated by noise correlations
- Larger batches: diminishing returns — averaging out signal, not just noise
- At `K_c`: optimal information density per gradient step

This connects to the *critical batch size* concept from [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) but requires no gradient computation — only loss values.

## License

**Dual-licensed** under:
- **AGPL-3.0** for open-source / non-commercial use ([license_AGPL.txt](license_AGPL.txt))
- **Commercial license** for proprietary integration ([license_COMMERCIAL.txt](license_COMMERCIAL.txt))

See [LICENSE](LICENSE) for full details.

For commercial licensing, contact: **nfo@forgottenforge.xyz**


## Related

- [sigmacore](https://github.com/forgottenforge/sigmacore) — The general sigma_c framework for critical scale detection across domains
