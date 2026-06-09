# 07 - Quantum Backends and Dependencies Specification

## Recommended Framework
Use **PyTorch + PennyLane**.

- PyTorch handles segmentation networks, dataloaders, losses, and optimizers.
- PennyLane supports PyTorch-compatible QNodes with `interface="torch"` and can convert QNodes into Torch layers.

## Dependency Additions
Add to `requirements.txt`:

```text
pennylane>=0.35
pennylane-lightning
```

Optional for strict Mamba implementations:

```text
mamba-ssm
causal-conv1d
```

If environment issues occur, keep QPIM default backend as `torch_angle_fidelity`, which does not require PennyLane.

## Why Multiple QPIM Backends Are Required
The method needs ablations to defend quantum use. Implement all backends with the same interface:

```python
kernel = self.compute_kernel(z_flat, prototypes)
```

Supported backends:

```text
torch_angle_fidelity  # default fast quantum-inspired/simulator-style fidelity
cosine                # classical baseline
rbf                   # classical kernel baseline
mlp_affinity          # classical nonlinear baseline
pennylane_state_fidelity # strict quantum circuit ablation, slow
```

## Default Backend: `torch_angle_fidelity`
This backend implements the fidelity of angle-encoded product states:

```python
def angle_fidelity(z, p):
    diff = z[:, None, :] - p[None, :, :]
    inner = torch.cos(0.5 * diff).prod(dim=-1)
    return inner.pow(2)
```

Advantages:
- GPU-friendly;
- fully vectorized;
- differentiable;
- scalable to low-resolution dense feature maps;
- compatible with the paper's angle-encoding motivation.

Limitation:
- Does not explicitly simulate trainable entanglement layers.

## Strict PennyLane Backend
Implement but do not use by default for full training.

### QNode Pseudocode

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=proj_dim)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_state(x, weights):
    for r in range(proj_dim):
        qml.RY(x[r], wires=r)
    for l in range(num_layers):
        for r in range(proj_dim):
            qml.RY(weights[l, r, 0], wires=r)
            qml.RZ(weights[l, r, 1], wires=r)
        for r in range(proj_dim):
            qml.CNOT(wires=[r, (r + 1) % proj_dim])
    return qml.state()
```

### Fidelity

```python
psi_z = quantum_state(z, weights)
psi_p = quantum_state(proto, weights)
fidelity = torch.abs(torch.sum(torch.conj(psi_z) * psi_p)) ** 2
```

### Performance Warnings
Do not call the PennyLane QNode for every full-resolution pixel. Use only:
- low-resolution feature maps;
- sampled pixels;
- small ablation experiments;
- unit tests.

## Backend Equivalence Test
For `proj_dim=4`, compare `torch_angle_fidelity` against a no-entanglement PennyLane circuit with only `RY` encoding. They should be numerically close.

## Config Flags

```python
--qpim_backend torch_angle_fidelity
--qpim_dim 8
--qpim_tau 0.5
--qpim_quantum_layers 2
--qpim_use_pennylane 0
```

If `--qpim_backend pennylane_state_fidelity`, automatically restrict feature tokens:

```python
--qpim_sample_tokens 1024
```

and log a warning.
