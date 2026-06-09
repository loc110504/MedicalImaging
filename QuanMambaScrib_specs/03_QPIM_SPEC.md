# 03 - Quantum Prototype Interaction Module Specification

## Purpose
QPIM converts sparse scribble evidence into class prototype affinity maps. It is not a segmentation backbone. It acts as a verifier for pseudo-label reliability.

## Class API
File: `code/networks/qpim.py`

```python
class QuantumPrototypeInteractionModule(nn.Module):
    def __init__(
        self,
        in_channels_u: int,
        in_channels_m: int,
        num_classes: int,
        proj_dim: int = 8,
        tau_q: float = 0.5,
        momentum: float = 0.99,
        ignore_index: int = None,
        backend: str = "torch_angle_fidelity",
        rbf_gamma: float = 1.0,
        mlp_hidden: int = 64,
        normalize_z: bool = True,
        detach_prototypes: bool = False,
    ):
        ...

    def forward(
        self,
        feat_u: torch.Tensor,
        feat_m: torch.Tensor,
        scribble_label: torch.Tensor,
        target_size: tuple,
        update_memory: bool = True,
    ) -> dict:
        ...
```

## Input Shapes

```text
feat_u:         [B, C_u, h_u, w_u]
feat_m:         [B, C_m, h_m, w_m]
scribble_label: [B, H, W]
target_size:    (H, W)
```

Scribble unknown pixels use:

```python
ignore_index = num_classes
```

## Output Dictionary

```python
{
    "Q_u": Q_u,                  # [B, K, H, W]
    "Q_m": Q_m,                  # [B, K, H, W]
    "q_logits_u": q_logits_u,    # [B, K, H, W]
    "q_logits_m": q_logits_m,    # [B, K, H, W]
    "Q_u_low": Q_u_low,          # [B, K, h_u, w_u]
    "Q_m_low": Q_m_low,          # [B, K, h_m, w_m]
    "proto_u": proto_u,          # [K, d]
    "proto_m": proto_m,          # [K, d]
    "present_classes": present,  # [K] bool
}
```

## Projection Heads

```python
self.proj_u = nn.Sequential(
    nn.Conv2d(in_channels_u, proj_dim, kernel_size=1, bias=False),
    nn.GroupNorm(1, proj_dim),
    nn.Tanh(),
)

self.proj_m = nn.Sequential(
    nn.Conv2d(in_channels_m, proj_dim, kernel_size=1, bias=False),
    nn.GroupNorm(1, proj_dim),
    nn.Tanh(),
)
```

`Tanh` keeps angle values bounded and improves numerical stability for angle encoding.

## Prototype Construction
Use full-resolution scribble labels to avoid losing thin scribbles during downsampling.

Recommended steps:

1. Project low-resolution features:

```python
Z_u_low = self.proj_u(feat_u)  # [B, d, h_u, w_u]
Z_m_low = self.proj_m(feat_m)  # [B, d, h_m, w_m]
```

2. Upsample projected features to label resolution for prototype extraction:

```python
Z_u_full = F.interpolate(Z_u_low, size=target_size, mode="bilinear", align_corners=False)
Z_m_full = F.interpolate(Z_m_low, size=target_size, mode="bilinear", align_corners=False)
```

3. Build class prototypes from labeled scribble pixels:

```python
valid = scribble_label != ignore_index
for k in range(num_classes):
    mask_k = (scribble_label == k) & valid
    if mask_k.sum() > 0:
        proto_k = Z_full.permute(0,2,3,1)[mask_k].mean(dim=0)
    else:
        proto_k = memory[k]
```

4. Normalize prototypes if `normalize_z=True`:

```python
proto = F.normalize(proto, dim=-1)
```

## Prototype Memory
Register buffers:

```python
self.register_buffer("memory_u", torch.zeros(num_classes, proj_dim))
self.register_buffer("memory_m", torch.zeros(num_classes, proj_dim))
self.register_buffer("memory_init_u", torch.zeros(num_classes, dtype=torch.bool))
self.register_buffer("memory_init_m", torch.zeros(num_classes, dtype=torch.bool))
```

Update rule:

```python
if class_present:
    if not memory_init[k]:
        memory[k] = proto.detach()
        memory_init[k] = True
    else:
        memory[k] = momentum * memory[k] + (1 - momentum) * proto.detach()
```

If class is absent and memory is not initialized, use a zero vector but do not allow NaN. Log a warning once.

## Kernel Backends
QPIM must support multiple backends for ablation.

### 1. `torch_angle_fidelity` - Default
Fast differentiable approximation of angle-encoded quantum fidelity.

For a vector `z` with dimension `d`, product-state angle encoding via `Ry(z_r)` gives per-qubit state:

```text
[cos(z_r / 2), sin(z_r / 2)]
```

The inner product between angle-encoded states factorizes:

```text
<psi(z)|psi(p)> = product_r cos((z_r - p_r) / 2)
```

Therefore:

```python
def angle_fidelity_kernel(z, proto):
    # z: [N, d], proto: [K, d]
    diff = z[:, None, :] - proto[None, :, :]
    inner = torch.cos(0.5 * diff).prod(dim=-1)
    kernel = inner.pow(2)
    return kernel  # [N, K]
```

This is efficient and matches the fidelity of angle-encoded product states. It does not include trainable entanglement, so use `pennylane_state_fidelity` for strict circuit ablation.

### 2. `cosine`

```python
kernel = torch.matmul(F.normalize(z), F.normalize(proto).t()).clamp(-1,1).pow(2)
```

### 3. `rbf`

```python
kernel = torch.exp(-gamma * torch.cdist(z, proto).pow(2))
```

### 4. `mlp_affinity`
Trainable classical nonlinear baseline.

```python
pair = torch.cat([z_i, proto_k, torch.abs(z_i - proto_k), z_i * proto_k], dim=-1)
score = mlp(pair)
```

### 5. `pennylane_state_fidelity` - Optional Strict Quantum Circuit
Use only for small maps or sampled tokens because it is slow.

Circuit structure:

```python
@qml.qnode(dev, interface="torch")
def quantum_state(x, weights):
    for r in range(d):
        qml.RY(x[r], wires=r)
    for layer in range(L):
        for r in range(d):
            qml.RY(weights[layer, r, 0], wires=r)
            qml.RZ(weights[layer, r, 1], wires=r)
        for r in range(d):
            qml.CNOT(wires=[r, (r + 1) % d])
    return qml.state()
```

Compute:

```python
kernel = abs(torch.sum(torch.conj(psi_z) * psi_proto)) ** 2
```

Do not run this for every full-resolution pixel. Restrict to:
- low-resolution feature map, e.g. 16x16 or 32x32;
- small batch;
- sampled pixels;
- ablation only.

## Affinity Prediction
For each branch:

```python
kernel_u = compute_kernel(Z_u_low_flat, proto_u)  # [B*h*w, K]
q_logits_u_low = kernel_u / tau_q
Q_u_low = torch.softmax(q_logits_u_low, dim=-1)
```

Reshape and upsample:

```python
Q_u_low = Q_u_low.view(B, h, w, K).permute(0, 3, 1, 2)
Q_u = F.interpolate(Q_u_low, size=target_size, mode="bilinear", align_corners=False)
Q_u = Q_u / (Q_u.sum(dim=1, keepdim=True) + 1e-8)
```

Repeat for Mamba branch.

## Numerical Safety
- Never divide by zero when no scribble pixels exist for a class.
- If `Q.sum(dim=1)` has tiny values, renormalize with epsilon.
- If reliable pixel mask is empty, loss returns `0.0` tensor on correct device.
- Use `torch.nan_to_num` defensively after kernel computation.

## Unit Tests
1. Random features + random scribble labels produce finite `Q_u` and `Q_m`.
2. `Q_u.sum(dim=1)` and `Q_m.sum(dim=1)` approximately equal 1.
3. If a class is absent, memory fallback works.
4. If all pixels are ignore_index, module does not crash and returns finite outputs.
5. All buffers move correctly with `.cuda()` and `state_dict` save/load.
