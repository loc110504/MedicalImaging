import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pennylane as qml
except ImportError:  # pragma: no cover - optional dependency
    qml = None


class QuantumPrototypeInteractionModule(nn.Module):
    def __init__(
        self,
        in_channels_u,
        in_channels_m,
        num_classes,
        proj_dim=8,
        tau_q=0.5,
        momentum=0.99,
        ignore_index=None,
        backend="torch_angle_fidelity",
        rbf_gamma=1.0,
        mlp_hidden=64,
        normalize_z=True,
        detach_prototypes=False,
        quantum_layers=2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.tau_q = tau_q
        self.momentum = momentum
        self.ignore_index = num_classes if ignore_index is None else ignore_index
        self.backend = backend
        self.rbf_gamma = rbf_gamma
        self.normalize_z = normalize_z
        self.detach_prototypes = detach_prototypes
        self.quantum_layers = quantum_layers

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
        self.affinity_mlp = nn.Sequential(
            nn.Linear(proj_dim * 4, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.quantum_weights = nn.Parameter(torch.zeros(quantum_layers, proj_dim, 2))

        self.register_buffer("memory_u", torch.zeros(num_classes, proj_dim))
        self.register_buffer("memory_m", torch.zeros(num_classes, proj_dim))
        self.register_buffer("memory_init_u", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("memory_init_m", torch.zeros(num_classes, dtype=torch.bool))

    def forward(self, feat_u, feat_m, scribble_label, target_size, update_memory=True):
        z_u_low = self.proj_u(feat_u)
        z_m_low = self.proj_m(feat_m)

        z_u_full = F.interpolate(z_u_low, size=target_size, mode="bilinear", align_corners=False)
        z_m_full = F.interpolate(z_m_low, size=target_size, mode="bilinear", align_corners=False)

        proto_u, present_u = self._build_prototypes(
            z_full=z_u_full,
            scribble_label=scribble_label,
            memory=self.memory_u,
            memory_init=self.memory_init_u,
            update_memory=update_memory,
        )
        proto_m, present_m = self._build_prototypes(
            z_full=z_m_full,
            scribble_label=scribble_label,
            memory=self.memory_m,
            memory_init=self.memory_init_m,
            update_memory=update_memory,
        )

        q_logits_u_low, q_u_low = self._predict_affinity(z_u_low, proto_u, self.memory_init_u | present_u)
        q_logits_m_low, q_m_low = self._predict_affinity(z_m_low, proto_m, self.memory_init_m | present_m)

        q_logits_u = F.interpolate(q_logits_u_low, size=target_size, mode="bilinear", align_corners=False)
        q_logits_m = F.interpolate(q_logits_m_low, size=target_size, mode="bilinear", align_corners=False)
        q_u = F.interpolate(q_u_low, size=target_size, mode="bilinear", align_corners=False)
        q_m = F.interpolate(q_m_low, size=target_size, mode="bilinear", align_corners=False)
        q_u = q_u / (q_u.sum(dim=1, keepdim=True) + 1e-8)
        q_m = q_m / (q_m.sum(dim=1, keepdim=True) + 1e-8)

        return {
            "Q_u": q_u,
            "Q_m": q_m,
            "q_logits_u": q_logits_u,
            "q_logits_m": q_logits_m,
            "Q_u_low": q_u_low,
            "Q_m_low": q_m_low,
            "proto_u": proto_u,
            "proto_m": proto_m,
            "present_classes": present_u | present_m,
        }

    def _normalize(self, z):
        if not self.normalize_z:
            return z
        return F.normalize(z, dim=-1, eps=1e-8)

    def _build_prototypes(self, z_full, scribble_label, memory, memory_init, update_memory):
        b, d, h, w = z_full.shape
        z_flat = z_full.permute(0, 2, 3, 1).reshape(-1, d)
        label_flat = scribble_label.reshape(-1)

        proto_list = []
        present = []
        for cls_idx in range(self.num_classes):
            cls_mask = label_flat == cls_idx
            is_present = bool(cls_mask.any())
            present.append(is_present)
            if is_present:
                proto = z_flat[cls_mask].mean(dim=0)
                proto = self._normalize(proto.unsqueeze(0)).squeeze(0)
                if update_memory:
                    self._update_memory(memory, memory_init, cls_idx, proto)
            elif memory_init[cls_idx]:
                proto = memory[cls_idx]
            else:
                proto = z_full.new_zeros(d)
            proto_list.append(proto)

        proto = torch.stack(proto_list, dim=0)
        if self.detach_prototypes:
            proto = proto.detach()
        return proto, torch.tensor(present, device=z_full.device, dtype=torch.bool)

    @torch.no_grad()
    def _update_memory(self, memory, memory_init, cls_idx, proto):
        proto_detached = proto.detach()
        if not memory_init[cls_idx]:
            memory[cls_idx] = proto_detached
            memory_init[cls_idx] = True
        else:
            memory[cls_idx] = self.momentum * memory[cls_idx] + (1.0 - self.momentum) * proto_detached
            memory[cls_idx] = self._normalize(memory[cls_idx].unsqueeze(0)).squeeze(0)

    def _predict_affinity(self, z_low, prototypes, available_mask):
        b, d, h, w = z_low.shape
        z_flat = z_low.permute(0, 2, 3, 1).reshape(-1, d)
        z_flat = self._normalize(z_flat)
        proto = self._normalize(prototypes)
        kernel = self.compute_kernel(z_flat, proto)
        kernel = torch.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
        q_logits = self._apply_available_mask(kernel / max(self.tau_q, 1e-8), available_mask)
        q_prob = torch.softmax(q_logits, dim=-1)
        q_logits = q_logits.view(b, h, w, self.num_classes).permute(0, 3, 1, 2).contiguous()
        q_prob = q_prob.view(b, h, w, self.num_classes).permute(0, 3, 1, 2).contiguous()
        return q_logits, q_prob

    def _apply_available_mask(self, logits, available_mask):
        if bool(available_mask.any()):
            logits = logits.clone()
            logits[:, ~available_mask] = -1e4
        return logits

    def compute_kernel(self, z, proto):
        if self.backend == "torch_angle_fidelity":
            diff = z[:, None, :] - proto[None, :, :]
            inner = torch.cos(0.5 * diff).prod(dim=-1)
            return inner.pow(2)

        if self.backend == "cosine":
            return torch.matmul(self._normalize(z), self._normalize(proto).t()).clamp(-1.0, 1.0).pow(2)

        if self.backend == "rbf":
            return torch.exp(-self.rbf_gamma * torch.cdist(z, proto).pow(2))

        if self.backend == "mlp_affinity":
            num_tokens = z.shape[0]
            num_classes = proto.shape[0]
            z_expand = z[:, None, :].expand(num_tokens, num_classes, self.proj_dim)
            p_expand = proto[None, :, :].expand(num_tokens, num_classes, self.proj_dim)
            pair = torch.cat([z_expand, p_expand, torch.abs(z_expand - p_expand), z_expand * p_expand], dim=-1)
            score = self.affinity_mlp(pair.reshape(-1, pair.shape[-1]))
            return score.view(num_tokens, num_classes)

        if self.backend == "pennylane_state_fidelity":
            return self._compute_pennylane_kernel(z, proto)

        raise ValueError("Unsupported qpim backend: {}".format(self.backend))

    def _compute_pennylane_kernel(self, z, proto):  # pragma: no cover - slow optional backend
        if qml is None:
            raise ImportError("pennylane is required for backend='pennylane_state_fidelity'")

        dev = qml.device("default.qubit", wires=self.proj_dim)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def quantum_state(x, weights):
            for idx in range(self.proj_dim):
                qml.RY(x[idx], wires=idx)
            for layer_idx in range(self.quantum_layers):
                for idx in range(self.proj_dim):
                    qml.RY(weights[layer_idx, idx, 0], wires=idx)
                    qml.RZ(weights[layer_idx, idx, 1], wires=idx)
                for idx in range(self.proj_dim):
                    qml.CNOT(wires=[idx, (idx + 1) % self.proj_dim])
            return qml.state()

        values = []
        for token in z:
            row = []
            psi_token = quantum_state(token, self.quantum_weights)
            for cur_proto in proto:
                psi_proto = quantum_state(cur_proto, self.quantum_weights)
                fidelity = torch.abs(torch.sum(torch.conj(psi_token) * psi_proto)) ** 2
                row.append(fidelity.real)
            values.append(torch.stack(row))
        return torch.stack(values, dim=0)
