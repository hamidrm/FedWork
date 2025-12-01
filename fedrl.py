# flsu_mnist_bitswitch.py
# Forward-Only Federated Learning with 1-bit / 2-bit Activations + Sign Updates
# - Toggle ACT_BITS = 1 or 2 to choose activation precision.
# - Clients: full forward; send low-bit activations {a0_q, h1_q, h2_q} + labels
# - Server: uses ONLY those to compute sign(grads) for ALL params; sends signs
# - Clients: apply ±η (optional velocity); Server mirrors same step
# - Periodically: average all per-client models -> broadcast global

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -----------------------------
# HYPERPARAMS
# -----------------------------
NUM_CLIENTS   = 10
ROUNDS        = 200
LOCAL_STEPS   = 1        # mini-batches per client per round
BATCH_SIZE    = 1
SERVER_BETA   = 0.90     # server momentum before sign
CLIENT_MOM    = 0.01     # client velocity (EF-lite); try 0.8 after it works
BASE_LR       = 3e-3     # base step; scaled per tensor by 1/sqrt(fan_in)
AVG_PERIOD    = 20       # rounds between global averaging/broadcast
ACT_BITS      = 1        # <<< set to 1 or 2
SEED          = 0
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
EPS           = 1e-8

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=0):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def copy_params_from(mod: nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in mod.parameters()]

def load_params_into(mod: nn.Module, params: List[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, v in zip(mod.parameters(), params): p.copy_(v)

def zeros_like_params_from(params: List[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in params]

def sign_tie_break(x: torch.Tensor) -> torch.Tensor:
    s = x.sign()
    return torch.where(s == 0, torch.ones_like(s), s)

# ----- quantization helpers -----
def quantize_input(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Input proxy:
      bits=1: bipolar sign {-1,+1}
      bits=2: symmetric 4-level {-1,-1/3,1/3,1} * s, s = amax(|x|)
    """
    if bits == 1:
        return sign_tie_break(x)
    elif bits == 2:
        s = x.detach().abs().amax() + EPS
        u = torch.clamp(x / s, -1.0, 1.0)
        # map to {-1, -1/3, 1/3, 1}
        idx = torch.round((u + 1.0) * 1.5)          # 0..3
        levels = torch.tensor([-1.0, -1.0/3.0, 1.0/3.0, 1.0], device=x.device, dtype=x.dtype)
        q = levels[(idx.long())]
        return q * s
    else:
        raise ValueError("ACT_BITS must be 1 or 2")

def quantize_hidden_relu(h_pos: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Hidden proxy (after ReLU):
      bits=1: gate {0,1}
      bits=2: non-negative 4-level {0, 1/3, 2/3, 1} * s, s = amax(h_pos)
    """
    if bits == 1:
        return (h_pos > 0).to(h_pos.dtype)
    elif bits == 2:
        s = h_pos.detach().amax() + EPS
        u = torch.clamp(h_pos / s, 0.0, 1.0)
        idx = torch.round(u * 3.0)                  # 0..3
        levels = torch.tensor([0.0, 1.0/3.0, 2.0/3.0, 1.0], device=h_pos.device, dtype=h_pos.dtype)
        q = levels[(idx.long())]
        return q * s
    else:
        raise ValueError("ACT_BITS must be 1 or 2")

# -----------------------------
# Model: 3-layer MLP (784→256→256→10)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=784, h=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, out_dim)
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            nn.init.zeros_(m.bias)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

# -----------------------------
# Federated state
# -----------------------------
@dataclass
class ClientState:
    params: List[torch.Tensor]
    velocity: List[torch.Tensor] = field(default_factory=list)

@dataclass
class ServerSlot:
    params: List[torch.Tensor]          # server's mirror of client's params
    mom:    List[torch.Tensor]          # momentum buffers (same shapes)

# -----------------------------
# Data (MNIST; IID split)
# -----------------------------
def make_mnist_loaders(data_dir="./data", batch_size=256, num_clients=5):
    tfm = transforms.Compose([
        transforms.ToTensor(),                 # [0,1]
        transforms.Normalize((0.5,), (0.5,)),  # ~[-1,1]
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    idxs = list(range(len(train_ds))); random.shuffle(idxs)
    per = len(train_ds) // num_clients
    splits = []
    for k in range(num_clients):
        sl = idxs[k*per:(k+1)*per] if k < num_clients-1 else idxs[k*per:]
        splits.append(Subset(train_ds, sl))

    client_loaders = [DataLoader(s, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
                      for s in splits]
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)
    return client_loaders, test_loader

# -----------------------------
# Client: full forward → low-bit bundle
# returns (a0_q, h1_q, h2_q)
# -----------------------------
@torch.no_grad()
def client_forward_collect_lowbit(x, model: MLP, params: List[torch.Tensor], device, act_bits: int):
    load_params_into(model, params)
    x = x.to(device).view(x.size(0), -1)                 # [B,784]

    # input proxy
    a0_q = quantize_input(x, act_bits)                   # [B,784]

    # layer 1 pre-activation and ReLU
    a1 = F.linear(x, model.fc1.weight, model.fc1.bias)   # [B,H]
    h1_pos = torch.clamp(a1, min=0.0)
    h1_q = quantize_hidden_relu(h1_pos, act_bits)        # [B,H]

    # layer 2 pre-activation and ReLU
    a2 = F.linear(h1_pos, model.fc2.weight, model.fc2.bias)  # [B,H]
    h2_pos = torch.clamp(a2, min=0.0)
    h2_q = quantize_hidden_relu(h2_pos, act_bits)            # [B,H]

    return a0_q, h1_q, h2_q

# -----------------------------
# Server: use ONLY low-bit bundle to compute sign updates for ALL params
# -----------------------------
def server_signed_update_from_lowbit(
    server_model: MLP,
    server_params: List[torch.Tensor],
    y: torch.Tensor,
    a0_q: torch.Tensor, h1_q: torch.Tensor, h2_q: torch.Tensor,
    mom: List[torch.Tensor],
    beta_mom: float,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    load_params_into(server_model, server_params)
    y = y.to(device)

    # derivative masks from proxies (works for both bits)
    g2 = (h2_q > 0).to(h2_q.dtype)      # {0,1}
    g1 = (h1_q > 0).to(h1_q.dtype)

    # forward (last layer only)
    z = F.linear(h2_q, server_model.fc3.weight, server_model.fc3.bias)
    probs = F.softmax(z, dim=1)
    loss = F.nll_loss((probs + EPS).log(), y)

    # grad wrt logits
    grad_z = probs
    grad_z[torch.arange(y.size(0), device=device), y] -= 1.0
    grad_z /= y.size(0)

    # ---- fc3 grads ----
    W3 = server_model.fc3.weight.detach()      # [10,H]
    gW3 = grad_z.t().matmul(h2_q)              # [10,H]
    gb3 = grad_z.sum(dim=0)                    # [10]
    grad_h2 = grad_z.matmul(W3)                # [B,H]

    # ---- fc2 grads (mask with g2) ----
    W2 = server_model.fc2.weight.detach()      # [H,H]
    delta2 = grad_h2 * g2
    gW2 = delta2.t().matmul(h1_q)              # [H,H]
    gb2 = delta2.sum(dim=0)                    # [H]
    grad_h1 = delta2.matmul(W2)                # [B,H]

    # ---- fc1 grads (mask with g1); input proxy is a0_q ----
    delta1 = grad_h1 * g1
    gW1 = delta1.t().matmul(a0_q)              # [H,784]
    gb1 = delta1.sum(dim=0)                    # [H]

    grads = [gW1, gb1, gW2, gb2, gW3, gb3]

    # momentum + sign
    signs = []
    with torch.no_grad():
        for g, m in zip(grads, mom):
            m.mul_(beta_mom).add_(g, alpha=1.0 - beta_mom)
            s = m.sign(); s[s == 0] = 1.0
            signs.append(s.detach())
    return signs, mom, float(loss.item())

# -----------------------------
# Apply ±η (client; with velocity) and mirror (server)
# -----------------------------
def apply_sign_step(params: List[torch.Tensor], signs: List[torch.Tensor],
                    lrs: List[float], velocity: List[torch.Tensor] = None,
                    momentum: float = 0.0):
    if velocity is None or len(velocity) != len(params):
        velocity = [torch.zeros_like(p) for p in params]
    with torch.no_grad():
        for i, (p, s) in enumerate(zip(params, signs)):
            step = -lrs[i] * s
            if momentum > 0.0:
                velocity[i].mul_(momentum).add_(step)
                p.add_(velocity[i])
            else:
                p.add_(step)
    return params, velocity

def mirror_sign_step(params: List[torch.Tensor], signs: List[torch.Tensor], lrs: List[float]):
    with torch.no_grad():
        for i, (p, s) in enumerate(zip(params, signs)):
            p.add_(s, alpha=-lrs[i])
    return params

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(full_model: MLP, params: List[torch.Tensor], loader: DataLoader, device) -> float:
    full_model.eval()
    load_params_into(full_model, params)
    tot, ok = 0, 0
    for x, y in loader:
        x = x.to(device).view(x.size(0), -1)
        y = y.to(device)
        logits = full_model(x)
        ok += (logits.argmax(1) == y).sum().item()
        tot += y.numel()
    return ok / tot

# -----------------------------
# Main training loop
# -----------------------------
def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    # Data
    client_loaders, test_loader = make_mnist_loaders(batch_size=BATCH_SIZE, num_clients=NUM_CLIENTS)

    # Templates
    template = MLP().to(device)
    init_params = copy_params_from(template)

    # Per-tensor LR scaling by 1/sqrt(fan_in)
    layer_fanin = []
    for p in template.parameters():
        fi = (p.numel() / p.shape[0]) if p.ndim >= 2 else 1.0
        layer_fanin.append(max(1.0, float(fi)))
    layer_lrs = [BASE_LR / math.sqrt(fi) for fi in layer_fanin]  # order: W1,b1,W2,b2,W3,b3

    # Clients
    clients: List[ClientState] = [
        ClientState(params=[p.clone() for p in init_params],
                    velocity=[torch.zeros_like(p) for p in init_params])
        for _ in range(NUM_CLIENTS)
    ]

    # Server mirrors & momenta
    server_slots: List[ServerSlot] = [
        ServerSlot(params=[p.clone() for p in init_params],
                   mom=[torch.zeros_like(p) for p in init_params])
        for _ in range(NUM_CLIENTS)
    ]

    # Working models
    client_model = MLP().to(device)
    server_model = MLP().to(device)

    for rnd in range(1, ROUNDS + 1):
        losses = []

        for cid, (client, loader) in enumerate(zip(clients, client_loaders)):
            it = iter(loader)
            for _ in range(LOCAL_STEPS):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader); x, y = next(it)

                # ---- CLIENT: forward-only → low-bit bundle ----
                a0_q, h1_q, h2_q = client_forward_collect_lowbit(x, client_model, client.params, device, ACT_BITS)

                # ---- SERVER: sign updates using ONLY low-bit & labels ----
                signs, server_slots[cid].mom, loss_val = server_signed_update_from_lowbit(
                    server_model, server_slots[cid].params, y.to(device),
                    a0_q, h1_q, h2_q,
                    server_slots[cid].mom, SERVER_BETA, device
                )
                losses.append(loss_val)

                # ---- CLIENT: apply ±η (velocity) ----
                client.params, client.velocity = apply_sign_step(
                    client.params, signs, layer_lrs, client.velocity, CLIENT_MOM
                )

                # ---- SERVER: mirror the SAME step ----
                server_slots[cid].params = mirror_sign_step(server_slots[cid].params, signs, layer_lrs)

        # ---- Periodic averaging → new global ----
        if rnd % AVG_PERIOD == 0:
            with torch.no_grad():
                avg = []
                for i in range(len(init_params)):
                    avg.append(torch.stack([slot.params[i] for slot in server_slots], dim=0).mean(dim=0))
                # broadcast
                for cid in range(NUM_CLIENTS):
                    for dst, src in zip(clients[cid].params, avg): dst.copy_(src)
                    for dst, src in zip(server_slots[cid].params, avg): dst.copy_(src)
                    # reset memories after sync (stability)
                    clients[cid].velocity = [torch.zeros_like(p) for p in clients[cid].params]
                    server_slots[cid].mom = [torch.zeros_like(p) for p in server_slots[cid].mom]

        # ---- Evaluate "global" as mean of mirrors (proxy) ----
        with torch.no_grad():
            glob = []
            for i in range(len(init_params)):
                glob.append(torch.stack([slot.params[i] for slot in server_slots], dim=0).mean(dim=0))
        acc = evaluate(template, glob, test_loader, device)

        #if rnd % AVG_PERIOD == 0:
        print(f"[Round {rnd:03d}] mean proxy loss: {sum(losses)/max(1,len(losses)):.4f} | "
                  f"global test acc: {acc:.4f} | ACT_BITS={ACT_BITS}")

    print("Done.")

if __name__ == "__main__":
    main()
