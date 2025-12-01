# flsu_mnist_bitswitch_cnn.py
# Forward-Only Federated Learning (CNN) with 1-bit / 2-bit Activations + Sign Updates
# - Toggle ACT_BITS = 1 or 2 to choose activation precision.
# - Clients: full forward; send low-bit activations {a0_q, h1_q..h6_q} + labels
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
from torch.nn.grad import conv2d_weight, conv2d_input
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -----------------------------
# HYPERPARAMS
# -----------------------------
NUM_CLIENTS   = 10
ROUNDS        = 200
LOCAL_STEPS   = 8        # mini-batches per client per round
BATCH_SIZE    = 8
SERVER_BETA   = 0.8     # server momentum before sign
CLIENT_MOM    = 0.1     # client velocity (EF-lite); try 0.8 after it works
BASE_LR       = 1e-3     # base step; scaled per tensor by 1/sqrt(fan_in)
AVG_PERIOD    = 10       # rounds between global averaging/broadcast
ACT_BITS      = 2        # <<< set to 1 or 2
SEED          = 0
DEVICE        = "cuda:0" if torch.cuda.is_available() else "cpu"
EPS           = 1e-8
DEBUG_MAP     = False    # set True to print param↔LR mapping once

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

def sign_tie_break(x: torch.Tensor) -> torch.Tensor:
    s = x.sign()
    return torch.where(s == 0, torch.ones_like(s), s)

# ----- quantization helpers -----
def quantize_input(x: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Input proxy (any shape):
      bits=1: bipolar sign {-1,+1}
      bits=2: symmetric 4-level {-1,-1/3,1/3,1} * s, with s = amax(|x|)
    """
    if bits == 1:
        return sign_tie_break(x)
    elif bits == 2:
        s = x.detach().abs().amax() + EPS
        u = torch.clamp(x / s, -1.0, 1.0)
        idx = torch.round((u + 1.0) * 1.5)   # 0..3
        levels = torch.tensor([-1.0, -1.0/3.0, 1.0/3.0, 1.0], device=x.device, dtype=x.dtype)
        q = levels[(idx.long())]
        return q * s
    else:
        raise ValueError("ACT_BITS must be 1 or 2")

def quantize_hidden_relu(h_pos: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Hidden proxy (post-ReLU, any non-neg shape):
      bits=1: gate {0,1}
      bits=2: {0, 1/3, 2/3, 1} * s, with s = amax(h_pos)
    """
    if bits == 1:
        return (h_pos > 0).to(h_pos.dtype)
    elif bits == 2:
        s = h_pos.detach().amax() + EPS
        u = torch.clamp(h_pos / s, 0.0, 1.0)
        idx = torch.round(u * 3.0)           # 0..3
        levels = torch.tensor([0.0, 1.0/3.0, 2.0/3.0, 1.0], device=h_pos.device, dtype=h_pos.dtype)
        q = levels[(idx.long())]
        return q * s
    else:
        raise ValueError("ACT_BITS must be 1 or 2")

# -----------------------------
# Model: 6-layer CNN + GAP + FC
# -----------------------------
class SimpleCNN6(nn.Module):
    """
    Channels: 1->16->16->32->32->64->64 ; all 3x3, stride=1, padding=1
    No pooling (keeps shapes simple); use GAP before FC.
    """
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16,   16, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16,   32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32,   32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(32,   64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(64,   64, kernel_size=3, stride=1, padding=1, bias=True)
        self.fc    = nn.Linear(64, num_classes)

        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: [B,1,28,28] for MNIST
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.relu(self.conv5(h4))
        h6 = F.relu(self.conv6(h5))
        gap = h6.mean(dim=(2,3))       # [B,64]
        return self.fc(gap)            # logits

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
# Client: forward-only → low-bit bundle
# returns (a0_q, [h1_q...h6_q])
# -----------------------------
@torch.no_grad()
def client_forward_collect_lowbit_cnn(x, model: SimpleCNN6, params: List[torch.Tensor], device, act_bits: int):
    load_params_into(model, params)
    x = x.to(device)                     # [B,1,28,28]
    a0_q = quantize_input(x, act_bits)   # input proxy

    a1 = model.conv1(x);  h1_pos = torch.clamp(a1, min=0.0); h1_q = quantize_hidden_relu(h1_pos, act_bits)
    a2 = model.conv2(h1_pos); h2_pos = torch.clamp(a2, min=0.0); h2_q = quantize_hidden_relu(h2_pos, act_bits)
    a3 = model.conv3(h2_pos); h3_pos = torch.clamp(a3, min=0.0); h3_q = quantize_hidden_relu(h3_pos, act_bits)
    a4 = model.conv4(h3_pos); h4_pos = torch.clamp(a4, min=0.0); h4_q = quantize_hidden_relu(h4_pos, act_bits)
    a5 = model.conv5(h4_pos); h5_pos = torch.clamp(a5, min=0.0); h5_q = quantize_hidden_relu(h5_pos, act_bits)
    a6 = model.conv6(h5_pos); h6_pos = torch.clamp(a6, min=0.0); h6_q = quantize_hidden_relu(h6_pos, act_bits)

    return a0_q, [h1_q, h2_q, h3_q, h4_q, h5_q, h6_q]

# -----------------------------
# Server: sign updates using ONLY low-bit bundle (CNN case)
# -----------------------------
def server_signed_update_from_lowbit_cnn(
    server_model: SimpleCNN6,
    server_params: List[torch.Tensor],
    y: torch.Tensor,
    a0_q: torch.Tensor, h_q_list: List[torch.Tensor],
    mom: List[torch.Tensor],
    beta_mom: float,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:

    load_params_into(server_model, server_params)
    y = y.to(device)

    # unpack
    h1_q, h2_q, h3_q, h4_q, h5_q, h6_q = h_q_list
    B, C6, H6, W6 = h6_q.shape

    # derivative masks from proxies (works for both bits)
    g1 = (h1_q > 0).to(h1_q.dtype)
    g2 = (h2_q > 0).to(h2_q.dtype)
    g3 = (h3_q > 0).to(h3_q.dtype)
    g4 = (h4_q > 0).to(h4_q.dtype)
    g5 = (h5_q > 0).to(h5_q.dtype)
    g6 = (h6_q > 0).to(h6_q.dtype)

    # -------- head forward (GAP + FC) using h6_q only --------
    gap = h6_q.mean(dim=(2,3))                 # [B,64]
    z = F.linear(gap, server_model.fc.weight, server_model.fc.bias)
    probs = F.softmax(z, dim=1)
    loss = F.nll_loss((probs + EPS).log(), y)

    # grad wrt logits
    grad_z = probs
    grad_z[torch.arange(y.size(0), device=device), y] -= 1.0
    grad_z /= y.size(0)                                 # [B,10]

    # ---- fc grads ----
    Wfc = server_model.fc.weight.detach()               # [10,64]
    gWfc = grad_z.t().matmul(gap)                       # [10,64]
    gbfc = grad_z.sum(dim=0)                            # [10]
    grad_gap = grad_z.matmul(Wfc)                       # [B,64]
    # distribute GAP gradient over spatial locations
    grad_h6 = grad_gap.view(B, C6, 1, 1).expand(-1, -1, H6, W6) / float(H6 * W6)  # [B,64,H,W]

    # ---- conv6 backward ----
    delta6 = grad_h6 * g6
    W6 = server_model.conv6.weight.detach()
    gW6 = conv2d_weight(h5_q, W6.shape, delta6, stride=1, padding=1, dilation=1, groups=1)
    gb6 = delta6.sum(dim=(0,2,3))
    grad_h5 = conv2d_input(h5_q.shape, W6, delta6, stride=1, padding=1, dilation=1, groups=1)

    # ---- conv5 ----
    delta5 = grad_h5 * g5
    W5 = server_model.conv5.weight.detach()
    gW5 = conv2d_weight(h4_q, W5.shape, delta5, stride=1, padding=1, dilation=1, groups=1)
    gb5 = delta5.sum(dim=(0,2,3))
    grad_h4 = conv2d_input(h4_q.shape, W5, delta5, stride=1, padding=1, dilation=1, groups=1)

    # ---- conv4 ----
    delta4 = grad_h4 * g4
    W4 = server_model.conv4.weight.detach()
    gW4 = conv2d_weight(h3_q, W4.shape, delta4, stride=1, padding=1, dilation=1, groups=1)
    gb4 = delta4.sum(dim=(0,2,3))
    grad_h3 = conv2d_input(h3_q.shape, W4, delta4, stride=1, padding=1, dilation=1, groups=1)

    # ---- conv3 ----
    delta3 = grad_h3 * g3
    W3 = server_model.conv3.weight.detach()
    gW3 = conv2d_weight(h2_q, W3.shape, delta3, stride=1, padding=1, dilation=1, groups=1)
    gb3 = delta3.sum(dim=(0,2,3))
    grad_h2 = conv2d_input(h2_q.shape, W3, delta3, stride=1, padding=1, dilation=1, groups=1)

    # ---- conv2 ----
    delta2 = grad_h2 * g2
    W2 = server_model.conv2.weight.detach()
    gW2 = conv2d_weight(h1_q, W2.shape, delta2, stride=1, padding=1, dilation=1, groups=1)
    gb2 = delta2.sum(dim=(0,2,3))
    grad_h1 = conv2d_input(h1_q.shape, W2, delta2, stride=1, padding=1, dilation=1, groups=1)

    # ---- conv1 ----
    delta1 = grad_h1 * g1
    W1 = server_model.conv1.weight.detach()
    gW1 = conv2d_weight(a0_q, W1.shape, delta1, stride=1, padding=1, dilation=1, groups=1)
    gb1 = delta1.sum(dim=(0,2,3))

    # ORDER must match model.parameters() exactly:
    grads = [gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4, gW5, gb5, gW6, gb6, gWfc, gbfc]

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
    if len(lrs) != len(params) or len(signs) != len(params):
        raise RuntimeError(f"Len mismatch: lrs={len(lrs)} params={len(params)} signs={len(signs)}")
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
    if len(lrs) != len(params) or len(signs) != len(params):
        raise RuntimeError(f"Len mismatch: lrs={len(lrs)} params={len(params)} signs={len(signs)}")
    with torch.no_grad():
        for i, (p, s) in enumerate(zip(params, signs)):
            p.add_(s, alpha=-lrs[i])
    return params

# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(full_model: SimpleCNN6, params: List[torch.Tensor], loader: DataLoader, device) -> float:
    full_model.eval()
    load_params_into(full_model, params)
    tot, ok = 0, 0
    for x, y in loader:
        x = x.to(device)
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
    template = SimpleCNN6().to(device)

    # Init params (ORDERED) and names
    init_params = copy_params_from(template)
    param_names = [n for n, _ in template.named_parameters()]

    # Per-parameter LR scaling by 1/sqrt(fan_in) computed FROM init_params so lengths & order match
    def per_param_fanin_from_shape(p: torch.Tensor) -> float:
        if p.ndim == 4:   # conv weight [out_c, in_c, kH, kW]
            return float(p.shape[1] * p.shape[2] * p.shape[3])
        elif p.ndim == 2: # fc weight [out, in]
            return float(p.shape[1])
        else:             # bias or scalars
            return 1.0

    layer_fanin = [max(1.0, per_param_fanin_from_shape(p)) for p in init_params]
    layer_lrs   = [BASE_LR / math.sqrt(fi) for fi in layer_fanin]
    assert len(layer_lrs) == len(init_params), "LR list must match params length"

    if DEBUG_MAP:
        print("Param ↔ LR mapping:")
        for i, ((n, p), lr) in enumerate(zip(template.named_parameters(), layer_lrs)):
            print(f"{i:02d} {n:20s} {tuple(p.shape)}  lr={lr:.3e}")

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
    client_model = SimpleCNN6().to(device)
    server_model = SimpleCNN6().to(device)

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
                a0_q, h_q_list = client_forward_collect_lowbit_cnn(x, client_model, client.params, device, ACT_BITS)

                # ---- SERVER: sign updates using ONLY low-bit & labels ----
                signs, server_slots[cid].mom, loss_val = server_signed_update_from_lowbit_cnn(
                    server_model, server_slots[cid].params, y.to(device),
                    a0_q, h_q_list,
                    server_slots[cid].mom, SERVER_BETA, device
                )
                losses.append(loss_val)

                # Defensive check once (first step)
                # if rnd == 1 and cid == 0: print("grads len ->", len(signs), "| params len ->", len(client.params))

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

        print(f"[Round {rnd:03d}] mean proxy loss: {sum(losses)/max(1,len(losses)):.4f} | "
              f"global test acc: {acc:.4f} | ACT_BITS={ACT_BITS}")

    print("Done.")

if __name__ == "__main__":
    main()
