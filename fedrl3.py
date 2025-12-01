# flsu_vww_mobilenet025_hybrid.py
# Hybrid FL on VWW (Visual Wake Words, person vs no_person, 96x96 RGB)
# Phase 1 (warm-up): ONLY high-power (HP) clients train with FP32 FedAvg.
# Phase 2 (hybrid):  HP clients keep training; MCU clients join with forward-only low-bit activations.
# Aggregation: during warm-up, average ONLY HP models; after warm-up, average HP + MCU mirrors.
# Evaluation: during warm-up, evaluate ONLY HP average; after warm-up, evaluate HP+MCU average.
#
# Data folder (ImageFolder):
#   data/vww/train/{no_person,person}/...
#   data/vww/val/{no_person,person}/...
#
# Run:
#   python flsu_vww_mobilenet025_hybrid.py

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_weight, conv2d_input
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# -----------------------------
# HYPERPARAMS
# -----------------------------
NUM_HP_CLIENTS   = 4     # high-power clients (FedAvg FP32)
NUM_MCU_CLIENTS  = 6     # forward-only low-bit clients
ROUNDS           = 80
WARMUP_ROUNDS    = 20    # ONLY HP clients in [1..WARMUP_ROUNDS]
LOCAL_STEPS      = 8     # mini-batches per client per round
BATCH_SIZE_HP    = 16
BATCH_SIZE_MCU   = 16

# HP local SGD
HP_LR            = 0.02
HP_MOM           = 0.9
HP_WD            = 1e-4

# MCU sign training
SERVER_BETA      = 0.95  # server momentum before sign
CLIENT_MOM       = 0.0   # client velocity (enable later if stable)
BASE_LR          = 1e-2  # per-parameter scaled by 1/sqrt(fan_in)
AVG_PERIOD       = 20    # rounds between global averaging/broadcast
ACT_BITS         = 2     # 1 or 2 for MCU activations

SEED             = 0
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR         = "./data/vww"
EPS              = 1e-8
PRINT_MAP        = False

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=0):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def copy_params_from(mod: nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in mod.parameters()]

def load_params_into(mod: nn.Module, params: List[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, v in zip(mod.parameters(), params):
            p.copy_(v)

def model_param_list(mod: nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in mod.parameters()]

def apply_params_(mod: nn.Module, src: List[torch.Tensor]):
    load_params_into(mod, src)

def sign_tie_break(x: torch.Tensor) -> torch.Tensor:
    s = x.sign()
    return torch.where(s == 0, torch.ones_like(s), s)

# ----- quantization helpers (channel-wise scales) -----
def quantize_input(x: torch.Tensor, bits: int) -> torch.Tensor:
    # x: [B,C,H,W]
    if bits == 1:
        return torch.where(x == 0, torch.ones_like(x), x.sign())
    elif bits == 2:
        s = x.detach().abs().amax(dim=(0,2,3), keepdim=True) + EPS
        u = torch.clamp(x / s, -1.0, 1.0)
        idx = torch.round((u + 1.0) * 1.5)  # 0..3
        levels = torch.tensor([-1.0, -1/3, 1/3, 1.0], device=x.device, dtype=x.dtype)
        q = levels[idx.long()]
        return q * s
    else:
        raise ValueError("ACT_BITS must be 1 or 2")

def quantize_hidden_relu(h_pos: torch.Tensor, bits: int) -> torch.Tensor:
    # h_pos >= 0, [B,C,H,W]
    if bits == 1:
        return (h_pos > 0).to(h_pos.dtype)
    elif bits == 2:
        s = h_pos.detach().amax(dim=(0,2,3), keepdim=True) + EPS
        u = torch.clamp(h_pos / s, 0.0, 1.0)
        idx = torch.round(u * 3.0)  # 0..3
        levels = torch.tensor([0.0, 1/3, 2/3, 1.0], device=h_pos.device, dtype=h_pos.dtype)
        q = levels[idx.long()]
        return q * s
    else:
        raise ValueError("ACT_BITS must be 1 or 2")

# -----------------------------
# Data (VWW via ImageFolder)
# -----------------------------
def make_vww_loaders(
    data_dir=DATA_DIR,
    batch_size_hp=BATCH_SIZE_HP,
    batch_size_mcu=BATCH_SIZE_MCU,
    num_hp=NUM_HP_CLIENTS,
    num_mcu=NUM_MCU_CLIENTS,
):
    num_clients = num_hp + num_mcu
    tfm_train = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = ImageFolder(root=f"{data_dir}/train", transform=tfm_train)
    val_ds   = ImageFolder(root=f"{data_dir}/val",   transform=tfm_val)

    idxs = list(range(len(train_ds))); random.shuffle(idxs)
    per = len(train_ds) // num_clients
    subsets = []
    for k in range(num_clients):
        lo, hi = k*per, (k+1)*per if k < num_clients-1 else len(train_ds)
        subsets.append(Subset(train_ds, idxs[lo:hi]))

    hp_loaders  = [DataLoader(s, batch_size=batch_size_hp,  shuffle=True, num_workers=0, pin_memory=False)
                   for s in subsets[:num_hp]]
    mcu_loaders = [DataLoader(s, batch_size=batch_size_mcu, shuffle=True, num_workers=0, pin_memory=False)
                   for s in subsets[num_hp:]]
    val_loader  = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=False)
    return hp_loaders, mcu_loaders, val_loader

# -----------------------------
# Model: MobileNetV1-0.25 (BN-free, ReLU)
# -----------------------------
class DWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=True)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.zeros_(self.dw.bias)
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity="relu")
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        x = F.relu(self.dw(x))
        x = F.relu(self.pw(x))
        return x

class MobileNetV1_025_Lite(nn.Module):
    def __init__(self, num_classes=2, alpha=0.25):
        super().__init__()
        def c(ch): return max(8, int(ch * alpha))
        self.conv1 = nn.Conv2d(3, c(32), 3, 2, 1, bias=True)  # 96->48
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        self.blocks = nn.ModuleList([
            DWBlock(c(32),  c(64),  1),
            DWBlock(c(64),  c(128), 2),
            DWBlock(c(128), c(128), 1),
            DWBlock(c(128), c(256), 2),
            DWBlock(c(256), c(256), 1),
            DWBlock(c(256), c(512), 2),
            DWBlock(c(512), c(512), 1),
            DWBlock(c(512), c(512), 1),
            DWBlock(c(512), c(512), 1),
            DWBlock(c(512), c(512), 1),
            DWBlock(c(512), c(512), 1),
            DWBlock(c(512), c(1024),2),
            DWBlock(c(1024),c(1024),1),
        ])
        self.fc = nn.Linear(c(1024), 2)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h0 = F.relu(self.conv1(x))         # [B,C1,48,48]
        h = h0
        for blk in self.blocks:
            h = blk(h)
        h = F.adaptive_avg_pool2d(h, (1,1)).view(x.size(0), -1)
        return self.fc(h)

# -----------------------------
# Federated state (MCU)
# -----------------------------
@dataclass
class ClientStateMCU:
    params: List[torch.Tensor]
    velocity: List[torch.Tensor] = field(default_factory=list)

@dataclass
class ServerSlot:
    params: List[torch.Tensor]
    mom:    List[torch.Tensor]

# -----------------------------
# MCU client forward-only bundle
# -----------------------------
@torch.no_grad()
def client_forward_collect_lowbit_mnv1(x, model: MobileNetV1_025_Lite,
                                       params: List[torch.Tensor], device, act_bits: int):
    load_params_into(model, params)
    x = x.to(device)                     # [B,3,96,96]
    a0_q = quantize_input(x, act_bits)

    a1 = model.conv1(x)
    h0_pos = torch.clamp(a1, min=0.0)
    h0_q = quantize_hidden_relu(h0_pos, act_bits)

    h = h0_pos
    h_dw_q_list = []
    h_pw_q_list = []
    for blk in model.blocks:
        a_dw = blk.dw(h)
        h_dw_pos = torch.clamp(a_dw, min=0.0)
        h_dw_q = quantize_hidden_relu(h_dw_pos, act_bits)

        a_pw = blk.pw(h_dw_pos)
        h_pw_pos = torch.clamp(a_pw, min=0.0)
        h_pw_q = quantize_hidden_relu(h_pw_pos, act_bits)

        h = h_pw_pos
        h_dw_q_list.append(h_dw_q)
        h_pw_q_list.append(h_pw_q)

    return a0_q, h0_q, h_dw_q_list, h_pw_q_list

# -----------------------------
# Server: sign updates from MCU bundles
# -----------------------------
def server_signed_update_from_lowbit_mnv1(
    server_model: MobileNetV1_025_Lite,
    server_params: List[torch.Tensor],
    y: torch.Tensor,
    a0_q: torch.Tensor, h0_q: torch.Tensor,
    h_dw_q_list: List[torch.Tensor], h_pw_q_list: List[torch.Tensor],
    mom: List[torch.Tensor],
    beta_mom: float,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:

    load_params_into(server_model, server_params)
    y = y.to(device)

    h_last = h_pw_q_list[-1]
    gap = F.adaptive_avg_pool2d(h_last, (1,1)).view(h_last.size(0), -1)
    z = F.linear(gap, server_model.fc.weight, server_model.fc.bias)
    probs = F.softmax(z, dim=1)
    loss = F.nll_loss((probs + EPS).log(), y)

    grad_z = probs
    grad_z[torch.arange(y.size(0), device=device), y] -= 1.0
    grad_z /= y.size(0)

    Wfc = server_model.fc.weight.detach()
    gWfc = grad_z.t().matmul(gap)
    gbfc = grad_z.sum(dim=0)
    grad_gap = grad_z.matmul(Wfc)
    B, C, H, W = h_last.shape
    grad_h = grad_gap.view(B, C, 1, 1).expand(-1, -1, H, W) / float(H * W)

    grads: List[torch.Tensor] = []
    delta = grad_h

    for bi in reversed(range(len(server_model.blocks))):
        blk = server_model.blocks[bi]
        h_pw_q = h_pw_q_list[bi]; h_dw_q = h_dw_q_list[bi]
        gate_pw = (h_pw_q > 0).to(h_pw_q.dtype)
        gate_dw = (h_dw_q > 0).to(h_dw_q.dtype)

        delta_pw = delta * gate_pw
        Wpw = blk.pw.weight.detach()
        gWpw = conv2d_weight(h_dw_q, Wpw.shape, delta_pw, stride=1, padding=0, dilation=1, groups=1)
        gbpw = delta_pw.sum(dim=(0,2,3))
        grad_hdw = conv2d_input(h_dw_q.shape, Wpw, delta_pw, stride=1, padding=0, dilation=1, groups=1)

        delta_dw = grad_hdw * gate_dw
        Wdw = blk.dw.weight.detach()
        stride_dw = blk.dw.stride; pad_dw = blk.dw.padding; groups_dw = blk.dw.groups
        inp_prev = h0_q if bi == 0 else h_pw_q_list[bi-1]
        gWdw = conv2d_weight(inp_prev, Wdw.shape, delta_dw,
                             stride=stride_dw, padding=pad_dw, dilation=1, groups=groups_dw)
        gbdw = delta_dw.sum(dim=(0,2,3))
        grad_inp_prev = conv2d_input(inp_prev.shape, Wdw, delta_dw,
                                     stride=stride_dw, padding=pad_dw, dilation=1, groups=groups_dw)

        grads.extend([gWdw, gbdw, gWpw, gbpw])
        delta = grad_inp_prev

    gate0 = (h0_q > 0).to(h0_q.dtype)
    delta0 = delta * gate0
    W1 = server_model.conv1.weight.detach()
    stride1 = server_model.conv1.stride; pad1 = server_model.conv1.padding
    gW1 = conv2d_weight(a0_q, W1.shape, delta0, stride=stride1, padding=pad1, dilation=1, groups=1)
    gb1 = delta0.sum(dim=(0,2,3))

    grads_ordered: List[torch.Tensor] = [gW1, gb1]
    per_block = 4
    blocks_grads = [grads[i:i+per_block] for i in range(0, len(grads), per_block)]
    blocks_grads.reverse()
    for gWdw, gbdw, gWpw, gbpw in blocks_grads:
        grads_ordered.extend([gWdw, gbdw, gWpw, gbpw])
    grads_ordered.extend([gWfc, gbfc])

    signs = []
    with torch.no_grad():
        for g, m in zip(grads_ordered, mom):
            m.mul_(beta_mom).add_(g, alpha=1.0 - beta_mom)
            s = m.sign(); s[s == 0] = 1.0
            signs.append(s.detach())
    return signs, mom, float(loss.item())

# -----------------------------
# MCU apply / mirror sign step
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
def evaluate(eval_model: nn.Module, params: List[torch.Tensor], loader: DataLoader, device) -> float:
    eval_model.eval()
    load_params_into(eval_model, params)
    tot, ok = 0, 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = eval_model(x)
        ok += (logits.argmax(1) == y).sum().item()
        tot += y.numel()
    return ok / tot

# -----------------------------
# Main training loop (hybrid with warm-up)
# -----------------------------
def main():
    set_seed(SEED)
    device = torch.device(DEVICE)

    # Data
    hp_loaders, mcu_loaders, val_loader = make_vww_loaders()

    # Template + per-parameter LR (for MCU sign steps)
    template = MobileNetV1_025_Lite().to(device)
    init_params = copy_params_from(template)

    def fanin(p: torch.Tensor) -> float:
        if p.ndim == 4:   # conv weight [out_c, in_c, kH, kW]
            return float(p.shape[1] * p.shape[2] * p.shape[3])
        elif p.ndim == 2: # fc weight [out, in]
            return float(p.shape[1])
        else:             # bias
            return 1.0
    layer_fanin = [max(1.0, fanin(p)) for p in init_params]
    layer_lrs   = []
    for p, fi in zip(template.parameters(), layer_fanin):
        if p.ndim == 1:  # bias: slightly larger step
            layer_lrs.append(3.0 * BASE_LR)
        else:
            layer_lrs.append(BASE_LR / math.sqrt(fi))

    # HP clients: models + optimizers
    hp_models: List[MobileNetV1_025_Lite] = [
        MobileNetV1_025_Lite().to(device) for _ in range(NUM_HP_CLIENTS)
    ]
    for m in hp_models:
        apply_params_(m, init_params)
    hp_opts = [torch.optim.SGD(m.parameters(), lr=HP_LR, momentum=HP_MOM, weight_decay=HP_WD)
               for m in hp_models]
    criterion = nn.CrossEntropyLoss()

    # MCU clients: params + server mirrors + momenta
    mcu_clients: List[ClientStateMCU] = [
        ClientStateMCU(params=[p.clone() for p in init_params],
                       velocity=[torch.zeros_like(p) for p in init_params])
        for _ in range(NUM_MCU_CLIENTS)
    ]
    server_slots: List[ServerSlot] = [
        ServerSlot(params=[p.clone() for p in init_params],
                   mom=[torch.zeros_like(p) for p in init_params])
        for _ in range(NUM_MCU_CLIENTS)
    ]

    # Working server/client models (shared buffers)
    server_model = MobileNetV1_025_Lite().to(device)
    mcu_work_model = MobileNetV1_025_Lite().to(device)
    eval_model = MobileNetV1_025_Lite().to(device)  # for eval only

    # Report parameter count
    print(f"Model params: {sum(p.numel() for p in init_params):,}")

    for rnd in range(1, ROUNDS + 1):
        losses_hp = []
        losses_mcu = []

        # ------------- HP phase (FedAvg local SGD) -------------
        for model, opt, loader in zip(hp_models, hp_opts, hp_loaders):
            it = iter(loader)
            model.train()
            for _ in range(LOCAL_STEPS):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader); x, y = next(it)
                x = x.to(device); y = y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt.step()
                losses_hp.append(loss.item())

        # ------------- MCU phase (only after warm-up) -------------
        if rnd > WARMUP_ROUNDS:
            for cid, (client, loader) in enumerate(zip(mcu_clients, mcu_loaders)):
                it = iter(loader)
                for _ in range(LOCAL_STEPS):
                    try:
                        x, y = next(it)
                    except StopIteration:
                        it = iter(loader); x, y = next(it)

                    # CLIENT forward-only bundle (low-bit)
                    a0_q, h0_q, h_dw_q_list, h_pw_q_list = client_forward_collect_lowbit_mnv1(
                        x, mcu_work_model, client.params, device, ACT_BITS
                    )
                    # SERVER: signs
                    signs, server_slots[cid].mom, loss_val = server_signed_update_from_lowbit_mnv1(
                        server_model, server_slots[cid].params, y.to(device),
                        a0_q, h0_q, h_dw_q_list, h_pw_q_list,
                        server_slots[cid].mom, SERVER_BETA, device
                    )
                    losses_mcu.append(loss_val)
                    # CLIENT apply
                    client.params, client.velocity = apply_sign_step(
                        client.params, signs, layer_lrs, client.velocity, CLIENT_MOM
                    )
                    # SERVER mirror
                    server_slots[cid].params = mirror_sign_step(server_slots[cid].params, signs, layer_lrs)

        # ------------- Aggregation & broadcast -------------
        if rnd % AVG_PERIOD == 0:
            with torch.no_grad():
                stacks = []
                # Always average trained HP models
                for m in hp_models:
                    stacks.append(model_param_list(m))

                if rnd > WARMUP_ROUNDS:
                    # Hybrid phase: include MCU mirrors
                    for slot in server_slots:
                        stacks.append([p.clone() for p in slot.params])

                # Compute average
                avg_params = []
                for idx in range(len(init_params)):
                    avg_params.append(torch.stack([st[idx] for st in stacks], dim=0).mean(dim=0))

                # Broadcast to HP
                for m in hp_models:
                    apply_params_(m, avg_params)

                if rnd > WARMUP_ROUNDS:
                    # Broadcast to MCU (hybrid ongoing)
                    for client in mcu_clients:
                        for dst, src in zip(client.params, avg_params): dst.copy_(src)
                        client.velocity = [torch.zeros_like(p) for p in client.params]
                    for slot in server_slots:
                        for dst, src in zip(slot.params, avg_params): dst.copy_(src)
                        slot.mom = [torch.zeros_like(p) for p in slot.mom]
                else:
                    # During warm-up, OPTIONAL: if this AVG aligns with the end of warm-up,
                    # pre-seed MCUs with the HP global so they join from a good starting point.
                    if rnd == WARMUP_ROUNDS:
                        for client in mcu_clients:
                            for dst, src in zip(client.params, avg_params): dst.copy_(src)
                            client.velocity = [torch.zeros_like(p) for p in client.params]
                        for slot in server_slots:
                            for dst, src in zip(slot.params, avg_params): dst.copy_(src)
                            slot.mom = [torch.zeros_like(p) for p in slot.mom]

        # ------------- Evaluate global -------------
        with torch.no_grad():
            if rnd <= WARMUP_ROUNDS:
                # Eval ONLY HP average during warm-up
                stacks_eval = [model_param_list(m) for m in hp_models]
            else:
                # Hybrid eval: HP + MCU mirrors
                stacks_eval = [model_param_list(m) for m in hp_models] + [slot.params for slot in server_slots]

            glob = []
            for idx in range(len(init_params)):
                glob.append(torch.stack([st[idx] for st in stacks_eval], dim=0).mean(dim=0))
        acc = evaluate(eval_model, glob, val_loader, device)

        # Print
        phase = 'warmup' if rnd <= WARMUP_ROUNDS else 'hybrid'
        hp_str  = f"hp_loss={sum(losses_hp)/max(1,len(losses_hp)):.4f}"
        mcu_str = (f" | mcu_loss={sum(losses_mcu)/max(1,len(losses_mcu)):.4f}"
                   if rnd > WARMUP_ROUNDS and len(losses_mcu) else "")
        print(f"[Round {rnd:03d}] {hp_str}{mcu_str} | val acc: {acc:.4f} | ACT_BITS={ACT_BITS} | phase={phase}")

    print("Done.")

if __name__ == "__main__":
    main()
