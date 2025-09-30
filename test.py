# fl_dirichlet_sequential.py
# A minimal, reproducible FL simulator with Dirichlet partitioning and sequential clients.

import os, math, json, hashlib, random, argparse
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS; set before torch import
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----------------------------- determinism helpers -----------------------------

BASE_SEED = 12345

def seed_everything(seed=BASE_SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch may not have it; that's fine.
        pass
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def _hash_seed(*xs):
    h = hashlib.blake2b(digest_size=8)
    for x in xs: h.update(str(x).encode())
    return int.from_bytes(h.digest(), "little")

def dl_generator(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g

# ----------------------------- datasets & models ------------------------------

def make_datasets(name):
    name = name.upper()
    if name == "MNIST":
        tf_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        tf_test  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train = datasets.MNIST("./data", train=True,  download=True, transform=tf_train)
        test  = datasets.MNIST("./data", train=False, download=True, transform=tf_test)
        n_classes = 10
    elif name == "CIFAR10":
        # No random crop/flip for reproducibility. (Add them only if you accept variance.)
        stats = ((0.49139968, 0.48215841, 0.44653091),
                 (0.24703223, 0.24348513, 0.26158784))
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train = datasets.CIFAR10("./data", train=True,  download=True, transform=tf)
        test  = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
        n_classes = 10
    elif name == "CIFAR100":
        stats = ((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        train = datasets.CIFAR100("./data", train=True,  download=True, transform=tf)
        test  = datasets.CIFAR100("./data", train=False, download=True, transform=tf)
        n_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return train, test, n_classes

class SmallCNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64*14*14, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SmallCNN_CIFAR(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)  # 16x16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # 8x8
        self.fc1   = nn.Linear(128*8*8, 256)
        self.fc2   = nn.Linear(256, n_classes)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x)); x = F.relu(self.conv4(x)); x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def make_model(dataset_name, n_classes):
    if dataset_name.upper() == "MNIST":
        return SmallCNN_MNIST()
    else:
        return SmallCNN_CIFAR(n_classes=n_classes)

# ----------------------------- dirichlet partition ----------------------------

def dirichlet_partitions(labels, n_clients, alpha, seed=BASE_SEED, min_per_client=10):
    """Return list of index lists, one per client."""
    rng = np.random.default_rng(_hash_seed(seed, "dirichlet", n_clients, alpha))
    labels = np.array(labels)
    classes = np.unique(labels)
    idx_by_cls = [np.where(labels == c)[0] for c in classes]
    for arr in idx_by_cls:
        rng.shuffle(arr)  # deterministic shuffle per class

    parts = [[] for _ in range(n_clients)]
    for c, idxs in enumerate(idx_by_cls):
        if len(idxs) == 0: continue
        # sample proportions for this class across clients
        p = rng.dirichlet(alpha * np.ones(n_clients))
        # turn into sizes that sum to len(idxs)
        sizes = (p * len(idxs)).astype(int)
        # fix rounding drift:
        while sizes.sum() < len(idxs): sizes[rng.integers(0, n_clients)] += 1
        while sizes.sum() > len(idxs): 
            j = rng.integers(0, n_clients)
            if sizes[j] > 0: sizes[j] -= 1
        start = 0
        for k, sz in enumerate(sizes):
            if sz > 0:
                parts[k].extend(idxs[start:start+sz].tolist())
                start += sz

    # ensure minimum per client (very small alpha can starve some)
    # simple repair: borrow from the largest client
    for k in range(n_clients):
        if len(parts[k]) < min_per_client:
            need = min_per_client - len(parts[k])
            donor = np.argmax([len(p) for p in parts])
            moved = parts[donor][:need]
            parts[k].extend(moved)
            del parts[donor][:need]

    # final per-client shuffle for training order (deterministic)
    out = []
    for k in range(n_clients):
        idxs = np.array(parts[k])
        rng_k = np.random.default_rng(_hash_seed(seed, "client", k))
        rng_k.shuffle(idxs)
        out.append(idxs.tolist())
    return out

# ----------------------------- loaders & eval --------------------------------

def make_client_loader(train_dataset, indices, batch_size, seed):
    subset = Subset(train_dataset, indices)
    g = dl_generator(seed)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

def make_test_loader(test_dataset, batch_size):
    # no generator needed: shuffle=False
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return loss_sum / max(total,1), correct / max(total,1)

# ----------------------------- training utils --------------------------------

def get_named_params(model):
    return dict(model.named_parameters())

def clone_state_dict(model, device="cpu", dtype=torch.float64):
    return {k: v.detach().to(device=device, dtype=dtype).clone() for k, v in model.state_dict().items()}

def apply_update(model, delta, device):
    with torch.no_grad():
        sd = model.state_dict()
        for k in sorted(sd.keys()):
            sd[k].add_(delta[k].to(sd[k].device, dtype=sd[k].dtype))
        model.load_state_dict(sd, strict=True)

def client_update(global_model, loader, epochs, lr, device):
    # train a local copy and return delta = local - global (in float64 on CPU)
    local = type(global_model)()  # fresh init
    local.load_state_dict(global_model.state_dict(), strict=True)
    local.to(device)
    local.train()
    opt = torch.optim.SGD(local.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    n_samples = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = local(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            n_samples += y.numel()

    # delta = local - global
    with torch.no_grad():
        delta = {}
        for (name, p_g), p_l in zip(global_model.named_parameters(), local.parameters()):
            d = (p_l.detach().to("cpu", torch.float64) - p_g.detach().to("cpu", torch.float64))
            delta[name] = d
    return n_samples, delta

def aggregate(global_model, client_deltas, client_sizes, weighting="size", gamma=1.0):
    # weighting ∈ {"uniform","size","power"}; for "power", weight ~ (size^gamma)
    names = sorted(get_named_params(global_model).keys())
    acc = {k: torch.zeros_like(client_deltas[0][k], device="cpu", dtype=torch.float64) for k in names}

    if weighting == "uniform":
        weights = [1.0 for _ in client_sizes]
    elif weighting == "size":
        weights = [float(s) for s in client_sizes]
    elif weighting == "power":
        weights = [float(s)**float(gamma) for s in client_sizes]
    else:
        raise ValueError("weighting must be one of {'uniform','size','power'}")

    wsum = sum(weights) if sum(weights) > 0 else 1.0
    for w, delta in zip(weights, client_deltas):
        for k in names:
            acc[k].add_(delta[k], alpha=(w/wsum))

    apply_update(global_model, acc, device="cpu")  # applies on correct device later
    return acc

# ----------------------------- main FL loop ----------------------------------

def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    train_ds, test_ds, n_classes = make_datasets(args.dataset)
    model = make_model(args.dataset, n_classes).to(device)

    # partitions & loaders
    labels = train_ds.targets if isinstance(train_ds.targets, list) else train_ds.targets.tolist() if hasattr(train_ds.targets, "tolist") else train_ds.targets
    parts = dirichlet_partitions(labels, args.clients, args.alpha, seed=args.seed, min_per_client=max(2,args.batch_size))
    client_loaders = []
    for cid, idxs in enumerate(parts):
        g_seed = _hash_seed(args.seed, "dl", cid)
        client_loaders.append(make_client_loader(train_ds, idxs, args.batch_size, g_seed))
    test_loader = make_test_loader(test_ds, args.test_batch_size)

    # save partitions for audit
    if args.save_partitions:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"partitions_seed{args.seed}.json"), "w") as f:
            json.dump({str(i): parts[i] for i in range(len(parts))}, f)

    m = max(1, int(args.frac * args.clients))  # clients per round
    print(f"Device: {device}, Rounds: {args.rounds}, Clients: {args.clients}, m={m}, Alpha={args.alpha}")

    # initial eval
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[Round 0] test_loss={test_loss:.4f}, test_acc={test_acc*100:.2f}%")

    for rnd in range(1, args.rounds+1):
        # deterministic client sampling
        rng = np.random.default_rng(_hash_seed(args.seed, "sample", rnd))
        selected = sorted(rng.choice(args.clients, size=m, replace=False).tolist())

        client_sizes, client_deltas = [], []
        for cid in selected:  # SEQUENTIAL clients
            n, delta = client_update(model, client_loaders[cid], args.local_epochs, args.lr, device)
            client_sizes.append(n); client_deltas.append(delta)

        # move model to CPU for stable aggregation, then back
        model_cpu = type(model)()
        model_cpu.load_state_dict(model.state_dict(), strict=True)
        model_cpu.to("cpu")
        aggregate(model_cpu, client_deltas, client_sizes, weighting=args.weighting, gamma=args.gamma)
        model.load_state_dict(model_cpu.state_dict(), strict=True)
        model.to(device)

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"[Round {rnd}] test_loss={test_loss:.4f}, test_acc={test_acc*100:.2f}%")

    # final save
    if args.save_model:
        os.makedirs(args.out, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.out, f"model_{args.dataset}_seed{args.seed}.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST","CIFAR10","CIFAR100"])
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration; lower = more non-IID")
    p.add_argument("--frac", type=float, default=0.1, help="fraction of clients per round")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--test_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weighting", type=str, default="size", choices=["uniform","size","power"])
    p.add_argument("--gamma", type=float, default=0.5, help="power weighting exponent if weighting=power")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--save_partitions", action="store_true")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--out", type=str, default="./out")
    args = p.parse_args()
    main(args)
