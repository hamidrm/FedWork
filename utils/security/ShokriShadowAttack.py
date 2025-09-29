import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

class _AttackNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(1)

@torch.no_grad()
def _probs(predict_fn, x):
    out = predict_fn(x)
    if isinstance(out, np.ndarray): out = torch.from_numpy(out).to(x.device)
    if out.ndim != 2: raise RuntimeError("predict_fn must return (N,C) logits/probs")
    s = out.sum(1, keepdim=True)
    if torch.allclose(s.mean(), torch.tensor(1.0, device=x.device), atol=1e-3, rtol=1e-3) and (out >= 0).all():
        return out
    return F.softmax(out, dim=1)

class ShokriShadowAttack:
    def __init__(self, num_classes, device="cpu", hidden=128):
        self.C = num_classes
        self.device = device
        self.hidden = hidden
        self.attack = {c: _AttackNet(self.C, hidden).to(device) for c in range(num_classes)}

    @torch.no_grad()
    def _collect(self, predict_fn, loader):
        X, Y = [], []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            p = _probs(predict_fn, x)
            X.append(p.cpu())
            Y.append(y.cpu())
        return torch.cat(X, 0), torch.cat(Y, 0)

    def fit(self, shadow_triplets, epochs=10, lr=1e-3, batch_size=256):
        Zc = {c: [] for c in range(self.C)}
        Tc = {c: [] for c in range(self.C)}
        for predict_fn, member_loader, nonmember_loader in shadow_triplets:
            Xm, Ym = self._collect(predict_fn, member_loader)
            Xu, Yu = self._collect(predict_fn, nonmember_loader)
            for c in range(self.C):
                idx_m = (Ym == c)
                idx_u = (Yu == c)
                if idx_m.any():
                    Zc[c].append(Xm[idx_m])
                    Tc[c].append(torch.ones(idx_m.sum(), dtype=torch.float32))
                if idx_u.any():
                    Zc[c].append(Xu[idx_u])
                    Tc[c].append(torch.zeros(idx_u.sum(), dtype=torch.float32))
        for c in range(self.C):
            if len(Zc[c]) == 0:
                self.attack[c] = None
                continue
            X = torch.cat(Zc[c], 0).to(self.device)
            t = torch.cat(Tc[c], 0).to(self.device)
            ds = TensorDataset(X, t)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
            model = self.attack[c]
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            bce = nn.BCEWithLogitsLoss()
            model.train()
            for _ in range(epochs):
                for xb, tb in dl:
                    opt.zero_grad()
                    loss = bce(model(xb), tb)
                    loss.backward()
                    opt.step()

    @torch.no_grad()
    def predict_scores(self, target_predict_fn, loader):
        S = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            p = _probs(target_predict_fn, x)
            for c in range(self.C):
                pass
            if hasattr(y, "shape"):
                y_flat = y.view(-1)
            else:
                y_flat = y
            s = torch.zeros(x.size(0), device=self.device)
            for cls in range(self.C):
                idx = (y_flat == cls)
                if idx.any():
                    m = self.attack.get(cls, None)
                    if m is None:
                        s[idx] = 0.5
                    else:
                        s[idx] = torch.sigmoid(m(p[idx]))
            S.append(s.cpu())
        return torch.cat(S, 0).numpy()

    @torch.no_grad()
    def predict(self, target_predict_fn, loader, threshold=0.5):
        scores = self.predict_scores(target_predict_fn, loader)
        return (scores >= threshold).astype(np.int32)

    @torch.no_grad()
    def evaluate(self, target_predict_fn, member_loader, nonmember_loader, target_fprs=(0.1, 0.02, 0.01, 0.001)):
        s_m = self.predict_scores(target_predict_fn, member_loader)
        s_u = self.predict_scores(target_predict_fn, nonmember_loader)
        scores = np.concatenate([s_m, s_u])
        labels = np.concatenate([np.ones_like(s_m), np.zeros_like(s_u)])
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thr = roc_curve(labels, scores)
        res = {"auc": float(auc)}
        for f in target_fprs:
            i = np.argmin(np.abs(fpr - f))
            res[f"tpr@fpr={f}"] = float(tpr[i])
        return res
