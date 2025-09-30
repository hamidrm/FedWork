import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

class YeomBlackBoxAttack:
    def __init__(self, predict_fn, device="cpu"):
        self.predict_fn = predict_fn
        self.device = device
        self.tau = None

    @torch.no_grad()
    def _probs(self, x):
        out = self.predict_fn(x)
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out).to(x.device)
        if out.ndim != 2:
            raise RuntimeError("predict_fn must return (N,C) logits or probs")
        s = out.sum(1, keepdim=True)
        if torch.allclose(s.mean(), torch.tensor(1.0, device=x.device), atol=1e-3, rtol=1e-3) and (out >= 0).all():
            return out
        return F.softmax(out, dim=1)

    @torch.no_grad()
    def _per_sample_ce(self, loader):
        losses = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            p = self._probs(x)
            p_y = p.gather(1, y.view(-1, 1)).clamp_min(1e-12).squeeze(1)
            losses.append((-p_y.log()).cpu())
        return torch.cat(losses, 0).numpy()

    def fit(self, member_calib=None, nonmember_calib=None, known_train_loss=None):
        if known_train_loss is not None:
            self.tau = float(known_train_loss)
            return self
        if member_calib is not None and nonmember_calib is None:
            m = self._per_sample_ce(member_calib)
            self.tau = float(m.mean())
            return self
        if member_calib is not None and nonmember_calib is not None:
            m = self._per_sample_ce(member_calib)
            u = self._per_sample_ce(nonmember_calib)
            cand = np.unique(np.concatenate([m, u]))
            best, best_tau = -1.0, None
            for t in cand:
                tpr = (m <= t).mean()
                fpr = (u <= t).mean()
                adv = tpr - fpr
                if adv > best:
                    best, best_tau = adv, t
            self.tau = float(best_tau)
            return self
        raise ValueError("Provide known_train_loss or member_calib (optionally nonmember_calib).")

    @torch.no_grad()
    def predict_scores(self, loader):
        return -self._per_sample_ce(loader)

    @torch.no_grad()
    def predict(self, loader):
        if self.tau is None:
            raise RuntimeError("Call fit(...) first.")
        loss = self._per_sample_ce(loader)
        return (loss <= self.tau).astype(np.int32)

    @torch.no_grad()
    def evaluate(self, member_loader, nonmember_loader, target_fprs=(0.1, 0.02, 0.01, 0.001)):
        s_m = self.predict_scores(member_loader)
        s_u = self.predict_scores(nonmember_loader)
        scores = np.concatenate([s_m, s_u])
        labels = np.concatenate([np.ones_like(s_m), np.zeros_like(s_u)])
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thr = roc_curve(labels, scores)
        res = {"auc": float(auc)}
        for f in target_fprs:
            i = np.argmin(np.abs(fpr - f))
            res[f"tpr@fpr={f}"] = float(tpr[i])
        if self.tau is not None:
            loss_m = -s_m
            loss_u = -s_u
            tpr_tau = float((loss_m <= self.tau).mean())
            fpr_tau = float((loss_u <= self.tau).mean())
            res.update({"tau": float(self.tau), "tpr@tau": tpr_tau, "fpr@tau": fpr_tau, "advantage@tau": tpr_tau - fpr_tau})
        return res
