from typing import Dict, Any

import torch

from core.FederatedLearningClass import FederatedLearningClass
from utils.logger import *


class FedVote(FederatedLearningClass):
    """
    Vanilla binary FedVote + native FedWork diagnostics.

    Expected architecture:
        WeightMode=2
        WeightScale=1.5
        ActivationMode=1
        BNMode=1
        FreezeFinalLayer=1

    This class keeps the FedVote update itself unchanged:
        local latent h
        -> tanh(k*h)
        -> stochastic binary rounding
        -> uniform soft vote p
        -> h_next = atanh(2p-1)/k
    """

    def __init__(self, method_name, fl_context, method_args):
        super().__init__(method_name, fl_context, method_args)

        self.contributors_percent = (
            float(self.get_arg(int, "contributors_percent", 100)) / 100.0
        )
        self.normalization_scale = self.get_arg(
            float, "normalization_scale", 1.5
        )
        # Paper Appendix hyperparameter: p_min=0.001, p_max=1-p_min.
        # Do not replace this with the public-repository 1e-4 normalized-weight
        # clamp; the released code and the paper differ on that detail.
        self.p_min = self.get_arg(float, "p_min", 0.001)

        if not (0.0 < self.p_min < 0.5):
            raise ValueError("FedVote requires 0 < p_min < 0.5")

        self.p_max = 1.0 - self.p_min

        self.diagnostics = bool(
            self.get_arg(int, "diagnostics", 1)
        )
        self.diagnostic_every = max(
            1, self.get_arg(int, "diagnostic_every", 1)
        )

        self._server_round = 0
        self._last_soft_vote = {}
        self._last_hard_vote = {}
        self._last_tie_mask = {}
        self._eval_tie_restore = {}

        self._log(
            "initialized: "
            f"k={self.normalization_scale}, "
            f"p_min={self.p_min}, "
            f"contributors={self.contributors_percent:.3f}"
        )

    # ------------------------------------------------------------------
    # Native FedWork logger
    # ------------------------------------------------------------------
    def _log(self, message, level="info"):
        text = f"[{self.get_name()}][DIAG] {message}"

        try:
            if level == "debug":
                logger.log_debug(text)
            elif level == "warning":
                logger.log_warning(text)
            elif level == "error":
                logger.log_error(text)
            else:
                logger.log_info(text)
            return
        except Exception:
            # Last-resort fallback only; normal FedWork runs should use logger.
            print(text, flush=True)

    def _diag_enabled_this_round(self):
        if not self.diagnostics:
            return False
        if self._server_round <= 3:
            return True
        return self._server_round % self.diagnostic_every == 0

    # ------------------------------------------------------------------
    # Framework API
    # ------------------------------------------------------------------
    def get_name(self):
        return self.method_name

    def select_clients_to_train(self, all_clients):
        if self.contributors_percent >= 1.0:
            self.num_of_contributor_nodes = len(all_clients)
            return all_clients
        return self.select_random_clients(
            all_clients, self.contributors_percent
        )

    # ------------------------------------------------------------------
    # Parameter classification
    # ------------------------------------------------------------------
    @staticmethod
    def _is_vote_weight(key: str, tensor: torch.Tensor) -> bool:
        # fc3 is the floating-point frozen classifier.
        if key == "fc3.weight":
            return False

        return (
            key.endswith(".weight")
            and torch.is_tensor(tensor)
            and tensor.ndim in (2, 4)
        )

    # ------------------------------------------------------------------
    # FedVote math
    # ------------------------------------------------------------------
    def _normalize(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.normalization_scale * h)

    def _p_to_h(self, p: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, self.p_min, self.p_max)
        x = torch.clamp(
            2.0 * p - 1.0,
            -1.0 + 1e-7,
            1.0 - 1e-7,
        )
        return torch.atanh(x) / self.normalization_scale

    @torch.no_grad()
    def _stochastic_round(self, h: torch.Tensor) -> torch.Tensor:
        w_tilde = self._normalize(h.float())
        p_plus = torch.clamp(
            (w_tilde + 1.0) * 0.5,
            0.0,
            1.0,
        )
        rnd = torch.rand_like(p_plus)
        return torch.where(
            rnd < p_plus,
            torch.ones_like(p_plus),
            -torch.ones_like(p_plus),
        )

    @torch.no_grad()
    def _hard_vote_from_p(self, p_raw: torch.Tensor):
        """Return the paper FedVote hard vote and the exact tie mask.

        Vanilla FedVote broadcasts the soft vote probability p for the next
        local optimization round, but defines the corresponding hard global
        binary weight by plurality vote.  The paper explicitly breaks exact
        ties at random.  With an even number of clients (e.g. our 10-client
        controlled benchmark), ties are possible and must not be mapped
        deterministically to +1.
        """
        hard = torch.where(
            p_raw > 0.5,
            torch.ones_like(p_raw),
            -torch.ones_like(p_raw),
        )
        tie_mask = p_raw == 0.5

        n_ties = int(tie_mask.sum().item())
        if n_ties:
            random_sign = torch.randint(
                low=0,
                high=2,
                size=(n_ties,),
                device=p_raw.device,
                dtype=torch.int64,
            ).to(dtype=p_raw.dtype)
            random_sign.mul_(2.0).sub_(1.0)
            hard[tie_mask] = random_sign

        return hard, tie_mask

    # ------------------------------------------------------------------
    # 1-bit codec
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_binary_tensor(w: torch.Tensor) -> Dict[str, Any]:
        bits = (
            w.detach()
            .reshape(-1)
            .cpu()
            .gt(0)
            .to(torch.uint8)
        )

        n = int(bits.numel())
        pad = (-n) % 8

        if pad:
            bits = torch.cat(
                [
                    bits,
                    torch.zeros(pad, dtype=torch.uint8),
                ],
                dim=0,
            )

        bits = bits.view(-1, 8)
        shifts = (
            2 ** torch.arange(8, dtype=torch.int64)
        ).view(1, 8)

        packed = (
            bits.to(torch.int64) * shifts
        ).sum(dim=1).to(torch.uint8)

        return {
            "shape": tuple(w.shape),
            "numel": n,
            "data": packed,
        }

    @staticmethod
    def _unpack_binary_tensor(packet: Dict[str, Any]) -> torch.Tensor:
        packed = (
            packet["data"]
            .detach()
            .cpu()
            .to(torch.int64)
            .reshape(-1, 1)
        )

        shifts = torch.arange(
            8, dtype=torch.int64
        ).view(1, 8)

        bits = (
            (packed >> shifts) & 1
        ).to(torch.float32).reshape(-1)

        bits = bits[: int(packet["numel"])]

        return (
            bits.mul(2.0)
            .sub(1.0)
            .reshape(tuple(packet["shape"]))
        )

    # ------------------------------------------------------------------
    # Client -> server
    # ------------------------------------------------------------------
    @torch.no_grad()
    def pack_client_model(self, raw_model, global_model, id):
        binary = {}

        for key, tensor in raw_model.items():
            if self._is_vote_weight(key, tensor):
                sampled = self._stochastic_round(tensor)
                binary[key] = self._pack_binary_tensor(sampled)

        return {
            "codec": "fedvote-binary-v4",
            "binary": binary,
        }

    @torch.no_grad()
    def unpack_client_model(self, packed_model):
        if (
            not isinstance(packed_model, dict)
            or packed_model.get("codec") != "fedvote-binary-v4"
        ):
            raise ValueError("Invalid FedVote client payload")

        return {
            "binary": {
                key: self._unpack_binary_tensor(packet)
                for key, packet in packed_model["binary"].items()
            }
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _log_vote_diagnostics(
        self,
        key,
        p_raw,
        p,
        h_before,
        h_after,
        votes,
    ):
        if not self._diag_enabled_this_round():
            return

        p_cpu = p.detach().float().cpu()
        p_raw_cpu = p_raw.detach().float().cpu()
        hb = h_before.detach().float().cpu()
        ha = h_after.detach().float().cpu()

        stacked_votes = torch.stack(
            [v.detach().float().cpu() for v in votes],
            dim=0,
        )

        vote_mean = stacked_votes.mean(dim=0)

        tie_rate = (
            (p_raw_cpu == 0.5)
            .float()
            .mean()
            .item()
        )

        near_half_rate = (
            ((p_raw_cpu >= 0.4) & (p_raw_cpu <= 0.6))
            .float()
            .mean()
            .item()
        )

        strong_rate = (
            ((p_raw_cpu <= 0.2) | (p_raw_cpu >= 0.8))
            .float()
            .mean()
            .item()
        )

        unanimous_rate = (
            ((p_raw_cpu == 0.0) | (p_raw_cpu == 1.0))
            .float()
            .mean()
            .item()
        )

        clipped_rate = (
            (p_raw_cpu != p_cpu)
            .float()
            .mean()
            .item()
        )

        consensus = (
            vote_mean.abs()
            .mean()
            .item()
        )

        strong_consensus = (
            (vote_mean.abs() >= 0.6)
            .float()
            .mean()
            .item()
        )

        sign_before = torch.where(
            hb >= 0,
            torch.ones_like(hb),
            -torch.ones_like(hb),
        )
        sign_after = torch.where(
            ha >= 0,
            torch.ones_like(ha),
            -torch.ones_like(ha),
        )

        sign_retention = (
            (sign_before == sign_after)
            .float()
            .mean()
            .item()
        )

        # Normalized-weight magnitude before aggregation.  With good FedVote
        # initialization this should NOT start near zero for almost everything.
        norm_before = torch.tanh(
            self.normalization_scale * hb
        )
        norm_after = torch.tanh(
            self.normalization_scale * ha
        )

        self._log(
            f"round={self._server_round} layer={key} "
            f"clients={stacked_votes.shape[0]} "
            f"p_mean={p_cpu.mean().item():.6f} "
            f"p_std={p_cpu.std().item():.6f} "
            f"p_min={p_cpu.min().item():.6f} "
            f"p_max={p_cpu.max().item():.6f} "
            f"tie={100.0*tie_rate:.2f}% "
            f"near_half={100.0*near_half_rate:.2f}% "
            f"strong={100.0*strong_rate:.2f}% "
            f"unanimous={100.0*unanimous_rate:.4f}% "
            f"clipped={100.0*clipped_rate:.4f}% "
            f"vote_consensus={consensus:.6f} "
            f"agree80={100.0*strong_consensus:.2f}% "
            f"|h_before|={hb.abs().mean().item():.6f} "
            f"|h_after|={ha.abs().mean().item():.6f} "
            f"|wtilde_before|={norm_before.abs().mean().item():.6f} "
            f"|wtilde_after|={norm_after.abs().mean().item():.6f} "
            f"sign_retention={100.0*sign_retention:.2f}%"
        )

    # ------------------------------------------------------------------
    # Server aggregation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def aggregate(self, clients_models, global_model):
        if not clients_models:
            return

        self._server_round += 1

        first_payload = clients_models[0][1]
        binary_keys = list(first_payload["binary"].keys())
        n_clients = float(len(clients_models))

        if self._diag_enabled_this_round():
            self._log(
                f"START aggregation round={self._server_round}, "
                f"voting_clients={int(n_clients)}"
            )

        for key in binary_keys:
            votes = []
            plus_count = None

            h_before = global_model[key].detach().clone()

            for _, payload in clients_models:
                vote = payload["binary"][key].float()
                votes.append(vote)

                indicator = (vote + 1.0) * 0.5
                if plus_count is None:
                    plus_count = indicator.clone()
                else:
                    plus_count += indicator

            p_raw = plus_count / n_clients
            p = torch.clamp(
                p_raw,
                self.p_min,
                self.p_max,
            )

            h_after = self._p_to_h(p)

            target = global_model[key]
            global_model[key] = h_after.to(
                device=target.device,
                dtype=target.dtype,
            )

            hard_vote, tie_mask = self._hard_vote_from_p(p_raw)

            self._last_soft_vote[key] = p.detach().cpu()
            self._last_hard_vote[key] = hard_vote.detach().cpu()
            self._last_tie_mask[key] = tie_mask.detach().cpu()

            self._log_vote_diagnostics(
                key=key,
                p_raw=p_raw,
                p=p,
                h_before=h_before,
                h_after=h_after,
                votes=votes,
            )

        if self._diag_enabled_this_round():
            self._log(
                f"END aggregation round={self._server_round}"
            )

    # ------------------------------------------------------------------
    # Server -> client
    # ------------------------------------------------------------------
    @torch.no_grad()
    def pack_server_model(self, raw_model):
        soft_vote = {}

        for key, tensor in raw_model.items():
            if not self._is_vote_weight(key, tensor):
                continue

            p = (
                self._normalize(tensor.float()) + 1.0
            ) * 0.5

            soft_vote[key] = torch.clamp(
                p,
                self.p_min,
                self.p_max,
            ).detach().cpu()

        return {
            "codec": "fedvote-soft-v4",
            "p": soft_vote,
        }

    @torch.no_grad()
    def unpack_server_model(self, packed_model, current_model=None):
        if (
            not isinstance(packed_model, dict)
            or packed_model.get("codec") != "fedvote-soft-v4"
        ):
            return packed_model

        if current_model is None:
            model = {}
        else:
            model = {
                key: value.detach().clone()
                for key, value in current_model.items()
            }

        for key, p_cpu in packed_model.get("p", {}).items():
            if key in model:
                p = p_cpu.to(
                    device=model[key].device,
                    dtype=torch.float32,
                )
                model[key] = self._p_to_h(p).to(
                    dtype=model[key].dtype
                )
            else:
                model[key] = self._p_to_h(
                    p_cpu.float()
                )

        return model


    # ------------------------------------------------------------------
    # Evaluation hooks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def before_global_evaluation(self, model):
        """Evaluate the deployed hard BNN without altering training state.

        For non-tied coordinates, sign(h) is exactly the hard plurality vote.
        For an exact vote tie, however, the broadcast soft probability is
        p=0.5 and therefore h=0.  The FedVote paper specifies *random* tie
        breaking for the hard global binary weight.  We temporarily nudge only
        those tied latent coordinates toward the already sampled hard vote,
        switch the forward pass to sign(h), and restore the exact h values after
        evaluation.  Local training therefore still starts from the unmodified
        soft-vote latent model.
        """
        self._eval_tie_restore = {}

        named_params = dict(model.named_parameters())
        for key, tie_mask_cpu in self._last_tie_mask.items():
            param = named_params.get(key)
            hard_vote_cpu = self._last_hard_vote.get(key)
            if param is None or hard_vote_cpu is None:
                continue

            tie_mask = tie_mask_cpu.to(device=param.device)
            if not bool(tie_mask.any()):
                continue

            original = param.detach()[tie_mask].clone()
            self._eval_tie_restore[key] = original

            hard_vote = hard_vote_cpu.to(
                device=param.device,
                dtype=param.dtype,
            )
            # Any nonzero magnitude is sufficient because hard evaluation uses
            # only the sign.  1e-12 is exactly harmless for the temporary state.
            param.data[tie_mask] = hard_vote[tie_mask] * 1e-12

        if hasattr(model, "set_hard_eval"):
            model.set_hard_eval(True)

            if self.diagnostics:
                n_ties = sum(int(m.sum().item()) for m in self._last_tie_mask.values())
                self._log(
                    "global evaluation mode = HARD BNN "
                    "(quantized layers use sign(h); "
                    f"paper-random ties={n_ties})"
                )

    @torch.no_grad()
    def after_global_evaluation(self, model):
        """Restore the exact soft-training latent model after hard evaluation."""
        if hasattr(model, "set_hard_eval"):
            model.set_hard_eval(False)

        if self._eval_tie_restore:
            named_params = dict(model.named_parameters())
            for key, original in self._eval_tie_restore.items():
                param = named_params.get(key)
                tie_mask_cpu = self._last_tie_mask.get(key)
                if param is None or tie_mask_cpu is None:
                    continue
                tie_mask = tie_mask_cpu.to(device=param.device)
                param.data[tie_mask] = original.to(
                    device=param.device,
                    dtype=param.dtype,
                )
            self._eval_tie_restore = {}

        if self.diagnostics:
            self._log(
                "global evaluation mode restored = SOFT TRAINING "
                "(quantized layers use tanh(k*h))"
            )

    def last_soft_vote(self):
        return self._last_soft_vote

    def last_hard_vote(self):
        return self._last_hard_vote
