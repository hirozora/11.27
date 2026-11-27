"""Train one paired development run. Execute from the football workspace."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "scene_regularity_release/src")
)
from train_seq import deterministic, rotate_orders, lr_at
from model_inputs import VOCAB


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.event = nn.Embedding(
            1 + 2 * len(VOCAB["sub_event_name"]), 8, padding_idx=0
        )
        self.zone = nn.Embedding(10, 4)
        self.sequence = nn.Sequential(
            nn.Linear(32 * 13 + 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(64 + 22 + 8, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def candidate_hidden(self, d, i, role):
        tok = d["token"][i].long()
        x = torch.cat(
            [
                self.event(tok),
                self.zone(d["wzone"][i].long()),
                d["age"][i, :, None].float(),
            ],
            -1,
        )
        c = self.sequence(
            torch.cat([(x * (tok > 0)[:, :, None]).flatten(1), d["context"][i]], 1)
        )
        state = torch.cat(
            [c[:, None].expand(-1, 11, -1), d["hist"][i].float(), role], -1
        )
        for layer in list(self.scorer.children())[:-1]:
            state = layer(state)
        return state, c

    def actor_logits(self, hidden, mask):
        return self.scorer[-1](hidden).squeeze(-1).masked_fill(~mask, -torch.inf)

    def forward(self, d, i, role):
        hidden, context = self.candidate_hidden(d, i, role)
        return self.actor_logits(hidden, d["mask"][i]), context


class Zone(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(64 + 8 + 9 + 22 + 9, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 9),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--heads", required=True, choices=["ABCD", "BD"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    deterministic(args.seed)
    dev = torch.device(args.device)
    with np.load(args.data) as src:
        players = src["player"]
        match = src["match"]
        event = src["event"]
        valflag = src["val"]
        slot = src["slot"]
        d = {
            k: torch.as_tensor(src[k], device=dev)
            for k in (
                "token",
                "wzone",
                "age",
                "context",
                "group",
                "group_prior",
                "hist",
                "personal",
                "zone",
                "slot",
            )
        }
    d["mask"] = torch.as_tensor(players != 0, device=dev)
    role = F.one_hot(d["group"].long(), 8).float()
    rows = np.flatnonzero(~valflag & (slot >= 0))
    valid = np.flatnonzero(valflag & (slot >= 0))
    val = torch.as_tensor(valid, device=dev)
    orders = rows[rotate_orders(match[rows], args.seed, 120, 128)]
    orders = torch.as_tensor(orders, device=dev)
    actor = Actor().to(dev)
    cfg = {"lr": 0.0027, "milestones": [96, 108], "factor": 0.3}
    frozen = torch.empty((len(slot), 64), device=dev)
    q = torch.empty((len(slot), 11), device=dev)
    # Cyclic permutation within occupied slots of the same team and pre-match group.
    group = d["group"][val].cpu().numpy()
    permutation = np.tile(np.arange(11), (len(valid), 1))
    for r, t in enumerate(valid):
        for g in np.unique(group[r, players[t] != 0]):
            j = np.flatnonzero((group[r] == g) & (players[t] != 0))
            permutation[r, j] = np.roll(j, 1)
    perm = torch.as_tensor(permutation, device=dev)
    out = {
        "event": event[valid],
        "match": match[valid],
        "exchangeable": (permutation != np.arange(11)).any(1),
    }

    def zfeatures(i, kind, slots=None, swap=None):
        select = (
            torch.arange(11, device=dev)[None, :] if slots is None else slots[:, None]
        )
        j = i[:, None]
        pick = select if swap is None else swap
        parts = [frozen[i, None].expand(-1, select.shape[-1], -1)]
        for tensor, needed, idx in [
            (role, "BCD", select),
            (d["group_prior"], "BCD", select),
            (d["hist"], "CD", pick),
            (d["personal"], "D", pick),
        ]:
            part = tensor[j, idx].float()
            parts.append(part if kind in needed else torch.zeros_like(part))
        return torch.cat(parts, -1)

    for kind in ["actor", *args.heads]:
        deterministic(args.seed)
        model = actor if kind == "actor" else Zone().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0001)

        def score(i):
            if kind == "actor":
                return model(d, i, role[i])[0], d["slot"][i].long()
            return (
                model(zfeatures(i, kind, d["slot"][i].long())).squeeze(1),
                d["zone"][i].long(),
            )

        best = float("inf")
        path = args.out / (kind + ".pt")
        curve = []
        for epoch, order in enumerate(orders, 1):
            model.train()
            for g in opt.param_groups:
                g["lr"] = lr_at(cfg, epoch)
            for i in order.split(256):
                logits, label = score(i)
                loss = F.cross_entropy(logits, label)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            if epoch % 4 == 0:
                model.eval()
                with torch.no_grad():
                    nll = sum(
                        F.cross_entropy(*score(i), reduction="sum").item()
                        for i in val.split(2048)
                    ) / len(val)
                if not np.isfinite(nll):
                    raise ValueError("nonfinite validation loss")
                curve.append([epoch, nll])
                if nll < best:
                    best = nll
                    torch.save(model.state_dict(), path)
        (args.out / (kind + "_curve.json")).write_text(json.dumps(curve))
        model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
        model.eval()
        with torch.no_grad():
            if kind == "actor":
                for i in torch.arange(len(slot), device=dev).split(2048):
                    logit, frozen[i] = actor(d, i, role[i])
                    q[i] = logit.log_softmax(-1)
                truth = d["slot"][val].long()
                prob = q[val].exp()
                assert torch.allclose(
                    prob.sum(-1), torch.ones_like(prob[:, 0]), atol=1e-5
                )
                out["actor"] = -q[val, truth].cpu().numpy()
                for k in [1, 3]:
                    target = prob.gather(1, truth[:, None])
                    higher = (prob > target).sum(1)
                    ties = (prob == target).sum(1)
                    out["top" + str(k)] = (
                        ((k - higher) / ties).clamp(0, 1).cpu().numpy()
                    )
            else:
                for name in [kind, kind + "_swap"] if kind == "D" else [kind]:
                    scores = []
                    for j in torch.arange(len(val), device=dev).split(1024):
                        i = val[j]
                        logits = model(
                            zfeatures(
                                i, kind, swap=perm[j] if name.endswith("swap") else None
                            )
                        )
                        conditional = logits.log_softmax(-1)
                        joint = q[i, :, None] + conditional
                        marg = joint.logsumexp(1)
                        assert marg.logsumexp(-1).abs().max() < 1e-5
                        a = d["slot"][i].long()
                        z = d["zone"][i].long()
                        b = torch.arange(len(i), device=dev)
                        scores.append(
                            torch.stack(
                                [-conditional[b, a, z], -marg[b, z], -joint[b, a, z]], 1
                            )
                            .cpu()
                            .numpy()
                        )
                    out[name] = np.concatenate(scores)
                    np.testing.assert_allclose(
                        out[name][:, 2], out["actor"] + out[name][:, 0], atol=2e-6
                    )
        print(kind, best, flush=True)
    np.savez(args.out / "validation.npz", **out)


if __name__ == "__main__":
    main()
