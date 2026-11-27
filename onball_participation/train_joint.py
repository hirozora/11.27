"""Matched auxiliary-location experiment, using the first-round actor encoder."""

import argparse
import json
from pathlib import Path

from train import Actor, deterministic, rotate_orders, lr_at
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=["detached", "joint"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    deterministic(args.seed)
    device = torch.device(args.device)
    with np.load(args.data) as src:
        slot, match, valflag = src["slot"], src["match"], src["val"]
        training = np.flatnonzero((slot >= 0) & ~valflag)
        validation = np.flatnonzero((slot >= 0) & valflag)
        events = src["event"][validation]
        d = {
            k: torch.as_tensor(src[k], device=device)
            for k in ("token", "wzone", "age", "context", "hist", "slot", "zone")
        }
        d["mask"] = torch.as_tensor(src["player"] != 0, device=device)
        role = F.one_hot(torch.as_tensor(src["group"], device=device).long(), 8).float()
    schedule = training[rotate_orders(match[training], args.seed, 120, 128)]
    orders = torch.as_tensor(schedule, device=device)
    val = torch.as_tensor(validation, device=device)
    actor = Actor().to(device)
    location = nn.Linear(64, 9).to(device)
    deterministic(args.seed)
    optimizer = torch.optim.AdamW(
        [*actor.parameters(), *location.parameters()], lr=0.0027, weight_decay=0.0001
    )
    config = {"lr": 0.0027, "milestones": [96, 108], "factor": 0.3}

    def forward(i):
        state, _ = actor.candidate_hidden(d, i, role[i])
        return actor.actor_logits(state, d["mask"][i]), state

    best = float("inf")
    curve = []
    for epoch, order in enumerate(orders, 1):
        actor.train()
        for g in optimizer.param_groups:
            g["lr"] = lr_at(config, epoch)
        for i in order.split(256):
            scores, state = forward(i)
            truth = d["slot"][i].long()
            chosen = state[torch.arange(len(i), device=device), truth]
            if args.arm == "detached":
                chosen = chosen.detach()
            loss = F.cross_entropy(scores, truth) + F.cross_entropy(
                location(chosen), d["zone"][i].long()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for model in (actor, location):
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % 4 == 0:
            actor.eval()
            with torch.no_grad():
                nll = sum(
                    F.cross_entropy(
                        forward(i)[0], d["slot"][i].long(), reduction="sum"
                    ).item()
                    for i in val.split(2048)
                ) / len(val)
            if not np.isfinite(nll):
                raise ValueError("nonfinite validation NLL")
            curve.append([epoch, nll])
            if nll < best:
                best = nll
                torch.save(
                    {"actor": actor.state_dict(), "location": location.state_dict()},
                    args.out / "best.pt",
                )
    checkpoint = torch.load(
        args.out / "best.pt", map_location=device, weights_only=True
    )
    actor.load_state_dict(checkpoint["actor"])
    location.load_state_dict(checkpoint["location"])
    results = []
    actor.eval()
    with torch.no_grad():
        for i in val.split(1024):
            logits, state = forward(i)
            q = logits.log_softmax(-1)
            c = location(state).log_softmax(-1)
            joint = q[:, :, None] + c
            marginal = joint.logsumexp(1)
            assert marginal.logsumexp(-1).abs().max() < 1e-5
            truth = d["slot"][i].long()
            region = d["zone"][i].long()
            b = torch.arange(len(i), device=device)
            target = q[b, truth]
            higher = (q > target[:, None]).sum(1)
            tied = (q == target[:, None]).sum(1)
            metrics = [
                -target,
                *[((k - higher) / tied).clamp(0, 1) for k in (1, 3)],
                -c[b, truth, region],
                -marginal[b, region],
                -joint[b, truth, region],
            ]
            results.append(torch.stack(metrics, 1).cpu().numpy())
    values = np.concatenate(results)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[:, 0] + values[:, 3], values[:, 5], atol=2e-6)
    np.savez(
        args.out / "validation.npz",
        event=events,
        match=match[validation],
        scores=values,
    )
    (args.out / "curve.json").write_text(json.dumps(curve) + "\n")
    print(args.arm, args.seed, best, flush=True)


if __name__ == "__main__":
    main()
