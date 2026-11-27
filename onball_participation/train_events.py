"""Fit matched next-action classifiers on one target-blind development view."""

import argparse
import json
from pathlib import Path

from train import Actor, SequenceEncoder, deterministic, lr_at, rotate_orders
from build import ACTION13_NAMES
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("direct10", "context", "zone", "role", "personal")


def center_log(p):
    logp = np.log(p)
    return (logp - logp.mean(axis=-1, keepdims=True)).astype(np.float32)


def role_prior(counts):
    pooled = counts.sum(axis=0) + 1
    pooled = pooled / pooled.sum(axis=-1, keepdims=True)
    return (counts + 50 * pooled) / (counts.sum(axis=-1, keepdims=True) + 50)


def action_profiles(matches, match, val, slot, zone, action, player, group):
    """Query whole UTC days before using their training outcomes; freeze for validation."""
    meta = matches.set_index("match_id").loc[match]
    assert np.array_equal(meta.block == "model_validation", val)
    day = meta.date.to_numpy()
    assert day[~val].max() < day[val].min()
    training = np.flatnonzero(~val & (slot >= 0))
    ids = np.unique(player)
    pid = np.searchsorted(ids, player)
    pc = np.zeros((len(ids), 9, 13), dtype=np.int64)
    rc = np.zeros((8, 9, 13), dtype=np.int64)
    role = np.zeros((len(match), 13), np.float32)
    person = np.zeros_like(role)
    for date in np.unique(day[training]):
        rows = training[day[training] == date]
        g, p = group[rows, slot[rows]], pid[rows, slot[rows]]
        z, y = zone[rows], action[rows]
        prior = role_prior(rc)[g, z]
        count = pc[p, z]
        role[rows] = center_log(prior)
        person[rows] = center_log((count + 50 * prior) / (count.sum(1, keepdims=True) + 50))
        np.add.at(rc, (g, z, y), 1)
        np.add.at(pc, (p, z, y), 1)
    prior = role_prior(rc)
    profile = (pc[:, None] + 50 * prior[None]) / (pc.sum(-1, keepdims=True)[:, None] + 50)
    # Lookup by (player, current role); unseen players inherit that role's distribution.
    return role, person, center_log(prior), center_log(profile).reshape(-1, 9, 13), pid * 8 + group


def swap_slots(player, group):
    columns = np.arange(player.shape[1])
    perm = np.tile(columns, (len(player), 1))
    for k in columns:
        same = (group == group[:, k, None]) & (player != 0)
        before = np.where(same & (columns < k), columns, -1).max(1)
        last = np.where(same, columns, -1).max(1)
        perm[:, k] = np.where(player[:, k] == 0, k, np.where(before >= 0, before, last))
    return perm


@torch.no_grad()
def participant_distribution(d, valid, group, checkpoint, device):
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    actor, location = Actor().to(device).eval(), nn.Linear(64, 9).to(device).eval()
    actor.load_state_dict(weights["actor"])
    location.load_state_dict(weights["location"])
    out = []
    for i in valid.split(1024):
        role = F.one_hot(group[i].long(), 8).float()
        state, _ = actor.candidate_hidden(d, i, role)
        who = actor.actor_logits(state, d["mask"][i]).softmax(-1)
        out.append(who[:, :, None] * location(state).softmax(-1))
    joint = torch.cat(out)
    torch.testing.assert_close(joint.sum((1, 2)), torch.ones_like(joint[:, 0, 0]))
    assert not joint[~d["mask"][valid]].any()
    return joint


class EventModel(SequenceEncoder):
    def __init__(self, classes):
        super().__init__()
        self.action = nn.Linear(64, classes)
        self.area = nn.Embedding(9, classes)
        self.profile = nn.Linear(13, classes, bias=False)
        nn.init.zeros_(self.area.weight)
        nn.init.zeros_(self.profile.weight)

    def context_logits(self, d, rows):
        return self.action(self.encode_context(d, rows))


@torch.no_grad()
def predict(model, arm, d, valid, joint, lookup, keys, perm, diagnostics=False):
    model.eval()
    out = {}
    offset = model.area.weight
    if arm in ("role", "personal"):
        offset = offset + model.profile(lookup[arm])
    for j in torch.arange(len(valid), device=valid.device).split(512):
        i = valid[j]
        logits = model.context_logits(d, i)
        weights = joint[j]
        predictions = {}
        if arm in ("direct10", "context"):
            prob = logits.softmax(-1)
        elif arm == "zone":
            branch = (logits[:, None, :] + offset).softmax(-1)
            prob = (weights.sum(1)[:, :, None] * branch).sum(1)
        else:
            key = keys[arm][j]
            branch = (logits[:, None, None, :] + offset[key]).softmax(-1)
            prob = (weights[:, :, :, None] * branch).sum((1, 2))
            if diagnostics:
                b = torch.arange(len(i), device=i.device)
                actor, zone = d["slot"][i], d["zone"][i]
                known_actor = weights[b, actor]
                known_zone = weights[b, :, zone]
                predictions["known_actor"] = (known_actor.unsqueeze(-1) * branch[b, actor]).sum(
                    1
                ) / known_actor.sum(1, keepdim=True)
                predictions["known_zone"] = (known_zone.unsqueeze(-1) * branch[b, :, zone]).sum(
                    1
                ) / known_zone.sum(1, keepdim=True)
                predictions["known_both"] = branch[b, actor, zone]
                if arm == "personal":
                    shuffled = branch[b[:, None], perm[j]]
                    predictions["swap"] = (weights.unsqueeze(-1) * shuffled).sum((1, 2))
        predictions["predicted"] = prob
        for name, p in predictions.items():
            out.setdefault(name, []).append(p)
    return {key: torch.cat(parts) for key, parts in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "matches", "upstream", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    deterministic(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    with np.load(args.data) as src:
        np.testing.assert_array_equal(src["action13_names"], ACTION13_NAMES)
        player, group, slot = src["player"], src["group"], src["slot"]
        match, event, val = src["match"], src["event"], src["val"]
        action, zone = src["action13"], src["zone"]
        d = {
            k: torch.as_tensor(src[k], device=device)
            for k in ("token", "wzone", "context", "age", "hist")
        }

    def tensor(x):
        return torch.as_tensor(x, device=device)

    train = np.flatnonzero(~val & (slot >= 0))
    valid = np.flatnonzero(val & (slot >= 0))
    d.update(mask=tensor(player != 0), slot=tensor(slot).long(), zone=tensor(zone).long())
    y = tensor(action).long()
    folded = torch.where(y >= 10, 0, y)
    group_t = tensor(group)
    vi = tensor(valid)
    joint = participant_distribution(d, vi, group_t, args.upstream / "best.pt", device)
    b = torch.arange(len(vi), device=device)
    joint_loss = -joint[b, d["slot"][vi], d["zone"][vi]].log()
    with np.load(args.upstream / "validation.npz") as old:
        np.testing.assert_array_equal(old["event"], event[valid])
        np.testing.assert_array_equal(old["match"], match[valid])
        np.testing.assert_allclose(
            joint_loss.cpu().numpy(), old["scores"][:, 5], atol=1e-5, rtol=1e-5
        )
    np.savez(
        args.out / "participants.npz",
        match=match[valid],
        event=event[valid],
        player=player[valid],
        joint=joint.cpu().numpy(),
    )
    role, personal, role_lookup, player_lookup, person_key = action_profiles(
        pd.read_parquet(args.matches), match, val, slot, zone, action, player, group
    )
    profiles = {"role": tensor(role), "personal": tensor(personal)}
    lookup = {"role": tensor(role_lookup), "personal": tensor(player_lookup)}
    keys = {"role": group_t[vi].long(), "personal": tensor(person_key[valid]).long()}
    perm = tensor(swap_slots(player[valid], group[valid]))
    exchangeable = (perm != torch.arange(11, device=device)).any(1).cpu().numpy()
    orders = tensor(train[rotate_orders(match[train], args.seed, 120, 128)])
    cfg = {"lr": 0.0027, "factor": 0.3, "milestones": [96, 108]}
    for arm in ARMS:
        path = args.out / arm
        path.mkdir()
        deterministic(args.seed)
        target = folded if arm == "direct10" else y
        model = EventModel(10 if arm == "direct10" else 13).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.0001)
        curve, best = [], float("inf")
        for epoch, order in enumerate(orders, 1):
            model.train()
            for g in optimizer.param_groups:
                g["lr"] = lr_at(cfg, epoch)
            for i in order.split(256):
                logits = model.context_logits(d, i)
                if arm in ("zone", "role", "personal"):
                    logits = logits + model.area(d["zone"][i])
                if arm in profiles:
                    logits = logits + model.profile(profiles[arm][i])
                loss = F.cross_entropy(logits, target[i])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            if epoch % 4 == 0:
                p = predict(model, arm, d, vi, joint, lookup, keys, perm)["predicted"]
                nll = -p[b, target[vi]].log().double().mean().item()
                if not np.isfinite(nll):
                    raise ValueError(f"Non-finite {arm} validation NLL at epoch {epoch}")
                curve.append([epoch, nll])
                if nll < best:
                    best = nll
                    torch.save(model.state_dict(), path / "best.pt")
                if epoch % 40 == 0:
                    print(arm, epoch, nll, flush=True)
        model.load_state_dict(torch.load(path / "best.pt", map_location=device, weights_only=True))
        predictions = predict(model, arm, d, vi, joint, lookup, keys, perm, diagnostics=True)
        for p in predictions.values():
            assert (p > 0).all() and torch.isfinite(p).all()
            torch.testing.assert_close(p.sum(-1), torch.ones_like(p[:, 0]))
        np.savez(
            path / "validation.npz",
            event=event[valid],
            match=match[valid],
            exchangeable=exchangeable,
            **{k: v.cpu().numpy() for k, v in predictions.items()},
        )
        (path / "curve.json").write_text(json.dumps(curve, allow_nan=False) + "\n")
        print(arm, "best", best, flush=True)


if __name__ == "__main__":
    main()
