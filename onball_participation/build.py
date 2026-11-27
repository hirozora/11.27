"""Build two target-blind views of the England development stream."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE = Path(__file__).resolve().parent.parent / "scene_regularity_release"
sys.path.insert(0, str(RELEASE / "src"))
from common import GROUP_OF, GROUPS
from event_schema import ACTIONS, ACTION_OF, ON_BALL
from model_inputs import VOCAB

K = 32


def preceding(indices, times, limit):
    end = np.searchsorted(times[indices], limit, side="left")
    pos = end[:, None] - np.arange(K, 0, -1)
    return (
        np.where(pos >= 0, indices[np.maximum(0, pos)], -1)
        if len(indices)
        else np.full(pos.shape, -1)
    )


def inputs(events, query, roster, subs, lag):
    """Only query attributes supplied here are team, time and period."""
    team = query.team_id.iloc[0]
    t = events.period.to_numpy() * 10000 + events.sec.to_numpy()
    qt = query.period.to_numpy() * 10000 + query.sec.to_numpy()
    own = events.team_id.to_numpy() == team
    sub = events.sub_event_name
    onball = (
        events.event_name.isin(ON_BALL).to_numpy()
        | (sub == "Ground attacking duel").to_numpy()
    )
    actions = (
        pd.Series(list(zip(events.event_name, sub)))
        .map(ACTION_OF)
        .map(dict(zip(ACTIONS, range(10))))
        .fillna(-1)
        .to_numpy(int)
    )
    early = preceding(np.arange(len(t)), t, qt - 2)
    recent = preceding(np.flatnonzero(own & onball), t, qt)
    recent[(recent < 0) | (t[recent] < qt[:, None] - 2) | lag] = -1
    w = np.sort(np.concatenate([early, recent], axis=1), axis=1)[:, -K:]
    valid = w >= 0
    xy = events[["x", "y"]].to_numpy()
    xy[~own] = 100 - xy[~own]
    zones = np.minimum((xy * 3 / 100).astype(int), 2) @ [3, 1]
    zones[~events.xy_valid.to_numpy()] = 9
    codes = pd.Index(VOCAB["sub_event_name"]).get_indexer(sub)
    assert min(codes) >= 0
    tokens = np.where(valid, 1 + codes[w] + len(VOCAB["sub_event_name"]) * (~own[w]), 0)
    wz = np.where(valid, zones[w], 9)
    ages = np.where(valid, np.log1p(np.clip(qt[:, None] - t[w], 0, 3600)) / 8, 0)
    # Consecutive own on-ball records in this visible window and half.
    own_run = np.zeros(len(qt))
    for j in range(K):
        seen = (
            valid[:, j]
            & onball[w[:, j]]
            & (events.period.to_numpy()[w[:, j]] == query.period.to_numpy())
        )
        own_run = np.where(seen, np.where(own[w[:, j]], own_run + 1, 0), own_run)
    ctx = np.column_stack([query.period, query.clock / 5400, own_run / K])
    n = len(qt)
    players = np.zeros((n, 11), int)
    hist = np.zeros((n, 11, 22), np.float32)
    mask_count = np.zeros(n, int)
    record_player = events.player_id.to_numpy()
    cutoff = qt - 2 if lag else qt

    # A substitution is observable by the minute rule or the incoming player's first visible event.
    def observed(pid):
        ix = np.flatnonzero((record_player == pid) & own)
        early_seen = np.searchsorted(t[ix], qt - 2) > 0
        return (
            early_seen
            if lag
            else early_seen | (np.searchsorted(t[ix[onball[ix]]], qt) > 0)
        )

    incoming = {}
    outgoing = {}
    for s in subs.itertuples():
        changed = (query.clock.to_numpy() >= (s.minute + 1) * 60) | (
            observed(s.player_in) if s.player_in else False
        )
        incoming[s.player_in] = changed
        outgoing[s.player_out] = changed
    for p in roster.sort_values("player_id").itertuples():
        pid = p.player_id
        active = np.ones(n, bool)
        if pid in incoming:
            active &= incoming[pid]
        if pid in outgoing:
            active &= ~outgoing[pid]
        if p.card_period >= 0:
            # A card is a match administration record, visible after its timestamp and embargo.
            active &= qt - 2 <= p.card_period * 10000 + p.card_sec
        rows = np.flatnonzero(active)
        slots = mask_count[rows]
        assert np.all(slots < 11)
        players[rows, slots] = pid
        mask_count[rows] += 1
        ix = np.flatnonzero(own & (record_player == pid) & (actions >= 0))
        past = preceding(ix, t, cutoff[rows])
        count = (past >= 0).sum(1)
        for j in range(9):
            hist[rows, slots, j] = ((past >= 0) & (zones[past] == j)).sum(
                1
            ) / np.maximum(1, count)
        for j in range(10):
            hist[rows, slots, 9 + j] = ((past >= 0) & (actions[past] == j)).sum(
                1
            ) / np.maximum(1, count)
        hist[rows, slots, 19] = count / K
        hist[rows, slots, 20] = np.where(
            count, np.log1p(np.clip(qt[rows] - t[past[:, -1]], 0, 3600)) / 8, 1
        )
        hist[rows, slots, 21] = np.where(count, (zones[past[:, -1]] + 1) / 10, 0)
    assert (mask_count > 0).all()
    return dict(
        token=tokens.astype("uint8"),
        wzone=wz.astype("uint8"),
        age=ages.astype("float16"),
        context=ctx.astype("float32"),
        player=players,
        hist=hist.astype("float16"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--view", choices=["main", "lag"], required=True)
    args = p.parse_args()
    d = args.data / "model"
    read = lambda name: pd.read_parquet(d / (name + ".parquet"))
    events = read("events_in")
    m = read("targets_meta")
    y = read("targets_label")
    loc = read("targets_in")
    matches = read("matches").set_index("match_id")
    assert np.array_equal(m.event_id, loc.event_id)
    roster = pd.read_parquet(args.data / "onpitch.parquet")
    subs = read("substitutions")
    prior = read("history")
    roles = json.loads((d / "summary.json").read_text())["roles"]
    group = {
        (a.match_id, a.player_id): (
            GROUPS.index(GROUP_OF[roles[a.prior_role]]) if a.prior_role >= 0 else 7
        )
        for a in prior.itertuples()
    }
    all_players = pd.Index(np.unique(roster.player_id))
    identity = pd.Index(np.r_[0, all_players])
    counts = np.zeros((len(identity), 10, 9), int)
    grouped = np.zeros((8, 10, 9), int)
    blocks = {}
    by_match = events.groupby("match_id").indices
    for day, games in matches.groupby("date", sort=True):
        updates = []
        for mid, game in games.iterrows():
            e = events.iloc[by_match[mid]].reset_index(drop=True)
            for team, query in m[m.match_id == mid].groupby("team_id"):
                q = query[["team_id", "period", "sec", "clock"]]
                r = roster[(roster.match_id == mid) & (roster.team_id == team)]
                s = subs[(subs.match_id == mid) & (subs.team_id == team)]
                x = inputs(e, q, r, s, args.view == "lag")
                if mid == matches.index[0]:
                    # Perturb all payloads of current and future events for a fixed query.
                    probe = q.iloc[[len(q) // 2]]
                    after = (e.period > probe.period.iloc[0]) | (
                        (e.period == probe.period.iloc[0])
                        & (e.sec >= probe.sec.iloc[0])
                    )
                    altered = e.copy()
                    altered.loc[after, ["x", "y", "player_id"]] = [11, 87, 0]
                    altered.loc[after, ["event_name", "sub_event_name"]] = [
                        "Shot",
                        "Shot",
                    ]
                    check = inputs(altered, probe, r, s, args.view == "lag")
                    assert all(
                        np.array_equal(v[len(q) // 2], check[k][0])
                        for k, v in x.items()
                    )
                ids = x["player"]
                g = np.array([group.get((mid, int(a)), 7) for a in np.unique(ids)])
                g = g[np.searchsorted(np.unique(ids), ids)]
                pi = identity.get_indexer(ids.ravel()).reshape(ids.shape)
                assert (pi >= 0).all()
                zprev = x["wzone"][:, -1, None]
                pool = grouped.sum(0) + 1
                gp = (
                    grouped[g, zprev] + 50 * (pool / pool.sum(-1, keepdims=True))[zprev]
                )
                gp = gp / gp.sum(-1, keepdims=True)
                pers = counts[pi, zprev] + 50 * gp
                x.update(
                    group=g.astype("uint8"),
                    group_prior=gp.astype("float16"),
                    personal=(pers / pers.sum(-1, keepdims=True)).astype("float16"),
                )
                ix = query.index.to_numpy()
                actor = y.player_id.to_numpy()[ix]
                raw_target = e.iloc[query.seq.to_numpy()]
                assert np.array_equal(raw_target.event_id, query.event_id)
                assert np.array_equal(raw_target.player_id, actor)
                raw_xy = raw_target[["x", "y"]].to_numpy()
                assert np.array_equal(
                    np.minimum((raw_xy * 3 / 100).astype(int), 2) @ [3, 1],
                    loc.zone.to_numpy()[ix],
                )
                hit = ids == actor[:, None]
                slot = np.where(hit.any(1), hit.argmax(1), -1)
                z = loc.zone.to_numpy()[ix]
                x.update(
                    slot=slot.astype("int8"),
                    zone=z.astype("int8"),
                    match=query.match_id.to_numpy(),
                    event=query.event_id.to_numpy(),
                    val=np.full(len(ix), game.block == "model_validation"),
                )
                for k, v in x.items():
                    if k not in blocks:
                        blocks[k] = np.empty((len(m), *v.shape[1:]), v.dtype)
                    blocks[k][ix] = v
                if game.block == "model_train":
                    ok = slot >= 0
                    updates.append(
                        (
                            identity.get_indexer(actor[ok]),
                            g[ok, slot[ok]],
                            zprev[ok, 0],
                            z[ok],
                        )
                    )
        for pid, g, h, z in updates:
            np.add.at(counts, (pid, h, z), 1)
            np.add.at(grouped, (g, h, z), 1)
    assert np.array_equal(blocks["event"], m.event_id)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / (args.view + ".npz"), **blocks)
    print(
        args.view,
        [
            (int(v.sum()), int((blocks["slot"][v] >= 0).sum()))
            for v in (~blocks["val"], blocks["val"])
        ],
        flush=True,
    )


if __name__ == "__main__":
    main()
