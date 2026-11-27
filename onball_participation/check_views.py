"""Raw-record perturbation and timestamp-boundary acceptance checks."""

import argparse
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from build import ACTION13_NAMES, inputs, target_action13
import build_model_data
from build_actions import onpitch_intervals
from model_inputs import VOCAB
from event_schema import ACTION_OF, ON_BALL, PERIOD_ORDER


def stream(raw, mid):
    with patch.object(build_model_data, "load", return_value=raw):
        return build_model_data.event_stream(
            {}, dict(raw_dir="data/raw", league="England", position_overrides=[]), {mid}
        )


def check_pass_boundaries():
    # The two 34 by 52.5 displacements are exactly diagonal in metres.
    cases = [
        (0, 0, 0, 0, "stationary"),
        (50, 50, 51, 50, "forward"),
        (50, 50, 50, 49, "lateral"),
        (0, 0, 3, 4, "forward"),
        (0, 0, 2, 4, "lateral"),
        (0, 0, 34, 52.5, "forward"),
        (34, 52.5, 0, 0, "backward"),
    ]
    frame = pd.DataFrame(cases, columns=["x", "y", "end_x", "end_y", "expected"])
    frame["event_id"] = frame.index
    frame["action"] = "short_pass"
    actual = np.asarray(ACTION13_NAMES)[target_action13(frame, frame)]
    np.testing.assert_array_equal(actual, "simple_pass_" + frame.expected)
    frame.loc[0, "end_y"] = np.nan
    with np.testing.assert_raises_regex(ValueError, "Invalid Simple pass coordinates"):
        target_action13(frame, frame)
    frame.loc[0, "action"] = "long_pass"
    assert ACTION13_NAMES[target_action13(frame, frame)[0]] == "long_pass"


def raw_action_name(event):
    action = ACTION_OF[event["eventName"], event["subEventName"]]
    if action != "short_pass":
        return action
    start, end = event["positions"]
    if start == end:
        return "simple_pass_stationary"
    dx = 105 * (end["x"] - start["x"])
    dy = 68 * (end["y"] - start["y"])
    angle = math.degrees(math.atan2(abs(dy), dx))
    if angle <= 45:
        return "simple_pass_forward"
    if angle >= 135:
        return "simple_pass_backward"
    return "simple_pass_lateral"


def check_boundaries():
    # At t=10 the opponent record and the own pass both have exactly two seconds of age.
    events = pd.DataFrame(
        {
            "period": [0] * 5,
            "sec": [7.0, 8.0, 8.0, 9.0, 10.0],
            "team_id": [2, 2, 1, 1, 1],
            "event_name": ["Pass", "Pass", "Pass", "Foul", "Pass"],
            "sub_event_name": [
                "Head pass",
                "Cross",
                "Simple pass",
                "Foul",
                "High pass",
            ],
            "player_id": [4, 4, 1, 2, 3],
            "x": [20.0] * 5,
            "y": [40.0] * 5,
            "xy_valid": [True] * 5,
        }
    )
    roster = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "card_period": [-1, 0, -1],
            "card_sec": [np.nan, 9.0, np.nan],
        }
    )
    subs = pd.DataFrame({"player_in": [3], "player_out": [1], "minute": [10.0]})
    q = pd.DataFrame(
        {"period": [0] * 4, "team_id": [1] * 4, "sec": [10.0, 10.01, 11.0, 12.01]}
    )
    q["clock"] = q.sec
    main = inputs(events, q, roster, subs, False)
    lag = inputs(events, q, roster, subs, True)
    names = VOCAB["sub_event_name"]
    head = 1 + names.index("Head pass") + len(names)
    simple = 1 + names.index("Simple pass")
    assert (
        main["token"][0, -2:].tolist() == [head, simple]
        and np.count_nonzero(main["token"][0]) == 2
    )
    assert lag["token"][0, -1] == head and np.count_nonzero(lag["token"][0]) == 1
    assert main["player"][0, :2].tolist() == [1, 2]
    assert main["player"][1, :2].tolist() == [2, 3]
    assert lag["player"][1, :2].tolist() == [1, 2]
    assert main["player"][2, :2].tolist() == [2, 3]
    assert lag["player"][3, 0] == 3 and np.count_nonzero(lag["player"][3]) == 1


def main():
    p = argparse.ArgumentParser()
    for arg in ["data", "raw", "results"]:
        p.add_argument("--" + arg, required=True, type=Path)
    a = p.parse_args()
    check_boundaries()
    check_pass_boundaries()
    m = pd.read_parquet(a.data / "model/targets_meta.parquet")
    original = pd.read_parquet(a.data / "onpitch.parquet")
    s = pd.read_parquet(a.data / "model/substitutions.parquet")
    usable = m[~m.match_id.isin(original.loc[original.card_period >= 0, "match_id"])]
    mids = [
        int(mid) for _, g in usable.groupby("block") for mid in g.match_id.unique()[:2]
    ]
    raw = {mid: [] for mid in mids}
    assert m.event_id.is_unique
    row_for_event = dict(zip(m.event_id, range(len(m))))
    raw_labels = np.full(len(m), -1, dtype=np.int8)
    for e in json.loads((a.raw / "events/events_England.json").read_text()):
        row = row_for_event.get(e["id"])
        if row is not None:
            raw_labels[row] = ACTION13_NAMES.index(raw_action_name(e))
        if e["matchId"] in raw:
            raw[e["matchId"]].append(e)
    assert (raw_labels >= 0).all()
    matches = {
        g["wyId"]: g
        for g in json.loads((a.raw / "matches/matches_England.json").read_text())
        if g["wyId"] in raw
    }
    roles = {
        g["wyId"]: g["role"]["code2"]
        for g in json.loads((a.raw / "entities/players.json").read_text())
    }
    # Match the production tensors at the tested rows, including the entire eligible roster.
    probe_indices = []
    for mid in mids:
        for _, qs in m[m.match_id == mid].groupby("team_id"):
            probe_indices.extend(qs.index[[0, len(qs) // 2, len(qs) - 1]].tolist())
    row_of = {ix: j for j, ix in enumerate(probe_indices)}
    saved = {}
    action_counts = {}
    for lag, view in ((False, "main"), (True, "lag")):
        with np.load(a.results / (view + ".npz")) as f:
            np.testing.assert_array_equal(f["event"], m.event_id)
            np.testing.assert_array_equal(f["action13_names"], ACTION13_NAMES)
            np.testing.assert_array_equal(f["action13"], raw_labels)
            reachable, val = f["slot"] >= 0, f["val"]
            action_counts[view] = {}
            for split, selected in (("train", ~val), ("validation", val)):
                counts = np.bincount(raw_labels[reachable & selected], minlength=len(ACTION13_NAMES))
                action_counts[view][split] = dict(zip(ACTION13_NAMES, counts.tolist()))
            saved[lag] = {
                k: f[k][probe_indices]
                for k in ("token", "wzone", "age", "context", "player", "hist")
            }
    count = 0
    for mid, records in raw.items():
        records.sort(key=lambda e: (PERIOD_ORDER[e["matchPeriod"]], e["eventSec"]))
        events = stream(records, mid)
        for team, qs in m[m.match_id == mid].groupby("team_id"):
            subs = s[(s.match_id == mid) & (s.team_id == team)]

            def team_roster(rec):
                rr = pd.DataFrame(onpitch_intervals(matches[mid], roles, rec, set())[0])
                return rr[rr.team_id == team]

            roster = team_roster(records)
            for ix in [0, len(qs) // 2, len(qs) - 1]:
                query = qs.iloc[[ix]][["team_id", "period", "sec", "clock"]]
                qt = query.iloc[0]
                mutated = copy.deepcopy(records)
                for e in mutated:
                    if (PERIOD_ORDER[e["matchPeriod"]], e["eventSec"]) >= (
                        qt.period,
                        qt.sec,
                    ):
                        e.update(
                            playerId=0,
                            eventName="Shot",
                            subEventName="Shot",
                            subEventId=100,
                            positions=[{"x": 93, "y": 2}, {"x": 11, "y": 57}],
                            tags=[{"id": 101}],
                        )
                changed = stream(mutated, mid)
                for lag in [False, True]:
                    baseline = inputs(events, query, roster, subs, lag)
                    altered = inputs(changed, query, team_roster(mutated), subs, lag)
                    for k in baseline:
                        np.testing.assert_array_equal(baseline[k], altered[k])
                    for k, v in baseline.items():
                        np.testing.assert_array_equal(
                            v[0], saved[lag][k][row_of[query.index[0]]]
                        )
                    # Independently enumerate the last 32 allowed records.
                    eligible = []
                    for e in records:
                        period = PERIOD_ORDER[e["matchPeriod"]]
                        dt = qt.sec - e["eventSec"]
                        own_ball = e["teamId"] == team and (
                            e["eventName"] in ON_BALL
                            or e["subEventName"] == "Ground attacking duel"
                        )
                        if period < qt.period or (
                            period == qt.period
                            and (dt > 2 or (not lag and dt > 0 and own_ball))
                        ):
                            eligible.append(e)
                    assert int((baseline["token"] != 0).sum()) == min(32, len(eligible))
                    code = [
                        1
                        + VOCAB["sub_event_name"].index(e["subEventName"])
                        + len(VOCAB["sub_event_name"]) * (e["teamId"] != team)
                        for e in eligible[-32:]
                    ]
                    np.testing.assert_array_equal(
                        baseline["token"][baseline["token"] != 0], code
                    )
                    count += 1
    report = {
        "raw_perturbation_queries_across_views": count,
        "matches": mids,
        "passed": True,
        "exact_timestamp_substitution_and_card_boundaries": True,
        "action13_raw_targets_checked_per_view": len(raw_labels),
        "action13_reachable_counts": action_counts,
    }
    (a.results / "view_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
