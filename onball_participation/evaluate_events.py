"""Score the action classifiers and pair per-event losses across seeds and matches."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import SEEDS, contrast


PASS = [0, 10, 11, 12]
ARMS = ("direct10", "context", "zone", "role", "personal")
TITLES = {
    "prior": "训练频率",
    "direct10": "直接十类",
    "context": "仅局面",
    "zone": "加预测区域",
    "role": "加位置组倾向",
    "personal": "加个人倾向",
    "personal.swap": "组内交换个人倾向",
}
PAIRS = (
    ("direct10", "context"),
    ("context", "zone"),
    ("zone", "role"),
    ("role", "personal"),
    ("context", "personal"),
    ("personal.swap", "personal"),
)


def confusion(truth, prediction, classes):
    pairs = truth.astype(np.int64, copy=False) * classes + prediction
    table = np.bincount(pairs, minlength=classes**2).reshape(classes, classes)
    denominator = table.sum(0) + table.sum(1)
    f1 = np.divide(2 * np.diag(table), denominator, out=np.zeros(classes), where=denominator > 0)
    return {
        "matrix": table.tolist(),
        "support": table.sum(1).tolist(),
        "class_f1": f1.tolist(),
        "macro_f1": f1.mean(),
        "accuracy": np.trace(table) / len(truth),
    }


def assess(p, y, sub):
    p = p.astype(np.float64)
    assert np.isfinite(p).all() and (p > 0).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=2e-6)
    coarse = np.where(y >= 10, 0, y)
    simple = coarse == 0
    rows = np.arange(len(y))
    losses, metrics = {}, {}
    if p.shape[1] == 13:
        losses["nll13"] = -np.log(p[rows, y])
        folded = p[:, :10].copy()
        folded[:, 0] = p[:, PASS].sum(1)
        losses["direction"] = -np.log(p[rows[simple], y[simple]] / folded[simple, 0])
        metrics["classes13"] = confusion(y, p.argmax(1), 13)
        direction = np.array([PASS.index(int(k)) for k in y[simple]])
        metrics["directions"] = confusion(direction, p[simple][:, PASS].argmax(1), 4)
        metrics["directions"]["class_nll"] = [
            losses["direction"][direction == k].mean() for k in range(4)
        ]
    else:
        assert p.shape[1] == 10
        folded = p
    pred = folded.argmax(1)
    losses["nll10"] = -np.log(folded[rows, coarse])
    if "nll13" in losses:
        expected = losses["nll10"].copy()
        expected[simple] += losses["direction"]
        np.testing.assert_allclose(expected, losses["nll13"], atol=1e-10)
    metrics["classes10"] = confusion(coarse, pred, 10)
    metrics["nll"] = {key: x.mean() for key, x in losses.items()}
    masks = {
        "High pass to simple": (sub == "High pass", 0),
        "Launch to simple": (sub == "Launch", 0),
        "Touch to simple": (coarse == 9, 0),
        "simple to long": (simple, 1),
        "simple to Touch": (simple, 9),
    }
    metrics["errors"] = {
        key: {"support": int(mask.sum()), "count": int((pred[mask] == wrong).sum())}
        for key, (mask, wrong) in masks.items()
    }
    metrics["long_pass_recall"] = {
        name: {
            "support": int((sub == name).sum()),
            "correct": int(((sub == name) & (pred == 1)).sum()),
        }
        for name in ("High pass", "Launch")
    }
    return metrics, losses


def compare(scores, left, right, match, simple):
    return {
        k: contrast(
            scores[left][k] - scores[right][k],
            match[simple] if k == "direction" else match,
            SEEDS[0],
        )
        for k in scores[left]
        if k in scores[right]
    }


def score_view(view, data, root, raw):
    with np.load(data / f"{view}.npz") as d:
        train = ~d["val"] & (d["slot"] >= 0)
        val = d["val"] & (d["slot"] >= 0)
        y = d["action13"][val].astype(int)
        names = d["action13_names"].tolist()
        p = np.bincount(d["action13"][train], minlength=13) / train.sum()
        event, match = d["event"][val], d["match"][val]
        player = d["player"][val]
        total = int(d["val"].sum())
    keys = pd.MultiIndex.from_arrays([match, event], names=["match_id", "event_id"])
    sub = raw.loc[keys].to_numpy()
    simple = np.isin(y, PASS)
    assert np.array_equal(sub == "Simple pass", simple)
    prior_metrics, _ = assess(np.broadcast_to(p, (len(y), 13)), y, sub)
    metrics = {}
    stored = {}
    epochs = {}
    for seed in SEEDS:
        run = root / f"{view}_{seed}"
        with np.load(run / "participants.npz") as f:
            np.testing.assert_array_equal(f["match"], match)
            np.testing.assert_array_equal(f["event"], event)
            np.testing.assert_array_equal(f["player"], player)
            q = f["joint"]
            assert q.shape == (*player.shape, 9) and (q >= 0).all() and (q[player == 0] == 0).all()
            np.testing.assert_allclose(q.sum((1, 2)), 1, atol=2e-6)
        for arm in ARMS:
            fit = run / arm
            curve = json.loads((fit / "curve.json").read_text())
            assert [e for e, _ in curve] == list(range(4, 121, 4))
            epoch, best = min(curve, key=lambda x: x[1])
            epochs.setdefault(arm, []).append(epoch)
            with np.load(fit / "validation.npz") as f:
                np.testing.assert_array_equal(f["event"], event)
                np.testing.assert_array_equal(f["match"], match)
                exchangeable = f["exchangeable"]
                for key in f.files:
                    if key in ("event", "match", "exchangeable"):
                        continue
                    name = arm if key == "predicted" else arm + "." + key
                    result, losses = assess(f[key], y, sub)
                    if key == "predicted":
                        metric = "nll10" if arm == "direct10" else "nll13"
                        assert abs(losses[metric].mean() - best) < 2e-6
                    metrics.setdefault(name, []).append(result)
                    for k, v in losses.items():
                        stored.setdefault(name, {}).setdefault(k, []).append(v)
    scores = {arm: {k: np.stack(v) for k, v in values.items()} for arm, values in stored.items()}
    pairs = list(PAIRS)
    for known in ("known_actor", "known_zone", "known_both"):
        pairs.append(("role." + known, "personal." + known))
        pairs.append(("personal", "personal." + known))
    gains = {f"{a} vs {b}": compare(scores, a, b, match, simple) for a, b in pairs}
    coarse = np.where(simple, 0, y)
    coarse_delta = scores["role"]["nll10"] - scores["personal"]["nll10"]
    by_class = {}
    for i, name in enumerate(["simple_pass", *names[1:10]]):
        keep = coarse == i
        by_class[name] = {
            "support": int(keep.sum()),
            "by_seed": coarse_delta[:, keep].mean(1).tolist(),
            "gain_within_class": coarse_delta[:, keep].mean(),
            "contribution_per_action": coarse_delta[:, keep].mean() * keep.mean(),
        }
    np.testing.assert_allclose(
        sum(v["contribution_per_action"] for v in by_class.values()),
        gains["role vs personal"]["nll10"]["gain"],
    )
    section = dict(
        training=int(train.sum()),
        validation=int(val.sum()),
        total_validation=total,
        actions=names,
        seeds=SEEDS,
        validation_prior=prior_metrics,
        per_seed=metrics,
        best_epochs=epochs,
        gains=gains,
        coarse_personal_gain_by_class=by_class,
        exchangeable=int(exchangeable.sum()),
    )
    return section, (event, match, y, scores["role"]["nll13"] - scores["personal"]["nll13"])


def gain_text(change):
    lo, hi = change["interval"]
    return f"{change['gain']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def report_view(view, section):
    title = "主视图" if view == "main" else "全部延迟两秒"
    lines = [
        f"## {title}",
        "",
        f"可达验证动作 {section['validation']:,}/{section['total_validation']:,}，训练可达 {section['training']:,}。按种子分别评分后取平均。",
        "",
        "| 模型 | 十三类 NLL | 折叠十类 NLL | 普通传球条件方向 NLL | 十三类 Macro-F1 | 十三类准确率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    all_metrics = {"prior": [section["validation_prior"]], **section["per_seed"]}
    for arm, label in TITLES.items():
        rows = all_metrics[arm]
        columns = []
        for k in ("nll13", "nll10", "direction"):
            columns.append(
                f"{np.mean([r['nll'][k] for r in rows]):.4f}" if k in rows[0]["nll"] else "—"
            )
        for k in ("macro_f1", "accuracy"):
            columns.append(
                f"{np.mean([r['classes13'][k] for r in rows]):.2%}"
                if "classes13" in rows[0]
                else "—"
            )
        lines.append("| " + " | ".join([label, *columns]) + " |")
    lines += [
        "",
        "| 左组到右组，各项 NLL 降幅 | 十三类 [95% 区间] | 十类 [95% 区间] | 方向 [95% 区间] |",
        "|---|---:|---:|---:|",
    ]
    for a, b in PAIRS:
        g = section["gains"][f"{a} vs {b}"]
        fields = [gain_text(g[k]) if k in g else "—" for k in ("nll13", "nll10", "direction")]
        lines.append("| " + " | ".join([TITLES[a] + " → " + TITLES[b], *fields]) + " |")
    primary = section["gains"]["role vs personal"]
    share = sum(all_metrics["role"][0]["directions"]["support"]) / section["validation"]
    lines += [
        "",
        "个人相对位置组的十三类 NLL 逐种子降幅为 "
        + "、".join(f"{x:+.4f}" for x in primary["nll13"]["by_seed"])
        + "。",
        f"按全部动作计，粗类判断贡献 {primary['nll10']['gain']:+.5f}，普通传球方向贡献 "
        f"{share * primary['direction']['gain']:+.5f}（普通传球占 {share:.2%}）。",
        "",
        "| 诊断时给定的信息 | 个人模型十三类 NLL | 相对正式推断的 NLL 降幅 [95% 区间] | 同条件下位置组到个人的 NLL 降幅 [95% 区间] |",
        "|---|---:|---:|---:|",
    ]
    for known, label in (
        ("known_actor", "给定真实执行者"),
        ("known_zone", "给定真实起点区域"),
        ("known_both", "同时给定两者"),
    ):
        personal = "personal." + known
        nll = np.mean([r["nll"]["nll13"] for r in all_metrics[personal]])
        gain = section["gains"]["personal vs " + personal]["nll13"]
        extra = section["gains"][f"role.{known} vs {personal}"]["nll13"]
        lines.append(f"| {label} | {nll:.4f} | {gain_text(gain)} | {gain_text(extra)} |")
    lines += [
        "",
        "| 已知为普通传球时的方向 | 动作数 | 位置组条件 NLL | 个人条件 NLL | 位置组召回率 | 个人召回率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, label in enumerate(("前向", "后向", "横传", "原地")):
        n = all_metrics["role"][0]["directions"]["support"][i]
        nll, recall = [], []
        for arm in ("role", "personal"):
            values = [r["directions"] for r in all_metrics[arm]]
            nll.append(np.mean([r["class_nll"][i] for r in values]))
            recall.append(np.mean([r["matrix"][i][i] for r in values]) / n)
        lines.append(
            f"| {label} | {n:,} | {nll[0]:.4f} | {nll[1]:.4f} | {recall[0]:.2%} | {recall[1]:.2%} |"
        )
    lines += [
        "",
        "| 折叠后的定向误判 | 分母 | 仅局面 | 位置组倾向 | 个人倾向 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = ("High pass → 普通", "Launch → 普通", "Touch → 普通", "普通 → 长传", "普通 → Touch")
    for label, k in zip(labels, all_metrics["context"][0]["errors"]):
        n = all_metrics["context"][0]["errors"][k]["support"]
        cells = [
            f"{np.mean([r['errors'][k]['count'] for r in all_metrics[arm]])/n:.2%}"
            for arm in ("context", "role", "personal")
        ]
        lines.append("| " + " | ".join([label, str(n), *cells]) + " |")
    lines += [
        "",
        "| 真实长传子类 | 分母 | 仅局面长传召回率 | 位置组长传召回率 | 个人长传召回率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in all_metrics["context"][0]["long_pass_recall"].items():
        n = metric["support"]
        cells = [
            f"{np.mean([r['long_pass_recall'][name]['correct'] for r in all_metrics[arm]])/n:.2%}"
            for arm in ("context", "role", "personal")
        ]
        lines.append("| " + " | ".join([name, str(n), *cells]) + " |")
    lines += [
        "",
        "| 真实粗类 | 动作数 | 个人相对位置组的类内 NLL 降幅 | 对全部动作粗类降幅的贡献 |",
        "|---|---:|---:|---:|",
    ]
    for name, group in section["coarse_personal_gain_by_class"].items():
        lines.append(
            f"| {name} | {group['support']:,} | {group['gain_within_class']:+.4f} | "
            f"{group['contribution_per_action']:+.5f} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "events"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_parquet(
        args.events, columns=["match_id", "event_id", "sub_event_name"]
    ).set_index(["match_id", "event_id"])["sub_event_name"]
    assert raw.index.is_unique
    summary, aligned = {}, {}
    lines = [
        "# 十三类事件预测开发结果",
        "",
        "英超训练 266 场，开发验证 57 场。球队与动作时刻已知。动作概率只使用可见历史与预测的球员、区域分布。",
    ]
    for view in ("main", "lag"):
        summary[view], aligned[view] = score_view(view, args.data, args.runs, raw)
        lines += ["", *report_view(view, summary[view])]
    e, i, j = np.intersect1d(aligned["main"][0], aligned["lag"][0], return_indices=True)
    np.testing.assert_array_equal(aligned["main"][1][i], aligned["lag"][1][j])
    np.testing.assert_array_equal(aligned["main"][2][i], aligned["lag"][2][j])
    matched = {}
    for v, ix in (("main", i), ("lag", j)):
        _, m, _, delta = aligned[v]
        matched[v] = contrast(delta[:, ix], m[ix], SEEDS[0])
    summary["common_role_to_personal"] = matched
    lines += [
        "",
        f"两视图共同可达的 {len(e):,} 条动作中，个人相对位置组的十三类 NLL 降幅分别为 "
        + gain_text(matched["main"])
        + " 和 "
        + gain_text(matched["lag"])
        + "。",
        "",
        "区间按比赛配对重抽 10,000 次。模型选轮和本次评分使用同一开发期。原地普通传球在主视图只有 18 条验证记录，参与总体指标与 Macro-F1。完整各类支持数、逐种子混淆矩阵和分项指标见[结果数据](summary.json)。[实验协议](../../EVENT_PROTOCOL.md)",
    ]
    (args.runs / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (args.runs / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
