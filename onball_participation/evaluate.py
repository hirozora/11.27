"""Aggregate matched seed scores. NLL gains are baseline minus proposed."""

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (20260715, 20260716, 20260717)
METRICS = ("conditional", "marginal", "joint")


def contrast(delta, match, seed):
    match_ids, idx = np.unique(match, return_inverse=True)
    n = np.bincount(idx)
    mass = np.bincount(idx, weights=delta.mean(0))
    draw = np.random.default_rng(seed).integers(len(n), size=(10000, len(n)))
    gains = mass[draw].sum(1) / n[draw].sum(1)
    return dict(
        gain=float(delta.mean()),
        by_seed=delta.mean(1).tolist(),
        interval=np.quantile(gains, [0.025, 0.975]).tolist(),
        actions=len(match),
        matches=len(match_ids),
    )


def write_report(report, path):
    lines = [
        "# 英超有球参与预测开发结果",
        "",
        "给定球队与动作时刻，预测执行者与动作起点的九宫格。三个种子的动作级损失取平均。",
        "",
    ]
    for view, title in (("main", "保留临近本方有球动作"), ("lag", "全部记录延迟两秒")):
        data = report[view]
        actor = data["actor"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"训练可达 {data['training_reached']:,}/{data['training_total']:,}，验证可达 "
                f"{data['validation_reached']:,}/{data['validation_total']:,}。执行者 NLL "
                f"{actor['actor']['mean']:.4f}，Top-1 {actor['top1']['mean']:.2%}，Top-3 {actor['top3']['mean']:.2%}。",
                "",
                "| 位置头 | 给定执行者的区域 NLL | 未知执行者的区域 NLL | 联合 NLL |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, scores in data["models"].items():
            lines.append(
                f"| {name} | {scores['conditional']:.4f} | {scores['marginal']:.4f} | {scores['joint']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| 对照（前者损失减后者损失） | 联合 NLL 增益及 95% 区间 | 未知执行者区域 NLL 增益及 95% 区间 |",
                "|---|---:|---:|",
            ]
        )
        for name, scores in data["comparisons"].items():
            cells = []
            for metric in ("joint", "marginal"):
                x = scores[metric]
                lo, hi = x["interval"]
                cells.append(f"{x['gain']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            lines.append(
                "| " + " | ".join([name.replace("_minus_", " 对 "), *cells]) + " |"
            )
        lines.append("")
    common = report["common_reachable_B_minus_D_joint"]
    lines.extend(
        [
            "## 两个视图的共同样本",
            "",
            f"在共同可达的 {common['main']['actions']:,} 个动作上，D 相对 B 的联合 NLL 降幅分别为 "
            f"{common['main']['gain']:.4f}（主视图）和 {common['lag']['gain']:.4f}（统一延迟）。",
            "",
            "A 只用球队局面，B 加位置组，C 再加本场个人履历，D 再加跨场个人区域分布。"
            "D_swap 在候选人位置组内轮换个人特征。每个种子内的执行者分布固定。"
            "所有区间按比赛配对重抽 10,000 次。模型选择与评分使用同一开发验证期。",
            "",
            "[逐种子比较与区间](summary.json) · [输入扰动核验](view_checks.json) · [实验计划](../EXPERIMENT_PLAN.md)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=Path)
    args = p.parse_args()
    root = args.results
    report = {}
    aligned = {}
    for view, kinds in [
        ("main", ["A", "B", "C", "D", "D_swap"]),
        ("lag", ["B", "D", "D_swap"]),
    ]:
        with np.load(root / (view + ".npz")) as dataset:
            val = dataset["val"]
            reached = val & (dataset["slot"] >= 0)
            events = dataset["event"][reached]
            match = dataset["match"][reached]
            section = {
                "validation_total": int(val.sum()),
                "validation_reached": int(reached.sum()),
                "training_total": int((~val).sum()),
                "training_reached": int(((~val) & (dataset["slot"] >= 0)).sum()),
            }
        arrays = []
        for seed in SEEDS:
            with np.load(root / f"{view}_{seed}/validation.npz") as f:
                assert np.array_equal(events, f["event"]) and np.array_equal(
                    match, f["match"]
                )
                arrays.append({k: f[k] for k in ["actor", "top1", "top3", *kinds]})
        scores = {k: np.stack([a[k] for a in arrays]).astype(float) for k in arrays[0]}
        section["actor"] = {
            k: dict(mean=float(scores[k].mean()), by_seed=scores[k].mean(1).tolist())
            for k in ("actor", "top1", "top3")
        }
        section["models"] = {
            k: {
                name: float(scores[k][:, :, j].mean()) for j, name in enumerate(METRICS)
            }
            for k in kinds
        }
        comparisons = [("B", "D"), ("D_swap", "D")]
        if view == "main":
            comparisons += [("B", "C"), ("C", "D"), ("A", "B")]
        section["comparisons"] = {
            f"{a}_minus_{b}": {
                name: contrast(scores[a][:, :, j] - scores[b][:, :, j], match, 20260715)
                for j, name in enumerate(METRICS)
            }
            for a, b in comparisons
        }
        report[view] = section
        aligned[view] = (events, match, scores)
    e, i, j = np.intersect1d(aligned["main"][0], aligned["lag"][0], return_indices=True)
    assert np.array_equal(aligned["main"][1][i], aligned["lag"][1][j])
    same = {}
    for name, ix in [("main", i), ("lag", j)]:
        _, match, scores = aligned[name]
        same[name] = contrast(
            scores["B"][:, ix, 2] - scores["D"][:, ix, 2], match[ix], 20260715
        )
    report["common_reachable_B_minus_D_joint"] = same
    (root / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    write_report(report, root / "RESULTS.md")
    print(json.dumps(same, indent=2))


if __name__ == "__main__":
    main()
