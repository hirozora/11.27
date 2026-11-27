"""Score the two gradient treatments on identical held-out queries."""

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate import SEEDS, contrast

METRICS = (
    "actor_nll",
    "top1",
    "top3",
    "conditional_zone_nll",
    "marginal_zone_nll",
    "joint_nll",
)
LABELS = (
    "执行者 NLL",
    "执行者 Top-1",
    "执行者 Top-3",
    "条件区域 NLL",
    "边缘区域 NLL",
    "联合 NLL",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--first-round", required=True, type=Path)
    args = parser.parse_args()
    summary = {}
    lines = [
        "# 执行者与位置联合训练结果",
        "",
        "两组在同一顺序编码器上训练相同的执行者头与位置头，区别是位置损失能否更新共享表示。每组按执行者验证 NLL 选择 checkpoint。以下均为三个种子的开发验证结果。",
        "",
    ]
    for view, title in [("main", "保留临近本方有球动作"), ("lag", "所有记录延迟两秒")]:
        paired = {"detached": [], "joint": []}
        epochs = {}
        replay = []
        for seed in SEEDS:
            with np.load(args.first_round / f"{view}_{seed}/validation.npz") as f:
                event, match, first = f["event"], f["match"], f["actor"]
            for arm in paired:
                path = args.runs / f"{view}_{arm}_{seed}"
                curve = json.loads((path / "curve.json").read_text())
                assert [e for e, _ in curve] == list(range(4, 121, 4))
                epochs[f"{arm}_{seed}"] = min(curve, key=lambda x: x[1])[0]
                with np.load(path / "validation.npz") as f:
                    assert np.array_equal(f["event"], event) and np.array_equal(
                        f["match"], match
                    )
                    scores = f["scores"]
                assert (
                    np.isfinite(scores).all()
                    and abs(scores[:, 0].mean(dtype=float) - min(v for _, v in curve))
                    < 1e-5
                )
                if arm == "detached":
                    replay.append(float(np.abs(scores[:, 0] - first).max()))
                paired[arm].append(scores.astype(float))
        base = np.stack(paired["detached"])
        joint = np.stack(paired["joint"])
        lines.extend(
            [
                f"## {title}",
                "",
                f"57 场比赛，{len(match):,} 个可达动作。差值统一为正代表联合训练改善。准确率差用百分点，损失用 nats/动作。",
                "",
                "| 指标 | 仅执行者梯度 | 联合梯度 | 改善 | 按比赛配对的 95% 区间 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        values = {}
        for j, (key, label) in enumerate(zip(METRICS, LABELS)):
            delta = base[:, :, j] - joint[:, :, j]
            accuracy = key.startswith("top")
            change = contrast(-delta if accuracy else delta, match, 20260715)
            b = float(base[:, :, j].mean())
            t = float(joint[:, :, j].mean())
            values[key] = dict(detached=b, joint=t, paired_improvement=change)
            scale = 100 if accuracy else 1
            lo, hi = np.array(change["interval"]) * scale
            left, right = (
                (f"{b:.2%}", f"{t:.2%}") if accuracy else (f"{b:.4f}", f"{t:.4f}")
            )
            lines.append(
                f"| {label} | {left} | {right} | {scale*change['gain']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |"
            )
        lines.append("")
        summary[view] = {
            "metrics": values,
            "best_epochs": epochs,
            "max_abs_actor_nll_difference_vs_first_round_by_seed": replay,
        }
    lines.extend(
        [
            "验证集同时用于选轮与本次评估，区间表示按比赛重采样的不确定性。真实执行者与区域参与损失计算，预测执行者时只使用预测时刻可见的特征。",
            "",
            "[协议与复算命令](../../JOINT_PROTOCOL.md) · [逐种子结果](summary.json)",
        ]
    )
    (args.runs / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (args.runs / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
