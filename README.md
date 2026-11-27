# 足球事件流中的球员有球参与预测

给定动作球队和时刻，用此前可见的 Wyscout 事件预测下一名执行者及动作起点的九宫格。当前结果来自英超 2017/18 赛季的 266 场训练和 57 场开发验证比赛。

## 当前实验

- [研究概要](onball_participation/RESEARCH_OVERVIEW.md)与[实验计划](onball_participation/EXPERIMENT_PLAN.md)说明候选名单、历史可见性、比赛切分和评估方法。
- [个人区域历史的四组对照](onball_participation/results/RESULTS.md)显示，本场个人历史贡献了联合预测改善的大部分。
- [位置监督与执行者联合训练](onball_participation/results/joint_training/RESULTS.md)在主视图将执行者 Top-1 从 31.70% 提高到 32.21%。[训练协议](onball_participation/JOINT_PROTOCOL.md)描述共享表示、梯度对照和三种子的复算方法。

## 历史结果

- [五任务 HGT 对照与消融](historical_results/five_task_hgt/tables.md)和[早期训练曲线](historical_results/five_task_hgt/early_training.md)记录了图与顺序模型的比较。
- [起点未知的事件预测](historical_results/type_unknown_start.md)、[已知起点对照](historical_results/type_known_start.md)及[球员模型结构比较](historical_results/england_structures.md)保留了早期任务定义下的开发结果。旧视图曾用目标起点筛选重合记录，其成绩与当前视图单列。
- [十类动作混淆矩阵](event_type_diagnostics/confusion.png)、[逐格计数](event_type_diagnostics/confusion.csv)及[技术标签诊断](event_type_diagnostics/tag_granularity.json)支持动作细分与球员信息的分析。早期诊断的详细数值见[结果表](event_type_diagnostics/results.json)。

## 使用方法

在仓库根目录安装 `scene_regularity_release/requirements.txt` 的依赖和与运行环境匹配的 PyTorch 2.6。当前研究目录为 `onball_participation/`。两轮协议中的训练命令以该目录为工作目录。使用保留的验证预测可直接重算表格：

```bash
cd onball_participation
python evaluate.py --results results
python evaluate_joint.py --runs results/joint_training --first-round results
```

重新训练时，可在 `onball_participation/` 下合并随仓库发布的输入分片，两组命令分别还原主视图和延迟两秒视图。复算已有验证结果可跳过这一步。

```bash
cat results/view_parts/main.npz.part-* > results/main.npz
cat results/view_parts/lag.npz.part-* > results/lag.npz
```

如需从原始记录重新建表，把 Wyscout 数据放入 `scene_regularity_release/data/raw/`，保持 `events/events_England.json`、`matches/matches_England.json` 和 `entities/{teams,players}.json` 的目录结构。运行以下命令后，再执行实验计划中的建表与输入扰动检验：

```bash
cd scene_regularity_release
python src/build_actions.py --config configs/data/England_nontest.json
python src/build_model_data.py --config configs/model/England_nontest.json
```
