# 执行者与位置联合训练

原计划后续路线第 1 项包含这个实验。本轮用首轮的顺序编码器、候选人和两个数据视图，检验位置损失进入共享网络能否改善执行者概率。

两组均在每个候选人的 64 维隐藏表示上接同一个九分类线性头。对照组对位置头输入停止梯度，执行者参数只接受执行者损失，位置头同步拟合。联合组允许真实执行者对应的位置交叉熵更新共享层，与执行者交叉熵等权相加。真实执行者和起点只用于选取训练损失及评分。推断时对所有有效候选输出概率。两组共享初始化、批序、dropout、预算和验证准则。分别对执行者网络和位置线性头的梯度裁剪至 5，保证对照组的执行者更新与首轮可比较。

三个种子、120 轮、每 4 轮按执行者验证 NLL 选优，其余设置同首轮。预设主指标为主视图执行者 NLL 的配对改善，次指标为 Top-1、Top-3、条件与边缘区域 NLL 及联合 NLL。两秒视图复核方向。按比赛配对重抽 10,000 次，区间反映验证比赛间波动。只使用原开发划分。

## 复算

从本目录运行。两组的优化器均为 AdamW，位置头和执行者网络各自按梯度范数 5 裁剪。执行者和位置头同时前向，两个训练条件使用相同的随机初始化与相同的执行者 dropout 随机序列。

```bash
for view in main lag; do
  for seed in 20260715 20260716 20260717; do
    for arm in detached joint; do
      python train_joint.py --data results/"$view".npz --seed "$seed" --device cuda:0 --arm "$arm" --out results/joint_training/"${view}_${arm}_${seed}"
    done
  done
done
python evaluate_joint.py --runs results/joint_training --first-round results
```
