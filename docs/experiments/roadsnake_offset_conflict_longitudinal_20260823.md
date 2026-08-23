# RoadSnake offset纵向梯度冲突免训练审计（2026-08-23）

## 结论

**RoadSnake-GCS：No-Go，不进入30E。**

此前D0在seed42 best checkpoint、12个关闭增强的batch上观察到offset分类/定位冲突率约83%。本次扩大到两个正式seed、四个训练阶段以及严格配对的真实训练增强batch后，该现象没有稳定复现。冲突方向显著依赖checkpoint、batch和seed，不能作为开发梯度手术方法的充分因果证据。

## 冻结协议

- 数据：Japan4-cleanV3 train split；禁止Test。
- 权重：RoadSnake-R1 seed42与修复初始化后的seed43。
- 检查点：`epoch5.pt`、`epoch20.pt`、`epoch50.pt`、`best.pt`。
- best训练轨迹：seed42 E57；seed43 E70。
- 共同增强batch：32个batch，batch=8，共803个增强后实例。
- 两个seed、八个checkpoint严格复用同一组增强后的图像和标签张量。
- 训练增强：mosaic=1.0、translate=0.1、scale=0.5、fliplr=0.5及原HSV配置。
- 使用AMP forward；BN running statistics冻结。
- 只计算O2M的`box+dfl`与`cls`对RoadSnake offset参数的梯度。
- 不创建optimizer、不调用参数更新；八个checkpoint审计前后模型hash全部一致。
- offset参数量：2,890。

## 总体offset梯度结果

| Seed | Checkpoint | Conflict fraction | Cosine median | cls/loc norm ratio median |
|---:|:---|---:|---:|---:|
| 42 | E5 | 0.6875 | -0.12334 | 0.7634 |
| 42 | E20 | 0.3750 | +0.05856 | 0.7553 |
| 42 | E50 | 0.5938 | -0.05702 | 0.7762 |
| 42 | best (E57) | 0.5313 | -0.00561 | 0.7432 |
| 43 | E5 | 0.5000 | -0.00549 | 0.8402 |
| 43 | E20 | 0.3125 | +0.14191 | 0.8792 |
| 43 | E50 | 0.4063 | +0.03733 | 0.9798 |
| 43 | best (E70) | 0.5313 | -0.00433 | 0.9172 |

梯度量级没有相差数个数量级，所有梯度均有限；失败原因不是norm失衡，而是冲突方向不稳定。

预先冻结的门槛要求每个seed至少3/4 checkpoint冲突率大于50%、cosine中位数多数为负、早期也出现、且至少两个类别支持。共同batch复核结果：

- seed42：通过（3/4 checkpoint）；
- seed43：失败（仅1/4 checkpoint严格超过50%，2/4中位数为负，早期证据不足）；
- 总判定：No-Go。

## 逐类汇总

| Seed | Class | Conflict fraction | Cosine median |
|---:|:---|---:|---:|
| 42 | D00 | 0.5313 | -0.02345 |
| 42 | D10 | 0.4098 | +0.07939 |
| 42 | D20 | 0.5313 | -0.01864 |
| 42 | D40 | 0.5062 | -0.00026 |
| 43 | D00 | 0.4922 | +0.00465 |
| 43 | D10 | 0.5041 | -0.00160 |
| 43 | D20 | 0.4375 | +0.04822 |
| 43 | D40 | 0.6104 | -0.03310 |

除D40外，不同seed支持的类别并不一致；多数中位数非常接近0，不支持一个稳定的offset冲突机制。

## 配对稳定性

- 相同batch、相同checkpoint阶段下，两seed冲突符号一致率：56.25%。
- 两seed cosine Spearman：0.145。
- 分阶段符号一致率仅50.0%到62.5%。

因此，原D0的83%更接近一个有限batch/单checkpoint快照，而不是RoadSnake训练全过程的结构性冲突。

## 决策边界

1. 不实现RoadSnake-GCS。
2. 不扫描PCGrad、CAGrad、投影频率、epsilon或loss权重。
3. 不用seed43单独通过的旧抽样结果为GCS辩护；最终结论只采用共同32-batch配对审计。
4. 本审计不否定RoadSnake-R1本身。R1的两seed检测收益仍成立；这里只是否定“offset稳定梯度冲突”作为第二创新的依据。

## 工件

- 脚本：`scripts/audit_roadsnake_offset_conflict_longitudinal.py`
- 最终报告：`reports/roadsnake_offset_conflict_longitudinal_shared32_20260823/summary.json`
- 原始梯度：`reports/roadsnake_offset_conflict_longitudinal_shared32_20260823/gradient_batches.csv`
- checkpoint汇总：`reports/roadsnake_offset_conflict_longitudinal_shared32_20260823/checkpoint_summary.csv`
- 逐类汇总：`reports/roadsnake_offset_conflict_longitudinal_shared32_20260823/class_summary.csv`
- 完整性审计：`reports/roadsnake_offset_conflict_longitudinal_shared32_20260823/checkpoint_integrity.csv`

