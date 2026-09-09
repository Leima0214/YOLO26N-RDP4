# YOLO12N Official SVRDD7 100E 结果记录

- 实验：`YOLO12N_OFFICIAL_SVRDD7_100E_seed42`
- 协议：SVRDD7 Val 1000，imgsz=640，batch=32，seed=42，官方 YOLO12 Detect/DFL/loss，O2M 无 Voting
- 完成状态：100/100 epoch；train exit=0；pipeline exit=0；统一评价 exit=0
- 训练耗时：5626.93 s（93.782 min）
- 原生验证选出的 best epoch：75
- 参数量：2,569,413；训练图 FLOPs：6.5 GFLOPs
- Test：未读取

## 统一 O2M 无 Voting（best.pt）

| AP50:95 | AP50 | AP75 | AP-small | AP-medium | AP-large | AR100 |
|---:|---:|---:|---:|---:|---:|---:|
| 39.384 | 66.267 | 40.224 | 16.845 | 34.671 | 60.828 | 59.210 |

## 分类别

| 类别 | AP50:95 | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| LC | 39.798 | 65.026 | 40.934 | 63.861 |
| TC | 30.989 | 56.240 | 29.075 | 58.230 |
| AC | 46.776 | 75.067 | 48.337 | 64.955 |
| P | 26.757 | 51.775 | 24.357 | 47.130 |
| MC | 44.360 | 75.912 | 46.925 | 56.948 |
| LP | 50.054 | 76.211 | 54.904 | 65.279 |
| TP | 36.956 | 63.642 | 37.033 | 58.069 |

## 原生训练验证记录

- best epoch 75：mAP50:95=39.160，mAP50=66.517，Precision=71.666，Recall=60.975。
- 统一评价重新读取同一 `best.pt`，论文比较采用上面的统一 O2M 无 Voting结果。

## 权重校验

- best.pt SHA256: `76a35d4bd5b7c9ed499c30b9b272d2c206ea10f81273ce4b5ccd4b040d38690b`
- last.pt SHA256: `b856a8f9e90ba3597ab613b3553bb32b810948c11b6bb7029ae360dfb995f76a`
