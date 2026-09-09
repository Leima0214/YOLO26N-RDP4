# YOLO12n B0 Diagnostic Report

## 1. Baseline

**FACT:** 本诊断只加载 `YOLO12N_OFFICIAL_SVRDD7_100E_seed42/weights/best.pt`，使用 SVRDD7 Val 1000 张、2612 个 GT；没有训练、微调、改模型、改 loss 或改 NMS，Test 未读取。模型为官方非 end-to-end `Detect`，2,569,413 参数、6.486 GFLOPs，best epoch 为 75。

统一结果：AP50:95 `39.384`、AP50 `66.267`、AP75 `40.224`、AP-small `16.845`、AP-medium `34.671`、AP-large `60.828`、AR100 `59.210`。

附件中的 D00/D10/D20/D40 并非 SVRDD7 类别。本数据集实际七类是 LC、TC、AC、P、MC、LP、TP，本诊断没有虚构跨数据集映射。

## 2. Per-class performance

| 类别 | AP50:95 | AP50 | AP75 | R@.50 | R@.75 |
|---|---:|---:|---:|---:|---:|
| LC | 39.798 | 65.026 | 40.934 | 92.007 | 70.578 |
| TC | 30.989 | 56.240 | 29.075 | 91.148 | 63.636 |
| AC | 46.776 | 75.067 | 48.337 | 94.595 | 70.270 |
| P | 26.757 | 51.775 | 24.357 | 80.000 | 48.696 |
| MC | 44.360 | 75.912 | 46.925 | 91.563 | 62.035 |
| LP | 50.054 | 76.211 | 54.904 | 93.642 | 70.906 |
| TP | 36.956 | 63.642 | 37.033 | 89.914 | 63.689 |

**FACT:** P 是最弱类别；TC 次之。P 同时具有最低 AP、R@.50 和 R@.75，不能只解释为类别排序问题。

## 3. Error decomposition

在用于案例分解的 `confidence >= 0.25` 工作点：

| 事件 | 数量 | 分解事件占比 |
|---|---:|---:|
| TP | 1759 | 48.63% |
| missed GT | 853 | 23.58% |
| background FP | 459 | 12.69% |
| localization error | 322 | 8.90% |
| duplicate | 132 | 3.65% |
| classification error | 92 | 2.54% |

**FACT:** 最大失败项是 missed GT，其次是 background FP 和 localization error。0.25 工作点的 miss 不能等同于模型没有候选，因为正式 AP 使用完整 score 排序。

## 4. IoU / localization analysis

| 类别 | AP50 | AP75 | AP50-AP75 | R@.50 | R@.75 |
|---|---:|---:|---:|---:|---:|
| MC | 75.912 | 46.925 | 28.986 | 91.563 | 62.035 |
| P | 51.775 | 24.357 | 27.418 | 80.000 | 48.696 |
| TC | 56.240 | 29.075 | 27.165 | 91.148 | 63.636 |
| AC | 75.067 | 48.337 | 26.730 | 94.595 | 70.270 |
| TP | 63.642 | 37.033 | 26.609 | 89.914 | 63.689 |
| overall | 66.267 | 40.224 | 26.044 | 91.539 | 66.309 |
| LC | 65.026 | 40.934 | 24.091 | 92.007 | 70.578 |
| LP | 76.211 | 54.904 | 21.307 | 93.642 | 70.906 |

**INFERENCE:** 严格定位仍有明显空间，但 AP50→AP75 差距广泛存在，并非某一细长类别独有。MC 的差距最大，P 的绝对定位与召回最弱。

## 5. Aspect-ratio analysis

| 长宽比 | GT | R@.50 | R@.75 | mean raw-best IoU | raw IoU≥.50 |
|---|---:|---:|---:|---:|---:|
| [1,2) | 821 | 91.96% | 69.67% | 0.855 | 96.95% |
| [2,4) | 865 | 93.06% | 70.64% | 0.854 | 97.69% |
| [4,8) | 440 | 87.27% | 53.64% | 0.778 | 93.41% |
| [8,16) | 324 | 91.36% | 67.28% | 0.818 | 95.06% |
| [16,+∞) | 162 | 93.21% | 58.64% | 0.792 | 95.68% |

**FACT:** 性能没有随长宽比单调下降；[4,8) 反而是最差桶，而 [8,16) 的 R@.75 回升到 67.28%。

**INFERENCE:** 当前数据不支持把问题定义为“越细长越难”的单调 Strip 定位瓶颈。

## 6. Scale analysis

| 尺度 | GT | R@.50 | R@.75 | raw-best IoU | raw IoU≥.50 | raw IoU≥.75 |
|---|---:|---:|---:|---:|---:|---:|
| small | 548 | 80.29% | 38.50% | 0.694 | 85.40% | 46.72% |
| medium | 980 | 93.57% | 67.04% | 0.843 | 98.67% | 83.16% |
| large | 1084 | 95.39% | 79.70% | 0.895 | 99.63% | 92.71% |

**FACT:** small 是清晰、幅度很大的瓶颈。与 medium 相比，small 的 raw IoU≥.50 少 13.27 个百分点，R@.75 少 28.54 个百分点。面积四分位也呈一致趋势：Q1 R@.75 为 41.65%，Q4 为 81.62%。

**INFERENCE:** 这里同时存在小目标候选生成和严格定位问题。

**LIMITATION:** 本诊断没有观察 P2 特征，因此不能证明信息在 P2→P3 下采样时丢失，也不能直接给 P2 检测层 GO。

## 7. Candidate survival

在 canonical score floor 0.001 下：

| 结果 | GT 数量 | 占比 |
|---|---:|---:|
| final TP@.50 | 2391 | 91.54% |
| raw geometry failure | 97 | 3.71% |
| final localization/assignment | 48 | 1.84% |
| confidence floor loss | 40 | 1.53% |
| NMS/maxDet loss | 26 | 1.00% |
| classification mismatch | 10 | 0.38% |

**FACT:** NMS 不是主要损失源。score floor 下 R@.50 为 91.54%，而 0.25 工作点只有 67.34% GT 成为 TP，说明大量正确候选位于较低置信度区间。

## 8. Score-IoU alignment

候选先按“同类别、最大 IoU”唯一归属到 GT，避免同图其他同类目标污染排序：

- Pearson：`0.457`
- Spearman：`0.505`
- 最高分候选平均 IoU：`0.689`
- 最佳几何候选平均 IoU：`0.833`
- 两者 IoU 差值超过 0.10 的 GT：`42.34%`
- 已有 IoU≥0.75 候选但其最高 score<0.25：`26.49%`

GT-oracle 保持全部框、类别和数量不变，只把 score 替换为同类 GT IoU：

| 评价 | AP | AP50 | AP75 | AP-small | AP-medium | AP-large | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原始 score | 39.385 | 66.268 | 40.224 | 16.846 | 34.673 | 60.828 | 59.210 |
| GT-IoU oracle | 56.669 | 81.980 | 63.665 | 32.250 | 55.656 | 72.878 | 59.230 |
| 上限差值 | +17.284 | +15.712 | +23.441 | +15.404 | +20.983 | +12.050 | +0.019 |

**FACT:** score–IoU 相关性为中等，不应描述为“完全失配”；但候选质量排序存在很大的可恢复上限，尤其是 AP75。AR100 几乎不变，说明 oracle 的收益来自排序而不是新增候选。

**INFERENCE:** Quality/Task Alignment 是当前唯一获得直接机制支持的模型改进方向。它应针对 score 的定位质量表达，不应改 NMS，也不应仅增加通用 Head 容量。

**LIMITATION:** GT oracle 是不可部署的理想上限，+17.284 AP 绝不是预期真实增益。

## 9. Background FP analysis

在 confidence≥0.25 的 1005 个 FP 中，459 个（45.67%）按 IoU 自动归为 background FP。分数分布为：0.25–0.40 共 264，0.40–0.60 共 123，0.60–0.80 共 55，0.80–1.00 共 17。

**FACT:** 自动 background FP 数量较高。

**MANUAL REVIEW:** 固定 seed 导出的高分样例中可见路面接缝、修补边缘、道路边缘、阴影及弱纹理区域；同时有些红框可能是真实但未标注的损伤或部分覆盖标注。

**INFERENCE:** 这支持继续做系统人工语义复核，但现阶段不能把 45.67% 全部归因于高频背景，也不能据此直接给 SET GO。

## 10. FPN-level analysis

- small：最佳 raw 候选 96.90% 来自 P3，最终最佳候选 97.26% 来自 P3。
- medium：最佳 raw 候选 76.12% 来自 P3，21.73% 来自 P4。
- large：最佳 raw 候选主要来自 P4（52.68%）和 P5（42.71%）。

**FACT:** P3/P4/P5 来源通过原生 feature tensor 长度和 NMS kept-anchor index 直接追踪，归因可靠。

**INFERENCE:** small 几乎完全依赖 P3。该事实说明 P3 是 small 的责任层，但仍不能证明必须新增 P2。

## 11. Failure cases

案例已按 small miss、高长宽比 miss、high-IoU low-score、NMS competition、localization error 和 background FP 分目录保存。background FP 仅标记为人工复核候选，不统计为漏标率。

## 12. Main bottlenecks

1. **质量排序/置信度表达：** 已有候选的 IoU 与 score 对齐不充分，oracle 上限大，AP75 潜力尤其明显。
2. **小目标候选和定位：** small 的 raw-best IoU、候选存在率和 R@.75 均显著低于 medium/large。
3. **背景误检：** 数量较高，但语义成因和标注歧义尚未系统确认。

## 13. Alternative explanations / limitations

- 单 seed、单 Val 的诊断只能授权下一轮实验，不能证明因果。
- box annotation 无法证明曲线形态或频域背景是错误原因。
- GT oracle 同时使用不可获得的类别内定位真值，只代表上限。
- confidence=0.25 用于可解释错误分解；正式 AP 仍使用 0.001 score floor 的完整排序。
- scale 诊断没有读取 backbone P2/P3 feature response，因此 P2 路线仍缺一项必要证据。

## 14. Route decision

| 路线 | 裁决 | 依据 |
|---|---|---|
| A Strip Localization | NO-GO | 长宽比趋势不单调，问题并非稳定集中于极细长目标。 |
| B Task/Quality-Aligned Head | **QUALIFIED GO** | 中等 score–IoU 相关性，但 42.34% GT 存在>0.10 排序质量差；GT-oracle AP +17.284、AP75 +23.441。 |
| C SET/Background Suppression | REVIEW | 自动 background FP 占 FP 45.67%，但视觉样例含标注歧义，尚未完成系统语义归类。 |
| D Multi-kernel | LOW PRIORITY | 尺度差异主要集中于 small，不能证明通用多感受野是直接解。 |
| E P2/Tiny Layer | NO-GO NOW | small 瓶颈明确，但尚未证明 P2→P3 信息丢失。 |
| F DSConv/DCNv4 | NO-GO | 当前 box 证据不支持复杂曲线/可变形算子。 |
| G Class-balanced loss | AUXILIARY | P/TC 较弱，但错误不只来自频次或分类。 |

## 15. Top-3 next experiments

只有一个模型方向达到有条件 GO，不为凑数增加 Top 2/Top 3。

### Top 1：轻量 Quality-Aligned Head

1. 候选方法：在 YOLO12n 原生 P3/P4/P5 Detect 内增加轻量、显式定位质量预测或 task interaction，只修正分类 score 对框质量的表达。
2. 证据：42.34% GT 的最高分候选与最佳候选相差>0.10 IoU；GT-IoU oracle AP +17.284、AP75 +23.441；NMS 损失仅 1.00%。
3. 针对问题：高质量框排序不足和低置信度生存。
4. 优先原因：它直接针对当前唯一被干预上限验证的机制。
5. 最大风险：oracle 使用 GT，真实质量预测可能学不到足够准确的排序。
6. 第一轮形式：B0、参数匹配普通 Head、Quality-Aligned Head 三组严格配对；保持 YOLO12 原生 TAL、DFL、NMS和其余训练设置。
7. 修改位置：Head。
8. 推理成本：预计增加，必须实测 Params、GFLOPs、延迟；首轮控制为轻量分支。
9. 消融：普通增容与质量机制必须分离。
10. Paper 适合度：有潜力成为主创新，但需真实 AP、AP75 和多 seed 证明。

在开发模型前还有两个诊断缺口：对 background FP 做盲化人工语义分类；若要考虑 P2，先做 P2→P3 feature survival。它们不是已经获准的第二、第三模型实验。
