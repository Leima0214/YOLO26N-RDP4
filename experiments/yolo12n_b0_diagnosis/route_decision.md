# Route decision

| 路线 | 裁决 | 核心证据 |
|---|---|---|
| A Strip Localization Head | NO-GO | aspect-ratio 性能非单调；[4,8) 比 [8,16) 更差。 |
| B Task / Quality-Aligned Head | **QUALIFIED GO** | 同 GT 排序 Spearman 0.505；42.34% GT 的质量差>0.10；GT-IoU oracle AP +17.284、AP75 +23.441，AR100 基本不变。 |
| C SET / Background Spectral Suppression | REVIEW | 自动 background FP 占 FP 45.67%，但样例存在标注歧义，尚不能证明频域背景是主因。 |
| D GFB / PKI-lite / Multi-kernel | LOW PRIORITY | small 差距突出，large 并非共同瓶颈。 |
| E P2 / Tiny Object Layer | NO-GO NOW | small 瓶颈明确；缺少 P2→P3 特征存活证据。 |
| F DSConv / DCNv4 | NO-GO | box 诊断不能证明曲线形态导致 raw geometry failure。 |
| G Class-balanced / Hard-sample Loss | AUXILIARY | 类别差异存在，但不是纯分类/频次问题。 |

## Evidence-ranked next experiments

只有一个模型方向达到有条件 GO，不为凑数填充 Top 2/Top 3。

### Top 1：轻量 Quality-Aligned Head

- **具体问题：** 已有高质量候选的 score 排序不足，尤其影响 AP75。
- **为何优先：** 保持框和类别不变的 oracle 显示显著 AP 上限，而 NMS 损失只有 1.00%。
- **第一轮：** B0 vs 参数匹配普通 Head vs Quality-Aligned Head；保持 YOLO12 原生 TAL、DFL、NMS、数据和训练配方。
- **修改范围：** Head。
- **推理成本：** 会增加，需实测；首轮使用轻量分支。
- **消融：** 分离普通增容和质量机制。
- **Paper 角色：** 候选主创新；只有真实配对、多 seed 的 AP/AP75 收益才能确认。
- **最大风险：** GT oracle 远强于可学习质量预测，理论上限不能当预期涨点。

## Pending diagnostic routes

- SET：先对高置信 background FP 做盲化人工语义复核。
- P2：先做 P2→P3 early-detail survival；当前只能确认 small/P3 瓶颈。

未启动任何训练。
