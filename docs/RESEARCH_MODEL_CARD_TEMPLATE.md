# 研究模型卡模板

> **模板状态：未填写 / 未授权 / 不代表仓库中存在研究模型。**
>
> 本模板用于 Sprint 2 Phase B 的受控审查。所有占位符均完成、研究证据可追溯且授权清单获得批准前，不得发布真实工件，不得启用公开研究模式，也不得从本模板推断临床性能。

v1 loader 还要求最终模型卡包含以下两行机器可检查的纯文本；第一行的值必须与 manifest 完全一致：

```text
model_id: <required>
clinical_use: false
```

## 1. 模型身份与发布状态

| 字段 | 内容 |
| --- | --- |
| 模型 ID | `<required>` |
| 模型版本 | `<required>` |
| 实验 ID | `<required; one experiment per ID>` |
| 模型 stage | `research_locked`（v1 唯一支持值） |
| 后端与格式 | `pytorch_state_dict`（v1 不支持 TorchScript/ONNX/其他格式） |
| 已知架构 | `bilateral_linear_v1` |
| Runtime | `PyTorch 2.13.0 / CPU / float32` |
| 输入/输出 shape | `[1, 9] -> [1, 2]`，单案例 |
| 来源仓库 commit | `<required>` |
| 工件 SHA-256 | `<required>` |
| 预处理统计 SHA-256 | `<required>` |
| 授权请求 reference / scope | `<required> / local_adapter_evaluation` |
| 包外审批 request SHA-256 | `<required>` |
| 包外 receipt ID / 到期时间 | `<required>` |
| 授权决定 | `not_reviewed`（外部治理登记状态；不是 manifest 字段，不可加载） |
| 允许的使用范围 | 无 |
| 是否进入公开仓库 | **否，除非另有独立的公开发布审批记录** |
| 是否由公开 `/predict` 使用 | **否** |
| 临床使用 | **否** |

关联授权记录：[研究工件授权清单](RESEARCH_ARTIFACT_AUTHORIZATION.md)。

## 2. 预期用途

请写明经批准的具体用途、使用人群、运行环境和输出解释：

`<required>`

最低要求：说明该模型是研究性内部模型、离线工程验证模型，还是获得其他明确批准的工件。不要使用“可用于临床”“辅助诊断”或“真实风险概率”等措辞，除非存在与之匹配的独立证据和治理批准。

## 3. 禁止用途

- 诊断、筛查、治疗决策或患者沟通；
- 对真实个人排序或采取行动；
- 将内部验证当作外部或前瞻性验证；
- 将未经校准的输出称为绝对风险概率；
- 超出授权范围复制、上传、演示或分发工件；
- 把 adapter parity 或运行速度描述为医疗效果。

如有额外禁止用途：`<required>`

## 4. 研究问题与预测目标

| 字段 | 内容 |
| --- | --- |
| 研究问题 | `<required>` |
| 预测时点 | `<required>` |
| 预测时间范围 | `<required>` |
| OD 目标定义 | `<required>` |
| OS 目标定义 | `<required>` |
| 标签来源 | `<required>` |
| 目标泄漏排除规则 | `<required>` |

## 5. 数据与队列

只记录经批准的聚合信息，不粘贴参与者级记录、路径或拆分 ID。

| 字段 | 内容 |
| --- | --- |
| 数据来源与授权依据 | `<required>` |
| 纳入/排除标准 | `<required>` |
| 总样本量 | `<required and evidenced>` |
| OD/OS 事件数 | `<required and evidenced>` |
| 随访间隔 | `<required>` |
| 缺失值处理 | `<required>` |
| 数据拆分单位 | `<participant / eye / ...>` |
| 防止同一参与者跨折泄漏的方法 | `<required>` |
| 人群与采集限制 | `<required>` |

## 6. 输入、预处理与特征契约

| 字段 | 内容 |
| --- | --- |
| 特征契约版本 | `<required>` |
| 特征顺序 | `<required or approved manifest reference>` |
| 单位 | `<required>` |
| 归一化来源 | `<training fold only / locked export / ...>` |
| 图像预处理 | `<required if applicable>` |
| 缺失值与异常值规则 | `<required>` |
| dtype 与输入 shape | `<required>` |
| 明确排除的字段 | `<identifiers, leakage variables, ...>` |

预处理统计只有在授权明确覆盖时才可导出；聚合统计不天然等于无隐私风险。

## 7. 模型与训练

| 字段 | 内容 |
| --- | --- |
| 架构 | `<required>` |
| 双眼/单眼设计 | `<required>` |
| 图像与临床融合方式 | `<required if applicable>` |
| 损失函数 | `<required>` |
| 优化器与主要超参数 | `<required>` |
| 随机种子策略 | `<required>` |
| checkpoint 选择规则 | `<required>` |
| 训练硬件与框架版本 | `<required>` |

## 8. 验证设计

| 字段 | 内容 |
| --- | --- |
| 验证类型 | `<internal CV / temporal / external / prospective>` |
| 重复次数与折数 | `<required>` |
| OOF/stacking 生成规则 | `<required if applicable>` |
| 超参数选择与最终评估隔离 | `<required>` |
| 置信区间方法 | `<required>` |
| 显著性检验与多重比较 | `<required if applicable>` |
| 外部验证 | `<not performed / details>` |
| 校准评估 | `<not performed / details>` |
| 亚组/公平性评估 | `<not performed / details>` |

内部重复交叉验证必须明确写作“内部验证”，不能简称为外部验证或真实世界验证。

## 9. 研究指标证据

每一行只描述一个明确实验，不得合并不同样本口径、模态、队列或消融结果。

| 实验 ID | 队列 N / 事件数 | 模型/对照 | 指标与区间 | 聚合结果文件 | 计算脚本 | 论文表格 | 来源 commit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `<required>` | `<required>` | `<required>` | `<required>` | `<required>` | `<required>` | `<required>` | `<required>` |

在全部证据列填写并复核前，指标状态为 `UNVERIFIED_FOR_PUBLIC_CLAIM`。合成 AUC 必须留在公开合成模型卡中，不得进入本表作为研究效果。

## 10. 导出工件与 adapter parity

| 字段 | 内容 |
| --- | --- |
| manifest schema 版本 | `<required>` |
| 导出工具与 commit | `<required>` |
| golden vector 来源 | `<synthetic / approved aggregate-safe fixture>` |
| 样例数 | `<required>` |
| 预设容差 | `<required before evaluation>` |
| 最大绝对误差 | `<required>` |
| 多个黄金向量一致性 | `<pass/fail>` |
| 重复推理确定性 | `<pass/fail>` |
| shape/dtype/NaN/Inf 失败测试 | `<pass/fail>` |
| receipt/checksum/inventory/stage/schema 失败测试 | `<pass/fail>` |
| comparison report | `<path and SHA>` |

黄金向量 parity 只证明获批包与 runtime 在给定输入上满足数值容差，不证明模型正确、无偏或具有临床效用。使用合成 fixture 得到的 parity 不能替代真实 checkpoint 的授权后验证。当前 v1 不支持 batch；批量一致性属于后续合同。

## 11. 工程性能

| 环境 | 工件大小 | 冷载入 | 首次推理 | Warm P50/P95/P99 | RSS | 批量大小 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<required>` | `<required>` | `<required>` | `<required>` | `<required>` | `<required>` | `<required>` |

必须注明是否包含网络、反向代理、容器和并发；本地进程内顺序测试不得表述为生产容量或 SLA。

## 12. 局限、风险与治理

至少覆盖：

- 人群代表性、样本量和事件数限制；
- 缺失、采集设备和时间漂移；
- 校准、亚组、公平性和外部验证缺口；
- checkpoint 与统计量的隐私风险；
- 误用、自动化偏差和输出解释风险；
- 监控、回滚和撤销授权的方法。

`<required>`

## 13. 简历与公开展示审批

拟使用的简历表述：

`<required>`

证据审查：

- [ ] 每个数字已映射到第9节的固定证据。
- [ ] 明确区分学术研究、adapter parity、合成演示和工程 benchmark。
- [ ] 未声称临床部署、临床有效性、正式合规或生产 SLA。
- [ ] 授权范围明确允许该表述和展示方式。
- [ ] 批准人和日期已记录。

| 审批字段 | 内容 |
| --- | --- |
| 指标复核人 | 待填写 |
| 工件/隐私复核人 | 待填写 |
| 公开表述批准 | 未批准 |
| 日期 | 待填写 |

未完成本节时，公开材料只能陈述 adapter、manifest 和合成 fixture 的工程实现，不得引用本研究模型的效果数字。
